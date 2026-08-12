"""
Common data filters for PV data processing. The data filters include
- nighttime filter
- clear-sky filter
- outlier filter
- variable cutoff filter
- IEC filter:
        - Irradiance [-6, 1500] W/m^2
        - Ambient temperature [-30,50] degC
        - Wind speed [0,31] m/s
        - AC power [-0.01Pnom,1.02Pnom]

Author: Lauri Karttunen.
Edited by: Väinö Anttalainen.
"""


import pandas as pd
import numpy as np
import pvlib
from sklearn.linear_model import RANSACRegressor


def IEC(data, p_rated, irradiance_col_name="poa"):
    """
    Generate a boolean mask based on IEC-inspired data quality thresholds
    for photovoltaic (PV) system measurements.

    The following filtering criteria are applied:
      - Irradiance:      -6 < irradiance < 1500 W/m²
      - Air temperature: -30 < T < 50 °C
      - Wind speed:      0 < wind (m/s)
      - AC power (if available):
            -0.01 * p_rated < pv_inv_out < 1.02 * p_rated

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing at least irradiance, temperature ('T'),
        and wind speed ('wind'). Optionally includes inverter output
        power ('pv_inv_out').
    p_rated : float
        Nominal rated power of the PV system (same unit as 'pv_inv_out').
    irradiance_col_name : str, optional
        Column name for irradiance values used in filtering.
        Default is "poa" (plane-of-array irradiance).

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates rows that satisfy all IEC
        filtering criteria.

    Notes
    -----
    - The thresholds are based on IEC-standard-inspired limits commonly
      used for PV data quality control, though they may be adapted for
      specific datasets.
    - Power filtering is only applied if 'pv_inv_out' exists in the input.
    - The function does not modify the input DataFrame.
    """

    
    mask = (data[irradiance_col_name] < 1500) & (data[irradiance_col_name] > -6) & (data['T'] > -30) & (data['T'] < 50) & \
           (data['wind'] > 0)
           
    if 'pv_inv_out' in data.columns:
        mask = mask & (data['pv_inv_out'] > -0.01 * p_rated) & (data['pv_inv_out'] < 1.02 * p_rated)

    return mask


def threshold(data, parameters: list, lowers=[0], uppers=[50000], negate=False): 
    """
    Generate a boolean mask by applying lower and upper threshold filters
    to one or more columns.

    Each specified parameter is filtered using the corresponding lower and
    upper bounds. All conditions are combined using a logical AND operation.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing the columns to be filtered.
    parameters : str or list of str
        Column name(s) to which the threshold filtering is applied.
    lowers : scalar or list of scalars, optional
        Lower bound(s) for each parameter. Default is 0.
    uppers : scalar or list of scalars, optional
        Upper bound(s) for each parameter. Default is 50000.
    negate : bool, optional
        If True, the resulting mask is inverted (i.e., values outside the
        specified thresholds are marked as True). Default is False.

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates rows that satisfy all threshold
        conditions (or their negation if `negate=True`).

    Notes
    -----
    - If a single parameter or bound is provided, it is automatically
      converted to a list.
    - Filtering uses pandas `Series.between()`, which includes both bounds.
    - All parameter conditions must be satisfied for a row to be True.
    - The function does not modify the input DataFrame.
    """

    if not isinstance(parameters, list):
        parameters = [parameters]
    if not isinstance(lowers, list):
        lowers = [lowers]
    if not isinstance(uppers, list):
        uppers = [uppers]

    mask = pd.Series(True, index=data.index)
    for parameter, lower, upper in zip(parameters, lowers, uppers):
        mask &= data[parameter].between(lower, upper)
    #mask = (lower <= data[parameter]) & (data[parameter] <= upper)
    
    if negate:
        return ~mask
    return mask


def nighttime(data, irradiance_col_name="poa"):
    """
    Identify night-time data points based on irradiance threshold.

    A data point is considered night-time if the irradiance value is
    below 5 W/m².

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing an irradiance column.
    irradiance_col_name : str, optional
        Name of the irradiance column used for filtering.
        Default is "poa" (plane-of-array irradiance).

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates night-time data points
        (irradiance < 5 W/m²).
    """
    
    mask = data[irradiance_col_name] < 5
    return mask


def daytime(data, irradiance_col_name="poa"):
    """
    Identify daytime data points based on irradiance threshold.

    A data point is considered daytime if the irradiance value is
    greater than or equal to 5 W/m² (i.e., the complement of the
    `nighttime` function).

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing an irradiance column.
    irradiance_col_name : str, optional
        Name of the irradiance column used for filtering.
        Default is "poa" (plane-of-array irradiance).

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates daytime data points
        (irradiance ≥ 5 W/m²).
    """

    return ~nighttime(data, irradiance_col_name)


def snow(data): 
    """
    Filter out data points affected by snow conditions.

    If a 'snow' column is present, only rows with zero snow depth
    (i.e., no snow present) are retained. If the column is not
    present, all data points are considered valid.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame that may contain a 'snow' column.

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates rows without snow influence.

    Notes
    -----
    - Assumes that 'snow' == 0 corresponds to snow-free conditions.
    - This filter is typically used to exclude measurements impacted
      by snow cover on PV panels or sensors.
    - If no snow data is available, no filtering is applied.
    """

    if "snow" in data.columns:
        mask = data["snow"] == 0
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask


def rolling_outlier_filter(data, window_size, window_width=2, irradiance_col_name="poa", datetime_col_name='utctime'):
    """
    Detect and filter power–irradiance outliers using a rolling window approach.

    The method evaluates the ratio of power to irradiance (power / irradiance)
    after sorting the data by irradiance. A rolling mean and standard deviation
    are computed over a specified window, and points that fall outside a
    configurable number of standard deviations from the rolling mean are
    flagged as outliers.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing at least 'power' and irradiance columns.
    window_size : int
        Number of data points in the rolling window used to compute statistics.
    window_width : float, optional
        Number of standard deviations used to define the acceptable range
        around the rolling mean. Default is 2.
    irradiance_col_name : str, optional
        Name of the irradiance column (e.g., "poa"). Default is "poa".
    datetime_col_name : str, optional
        Name of the datetime column used to restore chronological order
        after processing. Default is "utctime".

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates data points within the acceptable
        range (i.e., non-outliers), and False indicates outliers.

    Notes
    -----
    - The filtering is performed on the power-to-irradiance ratio to normalize
      power output relative to incoming irradiance.
    - Data is temporarily sorted by irradiance to enable the rolling-window
      filtering in irradiance space rather than time.
    - The DataFrame is restored to chronological order before returning.
    - The function temporarily adds a 'filter_mask' column but does not
      otherwise modify the input data structure.
    - Rolling statistics will produce NaN values at the edges, which may
      propagate into the mask.
    """

    # Sort data by irradiance to perform the rolling horizon
    data = data.sort_values(by=irradiance_col_name)

    p_poa = data['power'] / data[irradiance_col_name]

    # Set rolling window size
    window_size = window_size

    # Calculate rolling mean and standard deviation
    rolling_mean = p_poa.rolling(window=window_size).mean()
    rolling_sd = p_poa.rolling(window=window_size).std()

    # Define the upper and lower bounds
    upper_bound = rolling_mean + window_width * rolling_sd
    lower_bound = rolling_mean - window_width * rolling_sd

    p_poa_mask = (p_poa < upper_bound) & (p_poa > lower_bound)

    data['filter_mask'] = p_poa_mask
    data = data.sort_values(by=datetime_col_name)

    return data.filter_mask


def cut_outliers_ransac(data, irradiance_col_name="poa", lower=0, upper=10000, filter_threshold=10000):
    """
    Filter power–irradiance outliers using a RANSAC-fitted lower envelope.

    The method identifies the lower boundary of the power–irradiance relationship
    by:
      1. Restricting data to a specified irradiance range.
      2. Binning irradiance and extracting minimum power values per bin.
      3. Fitting a linear model to these lower-envelope points using RANSAC.
      4. Removing points that fall below the fitted lower bound for sufficiently
         high irradiance values.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing at least 'power' and irradiance columns.
    irradiance_col_name : str, optional
        Name of the irradiance column (e.g., "poa"). Default is "poa".
    lower : float, optional
        Lower bound of irradiance used to select the fitting range. Default is 0.
    upper : float, optional
        Upper bound of irradiance used to select the fitting range. Default is 10000.
    filter_threshold : float, optional
        Irradiance threshold above which the RANSAC-based filtering is applied.
        Below this threshold, all data points are retained. Default is 10000.
    lower_values : bool, optional
        Placeholder parameter (currently not used in implementation).
    upper_values : bool, optional
        Placeholder parameter (currently not used in implementation).

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates retained (non-outlier) points and
        False indicates outliers below the fitted lower envelope.

    Notes
    -----
    - The method focuses on removing low-power outliers (e.g., shading, snow,
      or system faults) relative to irradiance.
    - RANSAC is used to robustly estimate the lower boundary in the presence
      of noise and outliers.
    - The filtering is only enforced for irradiance values above
      `filter_threshold`; lower irradiance values are always kept.
    - The function does not modify the input DataFrame.
    """

    data_range = data.loc[(data[irradiance_col_name] >= lower) & (data[irradiance_col_name] <= upper), :]
    poa = data[irradiance_col_name]
    poa_range = data_range[irradiance_col_name]
    power = data["power"]
    power_range = data_range["power"]

    # Step 1: bin poa and take minimum power per bin
    bins = np.linspace(poa_range.min(), poa_range.max(), 40)
    digitized = np.digitize(poa_range, bins)
    poa_min = [poa_range[digitized == i].mean() for i in range(1, len(bins))]
    power_min = [power_range[digitized == i].min() for i in range(1, len(bins))]

    poa_min = np.array(poa_min)
    power_min = np.array(power_min)

    # Step 2: Fit RANSAC on the lower-envelope points
    X = poa_min.reshape(-1, 1)
    y = power_min
    nan_mask = ~np.isnan(X.ravel()) & ~np.isnan(y)
    ransac = RANSACRegressor(#base_estimator=LinearRegression(),
                             #min_samples=0.5,
                             #residual_threshold=20, 
                             random_state=10)
    ransac.fit(X[nan_mask], y[nan_mask])

    # --- Step 3: Predict line values for all poa ---
    y_pred_all = ransac.predict(poa.to_numpy().reshape(-1, 1))

    # --- Step 4: Filtering ---
    irr_threshold = filter_threshold  # <-- set your threshold here
    mask_keep = (poa <= irr_threshold) | (power >= y_pred_all)

    return mask_keep


def quality_control(data):
    """
    Apply quality control (QC) filtering based on a data QC flag for FMI's datasets.

    If a 'dataQC' column is present, only rows with a QC value of 1
    (indicating valid/accepted data) are retained. If the column is not
    present, all data points are considered valid.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame that may contain a 'dataQC' column.

    Returns
    -------
    pandas.Series
        Boolean mask where True indicates rows that pass quality control.

    Notes
    -----
    - Assumes that 'dataQC' == 1 represents valid data.
    - If no QC information is available, no filtering is applied.
    """

    if "dataQC" in data.columns:
        mask = data["dataQC"] == 1
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask



def viss_day(data):
    """
    NOTE: currently not used.
    """

    if "viss_day" in data.columns:
        mask = data['viss_day'] == '[0, 0, 0, 0]'
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask


def viss_instant(data):
    """
    NOTE: currently not used.
    """

    if "viss_instant" in data.columns:
        mask = ~data['viss_instant'].isin([2, 3, 4, 5, 6])
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask
    