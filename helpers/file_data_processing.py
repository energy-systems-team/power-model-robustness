from datetime import timedelta
import pytz
from helpers import config
from helpers import irradiance_decompositions
from helpers import irradiance_transpositions
from helpers import reflection_estimator
from helpers import panel_temperature_estimator
from helpers import preprocessing
from helpers import filters

import pandas as pd
import numpy as np


def __convert_to_utc(local_times):  
    """
    Convert Finnish local times (Europe/Helsinki) to naive UTC timestamps,
    handling daylight saving time (DST) issues.

    The function attempts to localize each datetime individually to detect
    invalid or ambiguous timestamps caused by DST transitions and replaces
    them with NaN.

    Parameters
    ----------
    local_times : list, numpy.ndarray, or pandas.Series
        Iterable of datetime-like objects representing Finnish local time.
        The input may include DST-affected timestamps.

    Returns
    -------
    pandas.Series
        Series of UTC timestamps (timezone-naive). Invalid timestamps are
        replaced with NaN.
    """
    # Convert the input to a Pandas Series if it's not already
    local_times_series = pd.Series(local_times)
    valid_times = []
    invalid_times = []

    for dt in local_times_series:
        try:
            # Attempt to localize each datetime individually
            localized = pd.Series([dt]).dt.tz_localize('Europe/Helsinki', ambiguous='NaT')
            if pd.isna(localized[0]):
                    raise pytz.NonExistentTimeError(f"Non-existent time: {dt}")
            valid_times.append(dt)
        except pytz.NonExistentTimeError as e:
            print(f"Invalid time: {dt}")
            valid_times.append(np.nan)
            invalid_times.append(dt)

    # Convert valid times to UTC
    valid_series = pd.Series(valid_times)
    finland_tz = valid_series.dt.tz_localize('Europe/Helsinki', ambiguous='NaT')
    utc_times = finland_tz.dt.tz_convert('UTC')
    utc_times = utc_times.dt.tz_localize(None)

    return utc_times


def merge_weather_and_pv_data(data_path, save_path=""):
    """
    Merge weather, PV production, and snow depth datasets into a single
    time-indexed DataFrame with UTC timestamps.

    The function reads multiple weather data files, a PV production file,
    and a snow depth file (all currently hard-coded for the TUAS Turku system),
    converts and aligns their timestamps to UTC, cleans duplicates and invalid
    entries, and combines them into a unified dataset.

    Processing steps include:
      - Weather data:
          * Read multiple CSV files
          * Parse local Finnish timestamps and convert them to UTC using
            __convert_to_utc (handles DST issues)
          * Remove invalid timestamps (NaN) and duplicates
          * Round timestamps to the nearest minute (floor)
      - PV data:
          * Read CSV and convert timestamps to UTC (fixed offset of -2 hours)
          * Remove duplicates
          * Compute DC power from current and voltage measurements
      - Snow depth data:
          * Construct UTC timestamps from separate date/time columns
          * Rename and retain snow depth variable

    Parameters
    ----------
    data_path : str
        Path to the directory containing all required input CSV files.
        File names are currently hard-coded.
    save_path : str, optional
        If provided, the merged DataFrame is saved to this path as a CSV
        file (semicolon-separated). Default is "" (no file is saved).

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame indexed by UTC timestamps, containing weather,
        PV, and snow depth data aligned in time.

    Notes
    -----
    - This function is currently tailored to TUAS Turku dataset structure
      and file naming conventions.
    - All timestamps in the output are timezone-naive UTC.
    - Missing values may exist where datasets do not overlap in time.
    """

    weather_files = ["weatherData_18-21_raw.csv"]
    df_weather_list = []

   # This processes the weather data from the different files. The times are converted to UTC.
    for weather_file in weather_files:
        df_weather_temp = pd.read_csv(data_path + "/" + weather_file, sep=",|;", decimal=".", engine="python")
        df_weather_temp["datetime"] = pd.to_datetime(df_weather_temp["datetime"].str.replace('"', ''))
        df_weather_temp["utctime"] = __convert_to_utc(df_weather_temp["datetime"])
        df_weather_temp.drop(columns=("datetime"), inplace=True)

        df_weather_list.append(df_weather_temp)

    df_weather = pd.concat(df_weather_list)

    df_weather.dropna(subset="utctime", inplace=True)
    df_weather = df_weather[~df_weather["utctime"].duplicated(keep='first')]

    # Round the times (using floor), and remove duplicates
    df_weather["utctime"] = df_weather["utctime"].dt.floor('min')
    df_weather = df_weather[~df_weather["utctime"].duplicated(keep='first')]

    df_weather = df_weather.set_index("utctime")

    df_pv = pd.read_csv(data_path + "/Etela_fx_DST_17_23.csv", sep=",")
    df_pv["utctime"] = pd.to_datetime(df_pv["datetime"]) - timedelta(hours=2)
    df_pv.drop(columns="datetime", inplace=True)
    df_pv = df_pv[~df_pv["utctime"].duplicated(keep='first')]
    df_pv = df_pv.set_index("utctime")

    df_pv["dc_power"] = df_pv["Idc_MPP1"] * df_pv["Vdc_MPP1"]

    df_snow_depth = pd.read_csv(data_path + "/Turku Artukainen_ 1.1.2018 - 31.12.2020_snow_depth_raw.csv")

    # Combine into one datetime
    df_snow_depth["utctime"] = pd.to_datetime(
        df_snow_depth["Vuosi"].astype(str) + "-" +
        df_snow_depth["Kuukausi"].astype(str).str.zfill(2) + "-" +
        df_snow_depth["Päivä"].astype(str).str.zfill(2) + " " +
        df_snow_depth["Aika [UTC]"]
    )

    df_snow_depth["snow_ground"] = df_snow_depth["Lumensyvyys [cm]"]
    df_snow_depth.drop(columns=["Havaintoasema", "Vuosi", "Kuukausi", "Päivä", "Aika [UTC]", "Lumensyvyys [cm]"], inplace=True)
    df_snow_depth = df_snow_depth.set_index("utctime")

    df = pd.concat([df_pv, 
                    df_weather, 
                    df_snow_depth
                    ], axis=1)

    if save_path != "":
        df.to_csv(save_path, sep=";")

    return df


def get_file_data(data=None, keep_only_necessary_columns=True):
    """
    Load, preprocess, enrich, filter, and return PV system data.

    The function performs a full preprocessing pipeline for PV and weather data.
    It can either read data from disk (based on configuration) or process a
    provided DataFrame. The pipeline unifies timestamps, imputes missing
    values, derives irradiance components, estimates temperatures, applies
    configurable data filtering, and optionally reduces the dataset to a
    standardized set of variables.

    Main processing steps:
      - Input handling:
          * Read CSV file if `data` is None using parameters from `config`
          * Rename columns and set UTC time index
      - Data preparation:
          * Impute missing values to consistent temporal resolution
          * Add albedo (if missing) based on snow depth
          * Merge duplicate irradiance measurements (e.g., multiple GHI sensors)
      - Irradiance modeling:
          * Decompose GHI into DNI and DHI if not available
          * Apply DNI correction
          * Transpose irradiance to plane-of-array (POA)
          * Apply reflection corrections and compute absorbed irradiance
      - Temperature estimation:
          * Estimate module and/or cell temperature if not measured
          * Optionally use Varjopuro coefficients for alternative modeling
      - Data cleaning:
          * Remove rows with missing key variables (e.g., power, irradiance)
      - Filtering:
          * Apply configurable filters (IEC, threshold, daytime, QC, snow, etc.)
          * Optionally perform rolling or RANSAC-based outlier removal
      - Post-processing:
          * Select relevant columns (optional)
          * Remove remaining NaNs (if column filtering is enabled)
          * Add power corrected for performance loss ratio (PLR)
          * Generate filtering diagnostic plots

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Input DataFrame. If None, the data is read from file using parameters
        defined in the `config` module.
    keep_only_necessary_columns : bool, optional
        If True, the output will contain only a predefined subset of relevant
        columns. If False, all processed columns are retained. Default is True.

    Returns
    -------
    pandas.DataFrame
        Fully processed and filtered DataFrame indexed by UTC timestamps,
        containing PV production, irradiance components, temperatures,
        and auxiliary variables.

    Notes
    -----
    - The function heavily depends on external configuration (`config`) and
      helper modules (e.g., preprocessing, filters, irradiance_transpositions).
    - Filtering behavior is controlled via `config.filter_data`.
    - Output timestamps are timezone-aware (UTC).
    - A diagnostic filtering plot is always generated as a side effect.
    """

    if data is None:
        # Read the file
        data = pd.read_csv(config.data_path + "/" + config.read_file_name, 
                                sep=config.data_file_sep, 
                                header=config.header_length
                                )
    data.rename(columns=config.col_name_dict, inplace=True)
    print(data.columns)
    data.set_index('utctime', inplace=True)
    data.index = pd.to_datetime(data.index, utc=True)
    data["time"] = pd.to_datetime(data.index, utc=True)

    # Impute to same resolution
    data = preprocessing.impute_columns(data, 
                                        col_names=config.cols_to_impute[0], 
                                        resolutions=config.cols_to_impute[1])
    # Set the albedo value
    if "albedo" not in data.columns:
        print("No albedo value in columns. Adding it...")
        data = irradiance_transpositions.add_albedo_from_snow_depth(data)
    
    # Average GHI measurements, if multiple are present in data
    if {"ghi_1", "ghi_2"}.issubset(data.columns):
        print("Two GHI values in columns. Averaging them...")
        data["ghi"] = preprocessing.merge_measurements(data, ["ghi_1", "ghi_2"], 10)
        data.drop(columns=["ghi_1", "ghi_2"], inplace=True)
    
    # Decompose DNI and/or DHI
    if not {"dni", "dhi"}.issubset(data.columns):
        print("DNI or DHI not in columns. Decomposing them from GHI...")
        data = irradiance_decompositions.get_dni_and_dhi(data)
    else:
        # Run the DNI correction even if it's measured
        print("Correcting the measured DNI values...")
        data["dni"] = irradiance_decompositions.dni_correction(data["dni"], data.index)
        
    # Project irradiance components to plane of array
    if not {"poa_beam", "poa_diffused", "poa_ground"}.issubset(data.columns):
        print("No projected irradiance components in columns. Calculating them...")
        data = irradiance_transpositions.irradiance_df_to_poa_df(data)
    else:
        print("Irradiance components already in data!")

    # Simulate how much of irradiance components is absorbed
    if not {"poa_beam_rc", "poa_diffused_rc", "poa_ground_rc"}.issubset(data.columns): 
        print("No reflection corrected components in columns. Calculating them...")
        data = reflection_estimator.add_reflection_corrected_poa_components_to_df(data)
    else:
        print("Reflection corrected components already in data!")

    # Compute sum of reflection-corrected components:
    if "poa_comp_rc" not in data.columns:
        print("No poa_comp_rc in columns. Calculating it...")
        data = reflection_estimator.add_reflection_corrected_poa_to_df(data)
    else:
        print("Reflection corrected POA already in data!")
    
    if "poa" in data.columns:
        poa_column = "poa"
    else:
        poa_column = "poa_comp"

    if config.use_varjopuro_coefs:
        # Varjopuro et al. calculated a and b coefficients for Sandia model, which gives directly the cell temperature.
        # This modifies the cell and panel temperature calculations so that first we calculate cell and then panel temperature.
        print("Using Varjopuro coefs")

        if "cell_temp" not in data.columns:
            print("No cell_temp in columns. Calculating it...")
            # We use the Sandia model for panel temperature, but name the resulting variable as cell_temp due to the use of Varjopuro coefs.
            data = panel_temperature_estimator.add_estimated_panel_temperature(data, irradiance_col_name=poa_column + "_rc",
                                                                                        estimated_variable='cell_temp')
        else:
            print("Cell temperature already in data!")
        
        if "module_temp" not in data.columns:
            if {"module_temp_1", "module_temp_2"}.issubset(data.columns):
                print("Two module_temp values in columns. Averaging them...")
                data["module_temp"] = preprocessing.merge_measurements(data, ["module_temp_1", "module_temp_2"], 10)
                data.drop(columns=["module_temp_1", "module_temp_2"], inplace=True)
            else:
                print("No module_temp in columns. Calculating it...")
                # We calculate the panel temperature with cell temp
                data = panel_temperature_estimator.add_panel_temperature_from_cell_temperature(data, cell_temp_name='cell_temp', 
                                                                                                       irradiance_col_name=poa_column + "_rc")
        else:
            print("Module temperature already in data!")

    else:
        # Estimate panel temperature based on wind speed, air temperature and absorbed radiation if it's not measured.
        if "module_temp" not in data.columns:
            if {"module_temp_1", "module_temp_2"}.issubset(data.columns):
                print("Two module_temp values in columns. Averaging them...")
                data["module_temp"] = preprocessing.merge_measurements(data, ["module_temp_1", "module_temp_2"], 10)
                data.drop(columns=["module_temp_1", "module_temp_2"], inplace=True)
            else:
                print("No module_temp in columns. Calculating it...")
                data = panel_temperature_estimator.add_estimated_panel_temperature(data, constant_a=-3.47, constant_b=-0.0594,
                                                                                           irradiance_col_name=poa_column + "_rc",
                                                                                           estimated_variable='module_temp')
        else:
            print("Module temperature already in data!")
        
        # Estimate cell temperature based on module temperature, and absorbed radiation if it's not measured
        if "cell_temp" not in data.columns:
            print("No cell_temp in columns. Calculating it...")
            data = panel_temperature_estimator.add_estimated_cell_temperature(data, 1, irradiance_col_name=poa_column + "_rc")
        else:
            print("Cell temperature already in data!")

        
    data = data.loc[~(data["power"].isna() | data[poa_column].isna()), :].copy()

    filter_dict_with_vals = config.filter_data
    print(f"filter dict {filter_dict_with_vals}")
    data_filtered = data.copy()
    filter_list = []
    filter_dict_with_masks = {}
    
    if 'iec' in filter_dict_with_vals:
        print("iec in filter_dict")
        iec_mask = filters.IEC(data, config.rated_power*1000, irradiance_col_name=poa_column)
        filter_list.append(iec_mask)
        filter_dict_with_masks['iec'] = iec_mask

    if 'threshold' in filter_dict_with_vals:
        print("threshold in filter_dict")
        if not isinstance(filter_dict_with_vals['threshold'], list):
            filter_dict_with_vals['threshold'] = [filter_dict_with_vals['threshold']]

        for parameters in filter_dict_with_vals['threshold']:
            if 'negate' in parameters:
                negate = parameters['negate']
            else:
                negate = False
                
            threshold_mask = filters.threshold(data, parameters['name'], lowers=parameters['lower'], uppers=parameters['upper'], negate=negate)
            filter_list.append(threshold_mask)
            filter_dict_with_masks['threshold'] = threshold_mask

    if 'daytime' in filter_dict_with_vals:
        print("daytime in filter_dict")
        daytime_mask = filters.daytime(data, irradiance_col_name=poa_column)
        filter_list.append(daytime_mask)
        filter_dict_with_masks['daytime'] = daytime_mask

    if 'qc' in filter_dict_with_vals:
        print("qc in filter_dict")
        qc_mask = filters.quality_control(data)
        filter_list.append(qc_mask)
        filter_dict_with_masks['qc'] = qc_mask

    if 'viss_day' in filter_dict_with_vals:
        print("viss_day in filter_dict")
        viss_day_mask = filters.viss_day(data)
        filter_list.append(viss_day_mask)
        filter_dict_with_masks['viss_day'] = viss_day_mask

    if 'viss_instant' in filter_dict_with_vals:
        print("viss_instant in filter_dict")
        viss_insta_mask = filters.viss_instant(data)
        filter_list.append(viss_insta_mask)
        filter_dict_with_masks['viss_instant'] = viss_insta_mask

    if 'snow' in filter_dict_with_vals:
        print("snow in filter_dict")
        snow_mask = filters.snow(data)
        filter_list.append(snow_mask)
        filter_dict_with_masks['snow'] = snow_mask

    if len(filter_list) != 0:
        combined_mask = np.logical_and.reduce(filter_list)    
        data_filtered = data_filtered[combined_mask]

    if 'outlier' in filter_dict_with_vals:
        print("outlier in filter_dict")
        parameters = filter_dict_with_vals['outlier']
        outlier_mask = filters.rolling_outlier_filter(data_filtered, 
                                                      parameters['window_size'], 
                                                      parameters['window_width'], 
                                                      irradiance_col_name=poa_column + "_rc")
        data_filtered = data_filtered[outlier_mask]
        filter_dict_with_masks['outlier'] = outlier_mask
    
    if 'cut_outliers' in filter_dict_with_vals:
        print("cut outliers in filter_dict")
        parameters = filter_dict_with_vals['cut_outliers']

        if ('lower' in parameters) & ('upper' in parameters) & ('filter_threshold' in parameters): 
            print(parameters['lower'], parameters['upper'], parameters['filter_threshold'])
            cut_outliers_mask = filters.cut_outliers_ransac(data_filtered, 
                                                            irradiance_col_name=poa_column + "_rc",
                                                            lower=parameters['lower'], 
                                                            upper=parameters['upper'],
                                                            filter_threshold=parameters['filter_threshold'])
        else:
            cut_outliers_mask = filters.cut_outliers_ransac(data_filtered, 
                                                            irradiance_col_name=poa_column + "_rc")

        data_filtered = data_filtered[cut_outliers_mask]
        filter_dict_with_masks['cut_outliers'] = cut_outliers_mask


    full_list_of_cols = ["time",
            "ghi", "dhi", "dni", "poa",
            "poa_beam", "poa_diffused", "poa_ground",
            "poa_beam_rc", "poa_diffused_rc", "poa_ground_rc",
            "poa_comp", "poa_comp_rc", "poa_rc",
            "module_temp", "cell_temp",
            "wind", "T", "albedo", "snow", "dataQC",
            "power", "pv_inv_out"]
    found_cols = [col_name for col_name in full_list_of_cols if col_name in data.columns]
    data = data[found_cols].copy()

    # Build kwargs automatically from filter_dict
    plot_kwargs = {
        f"{name}_mask": mask        # masks is the list stored in filter_dict
        for name, mask in filter_dict_with_masks.items()
    }


    if keep_only_necessary_columns:        
        # Keep only necessary columns
        print("Keeping only necessary columns...")
        full_list_of_cols = ["time",
                    "ghi", "dhi", "dni", "poa",
                    "poa_beam", "poa_diffused", "poa_ground",
                    "poa_beam_rc", "poa_diffused_rc", "poa_ground_rc",
                    "poa_comp", "poa_comp_rc", "poa_rc",
                    "module_temp", "cell_temp",
                    "wind", "T", "albedo",
                    "power", "pv_inv_out"]
        found_cols = [col_name for col_name in full_list_of_cols if col_name in data_filtered.columns]
        data_filtered = data_filtered[found_cols]
        data_filtered.dropna(axis=0, how='any', inplace=True)

    # Add column which has the power without the PLR losses.
    data_filtered = preprocessing.calculate_power_withour_plr_losses(data_filtered, config.plr_value, min(data_filtered.index))

    print(data_filtered.describe().T)
    print(f"Start date: {min(data_filtered.index)}, end date: {max(data_filtered.index)}")
    
    return data_filtered