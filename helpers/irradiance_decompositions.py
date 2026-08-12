from pvlib import irradiance
from helpers import astronomical_calculations
import numpy as np


def dni_correction(dni, utctimes, aoi=None, solar_zenith=None):
    """
    Apply quality control correction to Direct Normal Irradiance (DNI)
    measurements using the model proposed by Böök et al.
    (DOI: 10.1016/j.solener.2020.04.068).

    The correction limits DNI values based on the angle of incidence (AOI)
    to remove physically unrealistic high values, and sets DNI to zero for
    very low solar elevation angles.

    Parameters
    ----------
    dni : pandas.Series or numpy.ndarray
        Measured or modelled DNI values.
    utctimes : pandas.DatetimeIndex or array-like
        Corresponding UTC timestamps for the DNI values.
    aoi : pandas.Series or numpy.ndarray, optional
        Angle of incidence (degrees). If not provided, it is calculated
        from `utctimes`.
    solar_zenith : pandas.Series or numpy.ndarray, optional
        Solar zenith angle (degrees). If not provided, it is calculated
        from `utctimes`.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Corrected DNI values where:
          - DNI is capped by an AOI-dependent upper limit.
          - DNI is set to 0 when solar elevation < 0.5°.

    Notes
    -----
    - The correction uses the empirical limit:
          dni_limit = a * exp(b * AOI) + c
      where a = -838, b = -0.112, c = 951.
    - Solar elevation is computed as (90° - solar_zenith).
    """

    if aoi is None:
        aoi = astronomical_calculations.get_solar_angle_of_incidence_fast(utctimes)
    if solar_zenith is None:
        solar_azimuth, solar_zenith = astronomical_calculations.get_solar_azimuth_zenit_fast(utctimes)

    # DNI correction from Böök DOI: 10.1016/j.solener.2020.04.068
    a = -838
    b = -0.112
    c = 951

    dni_qc_limit = a * np.exp(b * aoi) + c
    dni_qc = np.minimum(dni, dni_qc_limit)

    # Set DNI values with low solar elevation (under 0.5 degrees) to zero.
    # This is recommended by Böök DOI: 10.1016/j.solener.2020.04.068
    solar_elevation = 90 - solar_zenith
    dni_qc[solar_elevation < 0.5] = 0

    return dni_qc


def get_dni_and_dhi(data, overwrite=False):
    """
    Estimate or populate Direct Normal Irradiance (DNI) and Diffuse Horizontal
    Irradiance (DHI) for a dataset.

    DNI and DHI are computed from Global Horizontal Irradiance (GHI) using the
    ERBS decomposition model. The resulting DNI values are additionally quality-
    controlled using the empirical correction proposed by Böök et al.
    (DOI: 10.1016/j.solener.2020.04.068).

    If DNI and/or DHI already exist in the input DataFrame, they are preserved
    unless `overwrite=True`.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame indexed by UTC timestamps. Must contain a 'ghi' column.
        May optionally include 'dni' and/or 'dhi' columns.
    overwrite : bool, optional
        If True, existing 'dni' and 'dhi' columns will be overwritten with
        newly computed values. If False (default), only missing columns are added.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with 'dni' and 'dhi' columns ensured:
          - DNI is derived using the ERBS model and corrected with the Böök method.
          - DHI is derived using the ERBS model.

    Notes
    -----
    - Solar position (zenith, azimuth) and angle of incidence (AOI) are computed
      internally from the DataFrame index.
    - The ERBS model is used for irradiance decomposition from GHI.
    - DNI correction limits unrealistic values and removes low solar elevation
      contributions (< 0.5°).
    - The function modifies the input DataFrame in place and also returns it.
    """

    utctimes = data.index

    solar_azimuth, solar_zenith = astronomical_calculations.get_solar_azimuth_zenit_fast(utctimes)
    aoi = astronomical_calculations.get_solar_angle_of_incidence_fast(utctimes)

    erbs_data = irradiance.erbs(ghi=data["ghi"], 
                                  zenith=solar_zenith,
                                  datetime_or_doy=utctimes)
    dni = erbs_data["dni"]
    dhi = erbs_data["dhi"]

    dni_qc = dni_correction(dni, utctimes, aoi=aoi, solar_zenith=solar_zenith) 

    if overwrite:
        data["dni"] = dni_qc
        data["dhi"] = dhi 
    else:
        if "dni" not in data.columns:
            data["dni"] = dni_qc
        if "dhi" not in data.columns:
            data["dhi"] = dhi

    return data




