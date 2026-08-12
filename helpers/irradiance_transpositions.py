"""
Irradiance transposition functions. Used for transforming different solar irradiance components to panel
projected irradiance components.

Terminology:
POA: Plane of array irradiance, the total amount of radiation which reaches the panel surface at a given time. This is
the sum of direct beam component, sky diffused component, and ground reflected component.
POA = "poa_beam" + "poa_diffused" + "poa_ground"

Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen
"""

import math
import numpy as np
import pandas as pd
import pvlib.irradiance
import helpers.astronomical_calculations as astronomical_calculations
from helpers import config


def irradiance_df_to_poa_df(irradiance_df:pd.DataFrame)-> pd.DataFrame:
    """
    Project irradiance components from horizontal plane to plane-of-array (POA).

    This function converts Global Horizontal Irradiance (GHI), Direct Normal
    Irradiance (DNI), and Diffuse Horizontal Irradiance (DHI) into their
    corresponding plane-of-array (POA) components:
      - Beam (direct) component
      - Sky diffuse component
      - Ground-reflected component

    The POA components are calculated using internal transposition models
    (geometry-based or more advanced formulations depending on implementation).
    If albedo is available, it is used for ground-reflected irradiance;
    otherwise, a default value is assumed.

    Parameters
    ----------
    irradiance_df : pandas.DataFrame
        Input DataFrame indexed by timestamps and containing at least:
          - 'ghi' : Global Horizontal Irradiance
          - 'dni' : Direct Normal Irradiance
          - 'dhi' : Diffuse Horizontal Irradiance
        Optionally:
          - 'albedo' : Surface reflectance for ground-reflected component

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns:
          - 'poa_beam'     : Beam irradiance on the plane of array
          - 'poa_diffused' : Sky diffuse irradiance on the plane of array
          - 'poa_ground'   : Ground-reflected irradiance
          - 'poa_comp'     : Total POA irradiance (sum of components),
                             added if not already present

    Notes
    -----
    - The function modifies the input DataFrame in place and also returns it.
    - The total POA irradiance ('poa_comp') is calculated only if it does not
      already exist.
    - The accuracy of the projections depends on the underlying transposition
      models and assumptions (e.g., tilt, orientation, albedo).
    """

    irradiance_df["poa_beam"] = __calculate_beam_component(irradiance_df["dni"], irradiance_df.index)
    irradiance_df["poa_diffused"] = __calculate_sky_diffused_component(irradiance_df.index, irradiance_df["dhi"], irradiance_df["dni"])

    if "albedo" in irradiance_df.columns:
        print("albedo in df, using it to calculate ground reflected component")
        irradiance_df["poa_ground"] = __calculate_ground_reflected_component(irradiance_df["ghi"], irradiance_df["albedo"])
    else:
        irradiance_df["poa_ground"] = __calculate_ground_reflected_component(irradiance_df["ghi"])

    # adding the sum of projections to df as poa if it's not measured in the original data
    if "poa_comp" not in irradiance_df.columns:
        irradiance_df["poa_comp"] = irradiance_df["poa_diffused"] + irradiance_df["poa_beam"] + irradiance_df["poa_ground"]

    return irradiance_df


def __calculate_beam_component(dni, dt)-> float:
    """
    Beam component of the radiation. Based on https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/plane-of-array-poa-irradiance
    /calculating-poa-irradiance/poa-beam/

    Parameters
    ----------
    dni : pandas.Series or numpy.ndarray
        Direct Normal Irradiance (W/m²), representing the direct component
        of solar radiation perpendicular to the sun’s rays.
    dt : pandas.DatetimeIndex or array-like
        Timestamps corresponding to the irradiance values, used to compute
        the solar angle of incidence.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Beam (direct) irradiance component on the plane of array (W/m²).


    Notes
    -----
    - This version of the function is fairly well optimized.
    """

    angle_of_incidence = astronomical_calculations.get_solar_angle_of_incidence_fast(dt)

    return np.abs(dni * np.cos(np.radians(angle_of_incidence)))


def __calculate_sky_diffused_component(time, dhi, dni)-> float:
    """
    Sky-diffused component of radiation. Alternative dhi model,
    Calculated internally by pvlib, pvlib documentation at:
    https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.perez.html

    Parameters
    ----------
    time : pandas.DatetimeIndex or array-like
        Timestamps corresponding to the irradiance values, used to compute
        solar position, extraterrestrial radiation, and air mass.
    dhi : pandas.Series or numpy.ndarray
        Diffuse Horizontal Irradiance (W/m²).
    dni : pandas.Series or numpy.ndarray
        Direct Normal Irradiance (W/m²), used to characterize sky conditions
        in the Perez model.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Sky-diffuse irradiance on the plane of array (W/m²).

    """

    # function parameters
    dni_extra = pvlib.irradiance.get_extra_radiation(time)

    # this should take sun-earth distance variation into account
    # empirical constant 1366.1 should work nearly as well

    # installation angles
    surface_tilt = config.tilt
    surface_azimuth = config.azimuth

    # sun angles
    solar_azimuth, solar_zenith = astronomical_calculations.get_solar_azimuth_zenit_fast(time)

    # air mass
    airmass = astronomical_calculations.get_air_mass_fast(time)

    return pvlib.irradiance.perez(surface_tilt, surface_azimuth,dhi, dni, dni_extra,  solar_zenith, solar_azimuth, airmass, return_components=False) 


def __calculate_ground_reflected_component(ghi, albedo=config.albedo)-> float:
    """
    Ground-reflected component of the radiation. Equation from
    https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-ground-reflected/

    Uses ground albedo and panel angles to estimate how much of the sunlight per 1m² of ground is radiated towards solar
    panel surfaces.
    
    Parameters
    ----------
    ghi : pandas.Series or numpy.ndarray
        Global Horizontal Irradiance (W/m²), representing total irradiance
        incident on a horizontal surface.
    albedo : float or pandas.Series, optional
        Ground reflectance (unitless). Default is taken from `config.albedo`.

    Returns
    -------
    pandas.Series or numpy.ndarray
        Ground-reflected irradiance component on the plane of array (W/m²).
    """
    step1 = (1.0-math.cos(np.radians(config.tilt)))/2
    step2 = ghi*albedo * step1
   
    return step2


def add_albedo_from_snow_depth(df):
    """
    Add an albedo column to the dataset based on snow depth information.

    The function assigns surface albedo values depending on the presence of snow:
      - Default albedo is taken from `config.albedo` (used for snow-free conditions)
      - Elevated albedo (0.7) is assigned during snow periods

    Snow periods are identified using the 'snow_ground' column by:
      1. Forward-filling snow depth values up to one day (1440 minutes)
      2. Creating a binary indicator for snow presence (> 0)
      3. Confirming persistent snow by checking both current and next-day values
      4. Assigning higher albedo where snow is present on consecutive days

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that may contain a 'snow_ground' column representing
        snow depth measurements.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an added 'albedo' column.

    Notes
    -----
    - The function modifies the input DataFrame in place and also returns it.
    - If 'snow_ground' is not present, a constant albedo (config.albedo) is applied.
    - The snow detection logic is designed to reduce noise by requiring
      consistent snow presence over consecutive days.
    - The value 0.7 is used as a typical albedo for snow-covered surfaces.
    """

    if "snow_ground" in df.columns:
         # Step 1: create daily snow indicator (snow present if > 0)
        daily_snow_filled = df['snow_ground'].ffill(limit=1440)  # 1440 minutes in a day
        daily_snow_bool = (daily_snow_filled > 0).astype(int)

        # Step 2: shift by one day to check both current & next measurements
        snow_next_bool = daily_snow_bool.shift(freq=pd.Timedelta(days=-1)).fillna(0).astype(int)

        # Step 3: condition for confirmed snow periods
        snow_periods = (daily_snow_bool.astype(bool) & snow_next_bool.astype(bool))

        # Step 4: assign albedo = 0.7 where both days show snow
        df["albedo"] = config.albedo # Default albedo for non-snow days
        df.loc[snow_periods.reindex(df.index, method="ffill").fillna(0).astype(bool), "albedo"] = 0.7
    else:
        df["albedo"] = config.albedo
        print("No snow_ground on columns. Only default albedo added to columns.")
        
    print(f"albedo in irradiance transposition: {df["albedo"].unique()}")

    return df