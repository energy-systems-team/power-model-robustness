"""
This file contains functions for estimating PV panel temperatures and transferring temperature data from another
dataframe.

Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen and Lauri Karttunen
"""

import math
from helpers import config


def add_estimated_panel_temperature(df, constant_a=-3.43, constant_b=-0.125, irradiance_col_name="poa", 
                                    estimated_variable='module_temp') ->float:
    """
    Estimate and add module (panel) temperature using the Sandia (King, 2004) model.

    The module temperature is computed as a function of plane-of-array irradiance,
    ambient temperature, and wind speed using an exponential empirical model.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
        - 'T' : ambient air temperature (°C)
        - 'wind' : wind speed (m/s)
        - irradiance_col_name : irradiance (W/m²)
    constant_a : float, optional
        Empirical coefficient (Sandia model). Default from Varjopuro et al.
    constant_b : float, optional
        Empirical coefficient (Sandia model). Default from Varjopuro et al.
    irradiance_col_name : str, optional
        Column name for irradiance (W/m²). Default is "poa".
    estimated_variable : str, optional
        Name of the output column. Default is "module_temp".

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added/updated module temperature column.

    Notes
    -----
    - Model form:
          T_module = G * exp(a + b * wind) + T_air
    - If required inputs are missing, the function prints a warning and
      returns the DataFrame unchanged.
    - Based on Sandia PV Array Performance Model (King, 2004).
    - Varjopuro et al. estimate the cell temperature directly from ambient temperature.
    - Default coefficients can be tuned (e.g., Varjopuro et al., 2025).

    References
    ----------
    King 2004 model
    D. King, J. Kratochvil, and W. Boyson,
    Photovoltaic Array Performance Model Vol. 8,
    PhD thesis (Sandia Naitional Laboratories, 2004).

    J. Varjopuro et al., 
    Computational simulation of perovskite and silicon solar panel operating temperatures in varying ambient conditions,
    Solar Energy Materials and Solar Cells 290 (2025) 113657,
    https://doi.org/10.1016/j.solmat.2025.113657
    """
    # checking that all required variables exist in df

    if "T" not in df.columns:
        print("No air temperature variable in given dataframe")
        print("Aborting")
        return df

    if "wind" not in df.columns:
        print("No wind speed variable in given dataframe")
        print("Aborting")
        return df

    if irradiance_col_name not in df.columns:
        print(f"no reflection corrected poa value in df '{irradiance_col_name}'")
        print("Aborting")
        return df
    
    absorbed_radiation = df[irradiance_col_name]
    wind = df["wind"]
    module_elevation = config.module_elevation
    air_temperature = df["T"]

    # wind is sometimes given as west/east components

    # wind speed at model elevation, assumes 0 speed at ground, wind speed vector len at 2m and forms a
    # curve which describes the wind speed transition from 0 to 10m wind speed to higher
    #wind_speed = (module_elevation / 10) ** 0.1429 * wind
    wind_speed = wind
    module_temperature = absorbed_radiation * math.e ** (constant_a + constant_b * wind_speed) + air_temperature
    df[estimated_variable] = module_temperature

    return df


def add_estimated_cell_temperature(df, deltaT=1, temperature_name='module_temp', irradiance_col_name='poa_rc') ->float:
    """
    Estimate and add cell temperature from module temperature and irradiance.

    The cell temperature is calculated by adding a temperature rise proportional
    to irradiance to the module temperature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing module temperature and irradiance.
    deltaT : float, optional
        Temperature rise coefficient (°C per kW/m²). Default is 1.
    temperature_name : str, optional
        Column name for module temperature. Default is "module_temp".
    irradiance_col_name : str, optional
        Column name for irradiance (W/m²). Default is "poa_rc".

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added 'cell_temp' column.

    Notes
    -----
    - Model form:
          T_cell = T_module + (G / 1000) * deltaT
    - Based on Sandia PV temperature modeling approach.
    """

    absorbed_radiation = df[irradiance_col_name]
    module_temp = df[temperature_name]

    cell_temperature = module_temp + absorbed_radiation / 1000 * deltaT

    df["cell_temp"] = cell_temperature

    return df


def add_panel_temperature_from_cell_temperature(df, deltaT=1, cell_temp_name='cell_temp', irradiance_col_name='poa_rc'):
    """
    Estimate module (panel) temperature from cell temperature.

    This is the inverse operation of the simplified Sandia temperature model,
    subtracting the irradiance-dependent temperature rise from cell temperature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing cell temperature and irradiance.
    deltaT : float, optional
        Temperature rise coefficient (°C per kW/m²). Default is 1.
    cell_temp_name : str, optional
        Column name for cell temperature. Default is "cell_temp".
    irradiance_col_name : str, optional
        Column name for irradiance (W/m²). Default is "poa_rc".

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added 'module_temp' column.

    Notes
    -----
    - Model form:
          T_module = T_cell - (G / 1000) * deltaT
    - Used when cell temperature is known and module temperature is required.
    """

    module_temp = df[cell_temp_name] - df[irradiance_col_name] / 1000 * deltaT

    df["module_temp"] = module_temp

    return df


