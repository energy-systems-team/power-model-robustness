"""
Configuration file for PV system simulations.

This file contains installation-specific and generally static parameters
such as location, panel orientation, system size, and data handling settings.
These values are intended to remain constant during a simulation.

Typical contents:
- Geolocation (latitude, longitude, elevation)
- Panel geometry (tilt, azimuth)
- System characteristics (rated power, albedo)
- Data processing settings (resolution, column mappings)


Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen and Lauri Karttunen.
"""



##### Plotting and output settings #####
########################################

# Name used in plots and output file names
site_name = "output_example"

# Directory for saving outputs (plots, CSVs)
save_directory = "output/"

# Toggle saving processed data to CSV
save_csv = False

# Toggle printing processed data to console
console_print = False



########################################
##### Known installation parameters ####
########################################

# --- Geographic coordinates (degrees) ---

latitude_helsinki = 60.2044
longitude_helsinki = 24.9625

latitude_kuopio = 62.8919
longitude_kuopio = 27.6349

latitude_sodankyla = 67.367
longitude_sodankyla = 26.650

latitude_turku = 60.45
longitude_turku = 22.30

# --- Elevation above sea level (meters) ---
elevation_helsinki = 17
elevation_kuopio = 10
elevation_sodankyla = 4
elevation_turku = 9


# --- Panel tilt angles (degrees) ---
# 0 = horizontal, 90 = vertical
tilt_helsinki = 15
tilt_kuopio = 15
tilt_sodankyla_20 = 20
tilt_sodankyla_90 = 90
tilt_turku = 15


# --- Panel azimuth (degrees) ---
# Convention: 0 = north, 90 = east, clockwise rotation
azimuth_helsinki = 135
azimuth_kuopio = 217
azimuth_sodankyla = 123
azimuth_turku = 180


# --- Rated system power (kWp, DC) ---
rated_power_kuopio = 20.28
rated_power_helsinki = 21
rated_power_sodankyla = 0.260
rated_power_turku = 4.5



########################################
##### Default simulation parameters ####
########################################

# Active system configuration (can be overridden by set_params_* functions)
latitude = latitude_helsinki
longitude = longitude_helsinki

# panel angles
tilt = 15 # degrees. Panel flat on the roof would have tilt of 0. Wall mounted panels have tilt of 90.
azimuth = azimuth_helsinki # degrees, north is 0 degrees, east 90. Clockwise rotation

# rated installation power in kW, PV output at standard testing conditions
rated_power = rated_power_helsinki # unit kW

# ground albedo near solar panels, 0.25 is PVlib default. Has to be in range [0,1], typical values [0.1, 0.4]
# grass is 0.25, snow 0.8, worn asphalt 0.12. Values can be found from wikipedia https://en.wikipedia.org/wiki/Albedo
albedo = 0.151

# module elevation, measured from ground
module_elevation = 8 # unit meters

# dummy wind speed(meter per second) value, this will be used if wind speed from fmi open is not used
wind_speed = 2

# air temp in Celsius, this will be used if temp from fmi open is not used
air_temp = 20


########################################
##### General data/settings ###########
########################################

# Timezone (currently not fully utilized due to plotting limitations)
timezone = "UTC"

data_path = "./data"
read_file_name = ""
write_file_name = ""
data_file_sep = ";"
save_data_csv = False

# Header length tells how many rows of the file are metatext
header_length_helsinki = 67
header_length_kuopio = 67
header_length_sodankyla = 93


########################################
##### Installation parameter presets ###
########################################

# These helper functions configure all relevant parameters
# for a specific measurement site / dataset.
# They modify global variables for convenience.

# Common behavior:
# - Override system geometry and location
# - Set input/output filenames
# - Configure filtering rules
# - Define column mappings and preprocessing settings
# - Optionally enable data export

def set_params_helsinki(filter_scenario, read_file="FMI_Helsinki_PV.csv", write_file=""):
    global latitude, longitude, tilt, azimuth, rated_power, module_elevation, site_name, read_file_name, write_file_name, save_data_csv, filter_data, header_length, col_name_dict, cols_to_impute, plr_value, use_varjopuro_coefs
    latitude = latitude_helsinki
    longitude = longitude_helsinki
    tilt = tilt_helsinki
    azimuth = azimuth_helsinki
    rated_power = rated_power_helsinki
    module_elevation = elevation_helsinki
    site_name = "HEL"
    read_file_name = read_file
    write_file_name = write_file

    if filter_scenario == "scenario_1":
        filter_data = {
            'daytime' : {}
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_2":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_3":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }  
        plr_value = -0.33
    elif filter_scenario == "scenario_4":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }  
        plr_value = -0.81
    else:
        print("Filter scenario has to be 1, 2, or 3")

    # If file is the original data file, skip the first metadata rows
    if "FMI_Helsinki_PV.csv" in read_file:
        header_length = 67
    else:
        header_length = 0
    col_name_dict = col_name_dict_helsinki
    cols_to_impute = [['T', 'wind'], [10, 10]]
    use_varjopuro_coefs = True


def set_params_kuopio(filter_scenario, read_file="FMI_Kuopio_PV.csv", write_file=""):
    global latitude, longitude, tilt, azimuth, rated_power, module_elevation, site_name, read_file_name, write_file_name, save_data_csv, filter_data, header_length, col_name_dict, cols_to_impute, plr_value, use_varjopuro_coefs
    latitude = latitude_kuopio
    longitude = longitude_kuopio
    tilt = tilt_kuopio
    azimuth = azimuth_kuopio
    rated_power = rated_power_kuopio
    module_elevation = elevation_kuopio
    site_name = "KUO"
    read_file_name = read_file
    write_file_name = write_file
    
    if filter_scenario == "scenario_1":
        filter_data = {
            'daytime' : {}
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_2":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_3":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }
        plr_value = -0.33
    elif filter_scenario == "scenario_4":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }  
        plr_value = -0.41
    else:
        print("Filter scenario has to be 1, 2, or 3")

    # If file is the original data file, skip the first metadata rows
    if "FMI_Kuopio_PV.csv" in read_file:
        header_length = 67
    else:
        header_length = 0
    col_name_dict = col_name_dict_kuopio
    cols_to_impute = [['T', 'wind'], [10, 10]]
    use_varjopuro_coefs = True


def set_params_sodankyla_20(filter_scenario, read_file="FMI_Sodankyla_20deg_PV.csv", write_file=""):
    global latitude, longitude, tilt, azimuth, rated_power, module_elevation, site_name, read_file_name, write_file_name, save_data_csv, filter_data, header_length, col_name_dict, cols_to_impute, plr_value, use_varjopuro_coefs
    latitude = latitude_sodankyla
    longitude = longitude_sodankyla
    tilt = tilt_sodankyla_20
    azimuth = azimuth_sodankyla
    rated_power = rated_power_sodankyla
    module_elevation = elevation_sodankyla
    site_name = "SOT-20"
    #file_name = "FMI_Sodankyla_20deg_PV"
    read_file_name = read_file
    write_file_name = write_file
    
    if filter_scenario == "scenario_1":
        filter_data = {
                'daytime' : {}
            }   
        plr_value = -0.0
    elif filter_scenario == "scenario_2":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_3":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            'cut_outliers':{'lower':400, 'upper':800, 'filter_threshold':800}
            }
        plr_value = -0.33
    elif filter_scenario == "scenario_4":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            'cut_outliers':{'lower':400, 'upper':800, 'filter_threshold':800}
                    }
        plr_value = -0.86
    else:
        print("Filter scenario has to be 1, 2, or 3")

    # If file is the original data file, skip the first metadata rows
    if "FMI_Sodankyla_20deg_PV.csv" in read_file:
        header_length = 93
    else:
        header_length = 0
    col_name_dict = col_name_dict_sodankyla_20
    cols_to_impute = [['T', 'wind'], [10, 10]]
    #plr_value = -0.33
    use_varjopuro_coefs = True


def set_params_sodankyla_90(filter_scenario, read_file="FMI_Sodankyla_90deg_PV.csv", write_file=""):
    global latitude, longitude, tilt, azimuth, rated_power, module_elevation, site_name, read_file_name, write_file_name, save_data_csv, filter_data, header_length, col_name_dict, cols_to_impute, plr_value, use_varjopuro_coefs
    latitude = latitude_sodankyla
    longitude = longitude_sodankyla
    tilt = tilt_sodankyla_90
    azimuth = azimuth_sodankyla
    rated_power = rated_power_sodankyla
    module_elevation = elevation_sodankyla
    site_name = "SOT-90"
    #file_name = "FMI_Sodankyla_20deg_PV"
    read_file_name = read_file
    write_file_name = write_file
    
    if filter_scenario == "scenario_1":
        filter_data = {
                'daytime' : {}
            } 
        plr_value = -0.0
    elif filter_scenario == "scenario_2":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_3":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }
        plr_value = -0.33
    elif filter_scenario == "scenario_4":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':'power', 'lower':0, 'upper':20000}, 
            'qc':{}, 
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }
        plr_value = -1.35
    else:
        print("Filter scenario has to be 1, 2, or 3")

    # If file is the original data file, skip the first metadata rows
    if "FMI_Sodankyla_90deg_PV.csv" in read_file:
        header_length = 93
    else:
        header_length = 0
    col_name_dict = col_name_dict_sodankyla_90
    cols_to_impute = [['T', 'wind'], [10, 10]]
    #plr_value = -0.33
    use_varjopuro_coefs = True


def set_params_turku(filter_scenario, read_file="KTK_South_PV_weather_non_cleaned.csv", write_file=""):
    global latitude, longitude, tilt, azimuth, rated_power, module_elevation, site_name, read_file_name, write_file_name, save_data_csv, filter_data, header_length, col_name_dict, cols_to_impute, plr_value, use_varjopuro_coefs
    latitude = latitude_turku
    longitude = longitude_turku
    tilt = tilt_turku
    azimuth = azimuth_turku
    rated_power = rated_power_turku
    module_elevation = elevation_turku
    site_name = "TKU"
    #read_file_name = "TUAS_Turku_PV"
    read_file_name = read_file
    write_file_name = write_file
    
    if filter_scenario == "scenario_1":
        filter_data = {
            'daytime' : {}
        }
        plr_value = -0.0
    elif filter_scenario == "scenario_2":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':['power', 'poa_comp'], 'lower':[0, 100], 'upper':[100, 2000], 'negate':True}, 
            }
        plr_value = -0.0
    elif filter_scenario == "scenario_3":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':['power', 'poa_comp'], 'lower':[0, 100], 'upper':[100, 2000], 'negate':True}, 
            'qc':{},  
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }
        plr_value = -0.33
    elif filter_scenario == "scenario_4":
        filter_data = {
            'daytime':{}, 
            'iec':{}, 
            'threshold':{'name':['power', 'poa_comp'], 'lower':[0, 100], 'upper':[100, 2000], 'negate':True}, 
            'qc':{},  
            'snow':{}, 
            'outlier':{'window_size':800, 'window_width':2},
            }
        plr_value = -2.65   
    else:
        print("Filter scenario has to be 1, 2, or 3")

    header_length = 0
    col_name_dict = col_name_dict_turku
    cols_to_impute = [[], []]
    #plr_value = -0.33
    use_varjopuro_coefs = True


########################################
##### Column name mappings #############
########################################

# Mapping from raw data column names → standardized names used in code
# This allows consistent processing across different datasets.

# NOTE:
# - "ghi", "dni", "dhi" → irradiance components
# - "poa" → plane-of-array irradiance (if available)
# - "T" → ambient temperature
# - "wind" → wind speed
# - "power" → measured PV power

col_name_dict_helsinki = {
    # --- Metadata ---
    'fmisid': 'id',
    'stationname': 'stationname',
    'utctime': 'utctime',

    # --- Irradiance ---
    'GLOB_PT1M_AVG': 'ghi',
    'DIFF_PT1M_AVG': 'dhi',
    'DIR_PT1M_AVG': 'dni',
    'GLOBA_PT1M_AVG(:31)': 'poa',

    # --- Temperatures ---
    'TA_PT1M_AVG': 'T',
    'TTECH_PT1M_AVG(:31)': 't_roof',
    'TTECH_PT1M_AVG(:32)': 'module_temp_1',
    'TTECH_PT1M_AVG(:33)': 'module_temp_2',

    # --- Weather ---
    'P0_PT1M_AVG': 'pressure',
    'RH_PT1M_AVG': 'relative_humid',
    'CLA_PT1M_ACC': 'cloud_coverage',
    'WS_PT10M_AVG': 'wind',
    'WD_PT10M_AVG': 'wind_dir',
    'PRA_PT1H_ACC': 'precipitation',
    'SND_P1D_INSTANT': 'snow_ground',

    # --- PV system ---
    'pv_inv_out': 'pv_inv_out',
    'pv_inv_in': 'power',
    'pv_str_1': 'pv_str_1',
    'pv_str_2': 'pv_str_2',

    # --- Quality / flags ---
    'dataQC': 'dataQC',
    'vis_SnoP': 'snow',
    'Viss_day': 'viss_day',
    'Viss_instant': 'viss_instant'
}

col_name_dict_kuopio = {
    # --- Metadata --
    'fmisid': 'id',
    'stationname': 'stationname',
    'utctime': 'utctime',

    # --- Irradiance ---
    'GLOB_PT1M_AVG': 'ghi',
    'DIFF_PT1M_AVG': 'dhi', 
    # 'DIR_PT1M_AVG': 'dni', # the measurements for DNI seems to be erroneous
    'GLOBA_PT1M_AVG(:31)': 'poa',

    # --- Temperatures ---
    'TA_PT1M_AVG': 'T',
    'TA_PT1M_AVG(:31)': 't_roof',
    'TTECH_PT1M_AVG(:32)': 'module_temp_1',
    'TTECH_PT1M_AVG(:33)': 'module_temp_2',

    # --- Weather ---
    'P0_PT1M_AVG': 'pressure',
    'RH_PT1M_AVG': 'relative_humid',
    'CLA_PT1M_ACC': 'cloud_coverage',
    'WS_PT10M_AVG': 'wind',
    'WD_PT10M_AVG': 'wind_dir',
    'PRA_PT1H_ACC': 'precipitation',
    'SND_P1D_INSTANT': 'snow_ground',

    # --- PV system ---
    'pv_inv_out': 'pv_inv_out',
    'pv_inv_in': 'power',
    'pv_str_1': 'pv_str_1',
    'pv_str_2': 'pv_str_2',

    # --- Quality / flags ---
    'dataQC': 'dataQC',
    'vis_SnoP': 'snow',
    'Viss_day': 'viss_day',
    'Viss_instant': 'viss_instant'
}

col_name_dict_sodankyla_20 = {
    # --- Metadata --
    'fmisid': 'id',
    'stationname': 'stationname',
    'utctime': 'utctime',

    # --- Irradiance ---
    'GLOB_PT1M_AVG': 'ghi',
    'DIFF_PT1M_AVG': 'dhi',
    'DIR_PT1M_AVG': 'dni',
    #'GLOBA_PT1M_AVG(:31)': 'poa', # POA is not measured here

    # --- Temperatures ---
    'TA_PT1M_AVG': 'T',
    'TA_PT1M_AVG(:101)': 't_roof',
    'TTECH_PT1M_AVG(:102)': 'module_temp',

    # --- Weather ---
    'P0_PT1M_AVG': 'pressure',
    'RH_PT1M_AVG': 'relative_humid',
    'CLA_PT1M_ACC': 'cloud_coverage',
    'WS_PT10M_AVG': 'wind',
    'WD_PT10M_AVG': 'wind_dir',
    'PRA_PT1H_ACC': 'precipitation',
    'SND_P1D_INSTANT': 'snow_ground',

    # --- PV system ---
    'pv_inv_out': 'pv_inv_out',
    'DC_P[W]': 'power',

    # --- Quality / flags ---
    'dataQC': 'dataQC',
    'vis_SnoP': 'snow',
    'Viss_day': 'viss_day',
    'Viss_instant': 'viss_instant'
}

col_name_dict_sodankyla_90 = {
    # --- Metadata --
    'fmisid': 'id',
    'stationname': 'stationname',
    'utctime': 'utctime',

    # --- Irradiance ---
    'GLOB_PT1M_AVG': 'ghi',
    'DIFF_PT1M_AVG': 'dhi',
    'DIR_PT1M_AVG': 'dni',
    'GLOBA_PT1M_AVG(:101)': 'poa',

    # --- Temperatures ---
    'TA_PT1M_AVG': 'T',
    'TA_PT1M_AVG(:101)': 't_roof',
    'TTECH_PT1M_AVG(:103)': 'module_temp',

    # --- Weather ---
    'P0_PT1M_AVG': 'pressure',
    'RH_PT1M_AVG': 'relative_humid',
    'CLA_PT1M_ACC': 'cloud_coverage',
    'WS_PT10M_AVG': 'wind',
    'WD_PT10M_AVG': 'wind_dir',
    'PRA_PT1H_ACC': 'precipitation',
    'SND_P1D_INSTANT': 'snow_ground',

    # --- PV system ---
    'pv_inv_out': 'pv_inv_out',
    'DC_P[W]': 'power',

    # --- Quality / flags ---
    'dataQC': 'dataQC',
    'vis_SnoP': 'snow',
    'Viss_day': 'viss_day',
    'Viss_instant': 'viss_instant'
}

col_name_dict_turku = {
    # --- Metadata ---
    'datetime': 'datetime',

    # --- Irradiance ---
    'irradiance1': 'ghi_1',
    'irradiance2': 'ghi_2',

    # --- Temperatures ---
    'TEMP': 'T',

    # --- Weather ---
    'windDirection': 'wind_dir',
    'windSpeed': 'wind',
    'rain': 'precipitation',
    'RH': 'relative_humid',
    
    # --- PV system ---
    'Iac_1' : 'Iac_1',
    'Iac_2' : 'Iac_2',
    'Iac_3' : 'Iac_3',
    'Idc_MPP1' : 'Idc_MPP1',
    'Idc_MPP2' : 'Idc_MPP2',
    'Energy' : 'Energy',
    'Energy_MPP1' : 'Energy_MPP1',
    'Specific_yield' : 'Specific_yield',
    'Vac_1' : 'Vac_1',
    'Vac_2' : 'Vac_2',
    'Vac_3' : 'Vac_3',
    'Vdc_MPP1' : 'Vdc_MPP1',
    'Vdc_MPP2' : 'Vdc_MPP2',
    'dc_power' : 'power'
}