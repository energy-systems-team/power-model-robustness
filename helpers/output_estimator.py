"""
This file contains functions for estimating PV system output using Huld et. al 2010 PV output model
Required input is a dataframe with columns which contain absorbed radiation and panel temperature.
These columns are named "poa_ref_cor"(plane of array irradiance with reflection corrections) and
"module_temp" for PV module temperature.
 
Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen and Lauri Karttunen
"""

import pandas as pd
import numpy as np
import datetime
import joblib
from scipy.optimize import curve_fit
from sklearn.base import clone


FEATURES = ["poa_rc", "T", "module_temp", "wind"]
 
 
def _year_offset(start: pd.Timestamp, years: int) -> pd.Timestamp:
    """
    Shift a timestamp forward (or backward) by a given number of years.

    Parameters
    ----------
    start : pandas.Timestamp
        The original timestamp.
    years : int
        Number of years to shift the timestamp. Can be negative to
        shift backward in time.

    Returns
    -------
    pandas.Timestamp
        Timestamp offset by the specified number of years.
    """

    return start + pd.DateOffset(years=years)
  

def estimate_pvwatts(rated_power, data, radiation_col_name="poa_rc", temp_coef = -0.004):
    """
    Estimate PV system power output using PVWatts model.

    The model scales the system's rated power based on incident irradiance
    and applies a linear temperature correction relative to a reference
    cell temperature of 25°C.

    Parameters
    ----------
    rated_power : float
        Nominal system power at Standard Test Conditions (STC) in watts (W).
    data : pandas.DataFrame
        Input DataFrame containing at least:
        - radiation_col_name : irradiance values (W/m²)
        - 'cell_temp' : cell temperature (°C)
    radiation_col_name : str, optional
        Column name for plane-of-array irradiance (W/m²).
        Default is "poa_rc" (reflection-corrected POA irradiance).
    temp_coef : float, optional
        Temperature coefficient of power (1/°C). Typically negative
        (e.g., -0.004 /°C). Default is -0.004.

    Returns
    -------
    pandas.Series
        Estimated PV power output (W).

    Notes
    -----
    - The model follows:
          P = P_rated * (G / 1000) * (1 + γ * (T_cell - 25))
      where:
          G = irradiance (W/m²)
          γ = temperature coefficient
          T_cell = cell temperature (°C)
    - The function does not modify the input DataFrame.
    """

    absorbed_radiation = data[radiation_col_name]
    c = 1 + temp_coef * (data['cell_temp'] - 25.0)
    output = c * rated_power * absorbed_radiation / 1000.0
 
    return output


def estimate_general_and_finetuned_huld(rated_power_list, data_list, system_name_list, power_col_name="power", training_years=0, read_general_model_path="", read_finetuned_model_path="", write_model_path=""):
    """
    Estimate PV system power output using the Huld model with three variants:
    default (literature), general (transfer learning using data from other systems), 
    and optionally fine-tuned (site-specific trained).

    The function processes multiple PV systems using a leave-one-system-out approach:
      1. Default model:
        - Uses fixed coefficients from Huld et al. (2011).
      2. General model:
        - Trained on all systems except the current one.
      3. Fine-tuned model (optional):
        - Further trained on a subset of the target system's own data.

    Predictions are added to each system's DataFrame as new columns.

    Parameters
    ----------
    rated_power_list : list of float
        List of rated system powers (W) corresponding to each dataset.
    data_list : list of pandas.DataFrame
        List of DataFrames, each containing PV system data. Each must include:
        - irradiance ('poa_rc' or 'poa_comp_rc')
        - module temperature ('module_temp')
        - measured power (defined by `power_col_name`)
    system_name_list : list of str
        List of system identifiers (used for logging and model file naming).
    power_col_name : str, optional
        Column name for measured power. Default is "power".
    training_years : int, optional
        Number of years of data from each test system used for fine-tuning.
        If 0 (default), no fine-tuning is performed.
    read_general_model_path : str, optional
        Path to directory containing pre-trained general Huld model coefficients.
        If provided, models are loaded instead of trained.
    read_finetuned_model_path : str, optional
        Path to directory containing pre-trained fine-tuned coefficients.
    write_model_path : str, optional
        Directory where trained coefficients will be saved (if provided).

    Returns
    -------
    list of pandas.DataFrame
        Updated list of DataFrames with added columns:
        - 'Huld_def' : prediction using default coefficients
        - 'Huld_gen' : prediction using general model
        - 'Huld_fit' : prediction using fine-tuned model (if enabled)

    Notes
    -----
    - The Huld model relates normalized power output to irradiance and module
      temperature using a polynomial-logarithmic formulation.
    - Power is internally normalized by rated power during model fitting.
    - General models are trained using data from all other systems to improve
      generalization.
    - Fine-tuning is performed only on an initial time period of each system,
      with predictions applied to subsequent data.
    - If model paths are provided, coefficients are loaded/saved using joblib.
    - The function modifies the DataFrames in `data_list` in place.

    Workflow Summary
    ----------------
    For each system:
      - Compute default Huld prediction
      - Train or load general model → compute prediction
      - Optionally fine-tune → compute prediction for later period

    References
    ----------
    T. Huld, G. Friesen, A. Skoczek, R. P. Kenny, T. Sample, M. Field,
    E. D. Dunlop, "A power-rating model for crystalline silicon PV modules",
    Solar Energy Materials and Solar Cells (2011),
    doi: 10.1016/j.solmat.2011.07.026
    """

    # huld 2011 constants
    k1 = -0.01724
    k2 = -0.04047
    k3 = -0.0047
    k4 = 0.000149
    k5 = 0.000147
    k6 = 0.000005

    default_coeffs = [k1, k2, k3, k4, k5, k6]

    for i in range(len(rated_power_list)):
        data = data_list[i]

        scaled_power_col_name = f"scaled_{power_col_name}"
        # Compute scaled_power and POA if not present
        if scaled_power_col_name not in data.columns:
            data[scaled_power_col_name] = data[power_col_name] / rated_power_list[i]
        if "poa_rc" not in data.columns and "poa_comp_rc" in data.columns:
            data["poa_rc"] = data["poa_comp_rc"]

    for i in range(len(rated_power_list)):
        print(f"\nProcessing system: {system_name_list[i]}")
        rated_power = rated_power_list[i]
        test_data = data_list[i]
        train_data = pd.concat([data_list[j] for j in range(len(data_list)) if j != i])

        # Calculate estimates with default coefficients for the test set
        pred_p_default = _huld_scaled([test_data["poa_rc"], test_data["module_temp"]], 
                               default_coeffs[0], default_coeffs[1], default_coeffs[2],
                               default_coeffs[3], default_coeffs[4], default_coeffs[5])
        
        data_list[i]["Huld_def"] = pred_p_default * rated_power

        # Train the general model using the data from other systems
        irr, mod_temp, power = (train_data["poa_rc"]).values.astype('float64'), \
                    (train_data['module_temp']).values.astype('float64'), \
                    (train_data[scaled_power_col_name]).values.astype('float64')
        
        if read_general_model_path:
            read_filename = f"{read_general_model_path}/huld_{system_name_list[i]}_general_coefficients.pkl"
            print(f"Reading model from: {read_filename}")
            with open(read_filename, "rb") as f:
                general_coeffs = joblib.load(f)
        else:
            print("Using default coefs from literature as starting values before curve fit")

            # Fit to train data and find the general coefficients 
            # NOTE: the power is scaled with rated power
            general_coeffs, pcov = curve_fit(f=_huld_scaled, 
                                 xdata=[irr, mod_temp], 
                                 ydata=power,
                                 p0=default_coeffs,
                                 method="lm")
        
        print(f"Huld general coefficients for {system_name_list[i]}: {np.array(general_coeffs)}")

        # Save model if path provided
        if write_model_path:
            write_filename = f"{write_model_path}/huld_{system_name_list[i]}_general_coefficients.pkl"
            print(f"Saving model to: {write_filename}")
            with open(write_filename, "wb") as f:
                joblib.dump(general_coeffs, f)
        
        pred_p_general = _huld_scaled([test_data["poa_rc"], test_data['module_temp']], 
                                 general_coeffs[0], general_coeffs[1], general_coeffs[2], 
                                 general_coeffs[3], general_coeffs[4], general_coeffs[5])

        # Add prediction to test data
        data_list[i]["Huld_gen"] = pred_p_general * rated_power_list[i]

        # Optional fine-tuning
        if training_years > 0:
            print(f"Fine-tuning model for {training_years} year(s)...")
            fine_tune_start = test_data.index.min()
            fine_tune_end = fine_tune_start + datetime.timedelta(days=365 * training_years)

            fine_train = test_data[(test_data.index >= fine_tune_start) & (test_data.index < fine_tune_end)]
            fine_test = test_data[(test_data.index >= fine_tune_end)]

            irr, mod_temp, power = (fine_train["poa_rc"]).values.astype('float64'), \
                    (fine_train['module_temp']).values.astype('float64'), \
                    (fine_train[scaled_power_col_name]).values.astype('float64')
            
            # Try to read fine-tuned coefficients
            if read_finetuned_model_path:
                read_filename = f"{read_finetuned_model_path}/huld_{system_name_list[i]}_fine-tuned_coefficients.pkl"
                print(f"Reading model from: {read_filename}")
                with open(read_filename, "rb") as f:
                    finetuned_coeffs = joblib.load(f)
            # Reuse previously gotten coeffs and use them as default
            else:
                finetuned_coeffs, pcov = curve_fit(f=_huld_scaled, 
                                    xdata=[irr, mod_temp], 
                                    ydata=power,
                                    p0=general_coeffs,
                                    method="lm")
            
            print(f"Huld fine-tuned coefficients for {system_name_list[i]}: {np.array(finetuned_coeffs)}")
            
            # Save fine-tuned coefficients if path provided
            if write_model_path:
                write_filename = f"{write_model_path}/huld_{system_name_list[i]}_fine-tuned_coefficients.pkl"
                print(f"Saving fine-tuned model to: {write_filename}")
                with open(write_filename, "wb") as f:
                    joblib.dump(finetuned_coeffs, f)

            pred_p_finetuned = _huld_scaled([fine_test["poa_rc"], fine_test['module_temp']], 
                                        finetuned_coeffs[0], finetuned_coeffs[1], finetuned_coeffs[2], 
                                        finetuned_coeffs[3], finetuned_coeffs[4], finetuned_coeffs[5])
            data_list[i]["Huld_fit"] = pd.Series(index=test_data.index, dtype=float)
            data_list[i].loc[fine_test.index, "Huld_fit"] = pred_p_finetuned * rated_power_list[i]

    return data_list


def _huld_scaled(X, k1_dot, k2_dot, k3_dot, k4_dot, k5_dot, k6_dot):
    """
    Compute normalized PV power output using the Huld model formulation.

    The Huld model expresses normalized power output as a function of
    irradiance and module temperature using a polynomial-logarithmic form.
    This implementation operates on *scaled* (normalized) power, meaning
    the output represents power relative to the system's rated capacity.

    Parameters
    ----------
    X : tuple or list of array-like
        Input variables:
        - X[0] : plane-of-array irradiance (W/m²)
        - X[1] : module temperature (°C)
    k1_dot, k2_dot, k3_dot, k4_dot, k5_dot, k6_dot : float
        Huld model coefficients.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Normalized PV power output (unitless), i.e., fraction of rated power.

    Notes
    -----
    - The model follows:
          P_norm = G * [1 + k1 ln(G) + k2 ln(G)² + k3 T
                        + k4 T ln(G) + k5 T ln(G)² + k6 T²]

      where:
          G = irradiance / 1000 (normalized irradiance)
          T = module temperature - 25°C (temperature deviation)

    - Irradiance values ≤ 0 are replaced with a small positive value
      (1e-6) to avoid numerical issues with the logarithm.
    - The output is scaled power (P / P_rated). Multiply by rated power
      to obtain absolute power (W).
    - This function is designed for use with curve fitting.
    """
    ref_temp = 25
    ref_irrad = 1000
 
    i, t = X
    G = i / ref_irrad
    G[G <= 0] = 0.000001 # Make zeros small floats for log
    T = t - ref_temp
    pred_p = G*(1 + k1_dot*np.log(G) + k2_dot*np.log(G)**2 + k3_dot*T + k4_dot*T*np.log(G) + k5_dot*T*np.log(G)**2 + k6_dot*T**2)
   
    return pred_p


def estimate_general_and_finetuned_pvusa(rated_power_list, data_list, system_name_list, power_col_name="power", training_years=0, read_general_model_path="", read_finetuned_model_path="", write_model_path=""):
    """
    Estimate PV system power output using the PVUSA model with two variants: 
    general (transfer learning using data from other systems), 
    and optionally fine-tuned (site-specific trained).

    The function processes multiple PV systems using a leave-one-system-out approach:
      1. General model:
        - Trained on all systems except the current one.
      2. Fine-tuned model (optional):
        - Further trained on a subset of the target system's own data.

    Predictions are added to each system's DataFrame as new columns.

    Parameters
    ----------
    rated_power_list : list of float
        List of nominal system powers (W) corresponding to each dataset.
    data_list : list of pandas.DataFrame
        List of DataFrames, each containing PV system data. Each must include:
        - irradiance ('poa_rc' or 'poa_comp_rc')
        - ambient temperature ('T')
        - wind speed ('wind')
        - measured power (defined by `power_col_name`)
    system_name_list : list of str
        Identifiers for each PV system (used for logging and file naming).
    power_col_name : str, optional
        Column name for measured power. Default is "power".
    training_years : int, optional
        Number of initial years of data used for system-specific fine-tuning.
        If 0 (default), no fine-tuning is performed.
    read_general_model_path : str, optional
        Path to directory containing pre-trained general PVUSA model coefficients.
        If provided, models are loaded instead of trained.
    read_finetuned_model_path : str, optional
        Path to directory containing pre-trained fine-tuned coefficients.
    write_model_path : str, optional
        Directory where trained coefficients will be saved (if provided).

    Returns
    -------
    list of pandas.DataFrame
        Updated list of DataFrames with added columns:
        - 'PVUSA_gen' : general model predictions
        - 'PVUSA_fit' : fine-tuned model predictions (if enabled)

    Notes
    -----
    - The PVUSA model expresses power as a function of irradiance, ambient
      temperature, and wind speed.
    - Power is internally normalized by rated power during model fitting.
    - General models are trained using data from all other systems to improve
      generalization.
    - Fine-tuning is performed only on an initial time period of each system,
      with predictions applied to subsequent data.
    - If model paths are provided, coefficients are loaded/saved using joblib.
    - The function modifies the DataFrames in `data_list` in place.

    Workflow Summary
    ----------------
    For each system:
      - Train or load general model → compute prediction
      - Optionally fine-tune → compute prediction for later period

    References
    ----------
    R. Steele, PVUSA progress report (1990), 
    doi: 10.2172/10163193,  

    Myers, D R. "Evaluation of the Performance of the PVUSA Rating Methodology 
    Applied to Dual Junction PV Technology: Preprint (Revised)." (2009)

    """

    for i in range(len(rated_power_list)):
        data = data_list[i]

        scaled_power_col_name = f"scaled_{power_col_name}"
        # Compute scaled_power and POA if not present
        if scaled_power_col_name not in data.columns:
            data[scaled_power_col_name] = data[power_col_name] / rated_power_list[i]
        if "poa_rc" not in data.columns and "poa_comp_rc" in data.columns:
            data["poa_rc"] = data["poa_comp_rc"]

    for i in range(len(rated_power_list)):
        print(f"\nProcessing system: {system_name_list[i]}")
        test_data = data_list[i]
        train_data = pd.concat([data_list[j] for j in range(len(data_list)) if j != i])

        irr, temp, wind, power = (train_data["poa_rc"]).values.astype('float64'), \
                    (train_data['T']).values.astype('float64'), \
                    (train_data['wind']).values.astype('float64'), \
                    (train_data[scaled_power_col_name]).values.astype('float64')
        
        if read_general_model_path:
            read_filename = f"{read_general_model_path}/pvusa_{system_name_list[i]}_general_coefficients.pkl"
            print(f"Reading model from: {read_filename}")
            with open(read_filename, "rb") as f:
                general_coeffs = joblib.load(f)
        else:
            print("Using default coefs from literature")
            # Coefficients from conference paper by Daryl Myers
            # "Evaluation of the Performance of the PVUSA Rating Methodology 
            # Applied to DUAL Junction PV Technology" (2009)
            a = 1.41870
            b = 0.000051
            c = 0.002291
            d = 0.000361

            default_coeffs = [a, b, c, d]

            # Fit to train data and find the general coefficients 
            # NOTE: the power is scaled with rated power
            general_coeffs, pcov = curve_fit(f=_pvusa, 
                                    xdata=[irr, temp, wind], 
                                    ydata=power,
                                    p0=default_coeffs,
                                    method="lm")
        
        print(f"PVUSA general coefficients for {system_name_list[i]}: {general_coeffs}")

        # Save model if path provided
        if write_model_path:
            write_filename = f"{write_model_path}/pvusa_{system_name_list[i]}_general_coefficients.pkl"
            print(f"Saving model to: {write_filename}")
            with open(write_filename, "wb") as f:
                joblib.dump(general_coeffs, f)
        
        pred_p_general = _pvusa([test_data["poa_rc"], 
                                 test_data['T'], 
                                 test_data['wind']], 
                                 general_coeffs[0], general_coeffs[1], general_coeffs[2], general_coeffs[3])

        # Add prediction to test data
        data_list[i]["PVUSA_gen"] = pred_p_general * rated_power_list[i]

        # Optional fine-tuning
        if training_years > 0:
            print(f"Fine-tuning model for {training_years} year(s)...")
            fine_tune_start = test_data.index.min()
            fine_tune_end = fine_tune_start + datetime.timedelta(days=365 * training_years)

            fine_train = test_data[(test_data.index >= fine_tune_start) & (test_data.index < fine_tune_end)]
            fine_test = test_data[(test_data.index >= fine_tune_end)]

            irr, temp, wind, power = (fine_train["poa_rc"]).values.astype('float64'), \
                    (fine_train['T']).values.astype('float64'), \
                    (fine_train['wind']).values.astype('float64'), \
                    (fine_train[scaled_power_col_name]).values.astype('float64')
        
            # Try to read fine-tuned coefficients
            if read_finetuned_model_path:
                read_filename = f"{read_finetuned_model_path}/pvusa_{system_name_list[i]}_fine-tuned_coefficients.pkl"
                print(f"Reading model from: {read_filename}")
                with open(read_filename, "rb") as f:
                    finetuned_coeffs = joblib.load(f)
            # Reuse previously gotten coeffs and use them as default
            else:
                finetuned_coeffs, pcov = curve_fit(f=_pvusa, 
                                    xdata=[irr, temp, wind], 
                                    ydata=power,
                                    p0=general_coeffs,
                                    method="lm")
            
            print(f"PVUSA fine-tuned coefficients for {system_name_list[i]}: {finetuned_coeffs}")

            # Save fine-tuned coefficients if path provided
            if write_model_path:
                write_filename = f"{write_model_path}/pvusa_{system_name_list[i]}_fine-tuned_coefficients.pkl"
                print(f"Saving fine-tuned model to: {write_filename}")
                with open(write_filename, "wb") as f:
                    joblib.dump(finetuned_coeffs, f)
            
            pred_p_finetuned = _pvusa([fine_test["poa_rc"], 
                                        fine_test['T'], 
                                        fine_test['wind']], 
                                        finetuned_coeffs[0], finetuned_coeffs[1], finetuned_coeffs[2], finetuned_coeffs[3])
            data_list[i]["PVUSA_fit"] = pd.Series(index=test_data.index, dtype=float)
            data_list[i].loc[fine_test.index, "PVUSA_fit"] = pred_p_finetuned * rated_power_list[i]

    return data_list


def _pvusa(X, a, b, c, d):
    """
    Compute normalized PV power output using the PVUSA model.

    The PVUSA model represents PV power output as a function of
    irradiance, ambient temperature, and wind speed using a linear
    formulation with interaction terms.

    Parameters
    ----------
    X : tuple or list of array-like
        Input variables:
        - X[0] : irradiance (W/m²)
        - X[1] : ambient temperature (°C)
        - X[2] : wind speed (m/s)
    a, b, c, d : float
        PVUSA model coefficients.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Normalized PV power output (unitless), i.e., fraction of rated power.

    Notes
    -----
    - The model follows:
          P_norm = G * (a + b * G + c * W + d * T)
      where:
          G = irradiance (W/m²)
          T = ambient temperature (°C)
          W = wind speed (m/s)
    - The output must be multiplied by rated power to obtain absolute
      power (W).
    - The model captures temperature and cooling effects via ambient
      temperature and wind speed.
    - This function is designed for use with curve fitting.
    """

    i, t, w = X
    return i * (a + b * i + c * w + d * t)


def estimate_from_saved_model_pipeline(model_type, rated_power_list, data_list, system_name_list,
                                   power_col_name="power", training_years=1,
                                   read_model_path="", run_iter="", load_fine_tuned=True):
    """
    Generate PV power predictions using pre-trained machine learning pipelines
    (MLP or Histogram Gradient Boosting), with optional fine-tuned models.

    This function performs inference only by loading previously trained pipelines,
    applying them to input data, and optionally using or creating system-specific
    fine-tuned models.

    For each system:
      - A general model (trained on other systems) is loaded and applied.
      - A fine-tuned model is either loaded from disk or trained using the
        initial portion of the system's data (if enabled).
      - Predictions are scaled by the system's rated power.

    Parameters
    ----------
    model_type : str
        Model type identifier. Supported values:
        - "MLP" : Multilayer Perceptron pipeline
        - "GBM" : Histogram Gradient Boosting pipeline
    rated_power_list : list of float
        List of rated power of the systems (W).
    data_list : list of pandas.DataFrame
        List of DataFrames, each containing system data. Each must include:
        - input features defined in global `FEATURES`
        - measured power (`power_col_name`)
    system_name_list : list of str
        Identifiers for each system (used for file naming and logging).
    power_col_name : str, optional
        Column name for measured power. Default is "power".
    training_years : int, optional
        Number of initial years of each system's data used for fine-tuning.
        Default is 1.
    read_model_path : str, optional
        Directory path from which pretrained pipelines are loaded.
    run_iter : str, optional
        Identifier or version tag used in model filenames.
    load_fine_tuned : bool, optional
        If True, fine-tuned models are loaded from disk.
        If False, fine-tuned models are trained on-the-fly using the system's
        initial data window. Default is True.

    Returns
    -------
    list of pandas.DataFrame
        Updated list of DataFrames with added prediction columns:
        - "{model_type}_gen" : general model predictions
        - "{model_type}_fit" : fine-tuned model predictions

    Notes
    -----
    - Pipelines are assumed to include preprocessing steps (e.g., scaling),
      so raw feature values can be passed directly.
    - Predictions are initially normalized and then scaled by rated power.
    - Fine-tuning is performed using a time-based split:
          * training: first `training_years`
          * testing: remaining period
    - If `load_fine_tuned=True` and no file is found, fine-tuned predictions
      are skipped.
    - If `load_fine_tuned=False`, a new model is cloned from the general
      pipeline and refit on the system-specific training window.
    - The function modifies the input DataFrames in place.
    - Requires a global `FEATURES` list defining model input variables.

    Workflow Summary
    ----------------
    For each system:
      1. Ensure normalized target and required irradiance column exist.
      2. Load general pipeline → predict full dataset.
      3. Define fine-tuning window.
      4. Load or train fine-tuned pipeline → predict remaining data.

    Raises
    ------
    ValueError
        If required input features (defined in `FEATURES`) are missing
        from a dataset.
    """

    if model_type == "MLP":
        print("estimating MLP model")
        model_type_path = "mlp_pipeline"
    elif model_type == "GBM":
        print("estimating GBM model")
        model_type_path = "hgbr_pipeline"

    scaled_power_col_name = f"scaled_{power_col_name}"

    # Ensure scaled target and poa_rc exist
    for i in range(len(rated_power_list)):
        data = data_list[i]
        if scaled_power_col_name not in data.columns:
            data[scaled_power_col_name] = data[power_col_name] / rated_power_list[i]
        if "poa_rc" not in data.columns and "poa_comp_rc" in data.columns:
            data["poa_rc"] = data["poa_comp_rc"]

    for i in range(len(rated_power_list)):
        print(f"\nProcessing system: {system_name_list[i]}")
        test_data = data_list[i]

        # Validate features
        missing = [f for f in FEATURES if f not in test_data.columns]
        if missing:
            raise ValueError(f"Missing features in test data: {missing}")


        # Load GENERAL model
        general_pipe_file = f"{read_model_path}/{model_type_path}_{system_name_list[i]}_{run_iter}.pkl"
        try:
            pipe_general = joblib.load(general_pipe_file)
            print("Loaded pretrained general model.")

            # Predict on raw features (pipeline handles scaling)
            X_test = test_data[FEATURES]
            general_predictions_norm = pipe_general.predict(X_test)
            data_list[i][f"{model_type}_gen"] = general_predictions_norm * rated_power_list[i]

        except FileNotFoundError:
            print("General model file not found.")

        # FINE-TUNED model
        fine_tune_start = test_data.index.min()
        fine_tune_end = _year_offset(fine_tune_start, training_years)

        fine_test = test_data[(test_data.index >= fine_tune_end)]
        if fine_test.empty:
            print("No data after fine-tune window.")
            continue

        X_fine_test = fine_test[FEATURES]

        # Load from file
        if load_fine_tuned:
            fine_pipe_file = (
                f"{read_model_path}/{model_type_path}_{system_name_list[i]}_{run_iter}"
                f"_fine-tuned_{str(training_years)}_years.pkl"
            )
            try:
                pipe_fine = joblib.load(fine_pipe_file)
                print("Loaded pretrained fine-tuned model.")

                fine_predictions_norm = pipe_fine.predict(X_fine_test)

                # Store predictions
                if f"{model_type}_fit" not in data_list[i].columns:
                    data_list[i][f"{model_type}_fit"] = pd.Series(index=test_data.index, dtype=float)
                data_list[i].loc[fine_test.index, f"{model_type}_fit"] = fine_predictions_norm * rated_power_list[i]

            except FileNotFoundError:
                print("Fine-tuned model file not found.")

        # Fine-tune new pipeline
        else:
            pipe_fine = clone(pipe_general)

            fine_train = test_data[(test_data.index >= fine_tune_start) & (test_data.index < fine_tune_end)]

            if fine_train.empty:
                print("Warning: fine-tune window is empty; skipping fine-tuning fit.")
            else:
                X_fine_train = fine_train[FEATURES]
                y_fine_train = fine_train[scaled_power_col_name]
                print(f"Fitting fine-tuned pipeline on {len(X_fine_train)} samples...")
                # Note: This refits a separate model on the fine window only.
                pipe_fine.fit(X_fine_train, y_fine_train)

            # Predict on the post-fine-tune period
            if fine_test.empty:
                print(f"No samples after fine-tune window; '{model_type}_fit' will remain NaN.")
                data_list[i][f"{model_type}_fit"] = np.nan
            else:
                pred_fine_norm = pipe_fine.predict(X_fine_test)
                if f"{model_type}_fit" not in data_list[i].columns:
                    data_list[i][f"{model_type}_fit"] = np.nan
                data_list[i].loc[X_fine_test.index, f"{model_type}_fit"] = pred_fine_norm * rated_power_list[i]


    return data_list