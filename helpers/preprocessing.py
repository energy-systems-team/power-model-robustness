"""
This file contains functions for pre-processing data.

Original author: Väinö Anttalainen.
Other editors: Lauri Karttunen
"""

import pandas as pd
import numpy as np

def merge_measurements(df, col_names, diff=10):
        """
        Merge two measurement columns by averaging consistent values.

        The function compares two measurement series and computes their
        average if their absolute difference is below a specified threshold.
        If the difference exceeds the threshold, the result is set to NaN.

        Parameters
        ----------
        df : pandas.DataFrame
                Input DataFrame containing the measurement columns.
        col_names : list of str
                List containing exactly two column names to be merged.
        diff : float, optional
                Maximum allowed absolute difference between the two measurements
                for averaging. Default is 10.

        Returns
        -------
        pandas.Series
                Averaged measurement values where the difference is within the
                threshold; otherwise NaN.

        Notes
        -----
        - Useful for combining redundant sensors while filtering out inconsistent values.
        - Does not modify the input DataFrame.
        - Requires exactly two column names in `col_names`.
        """

        measurement_1 = df[col_names[0]]
        measurement_2 = df[col_names[1]]

        measurements_df = pd.DataFrame({'measurement_1': measurement_1, 'measurement_2': measurement_2})

        # Calculate the average where the absolute difference is less than diff
        measurements_df['avg'] = np.where(
                np.abs(measurements_df['measurement_1'] - measurements_df['measurement_2']) < diff, 
                measurements_df[['measurement_1', 'measurement_2']].mean(axis=1), np.nan)

        return measurements_df['avg']


def impute_columns(df, col_names, resolutions):
        """
        Impute missing values in columns with lower temporal resolution.

        The function fills missing values by propagating the next available
        observation backward over a number of rows based on the measurement
        resolution.

        Parameters
        ----------
        df : pandas.DataFrame
                Input DataFrame.
        col_names : list of str
                Column names to impute.
        resolutions : list of int
                Measurement resolutions (in minutes) corresponding to each column.

        Returns
        -------
        pandas.DataFrame
                DataFrame with imputed columns.

        Notes
        -----
        - Each resolution determines how many preceding rows are filled:
                rows_to_fill = resolution - 1
        - Only gaps consistent with the resolution are filled.
        - If `col_names` and `resolutions` lengths do not match, no imputation is performed.
        - The function modifies the DataFrame in place.
        """

        if len(col_names) != len(resolutions):
                print("Length of col_names and resolutions must match. Columns not imputed.")
        else:
                num_of_rows_to_impute_list =  [res - 1 for res in resolutions]
                for i in range(len(col_names)):
                        num_of_rows_to_impute = num_of_rows_to_impute_list[i]
                        column = col_names[i]
                        values = df[column].values.copy()
                        non_na_indices = np.where(~pd.isna(values))[0]
                        ind_prev = 0
                        for ind in non_na_indices:
                                start_ind = max(0, ind - num_of_rows_to_impute) 
                                if ind - ind_prev >= num_of_rows_to_impute:
                                        values[start_ind:ind] = values[ind]
                                ind_prev = ind
                        
                        df[column] = values
        return df


def calculate_power_withour_plr_losses(df, plr_value, initial_date):
        """
        Estimate power output without performance loss ratio (PLR) degradation.

        The function removes the effect of annual degradation from measured
        power values based on a constant PLR (% per year).

        Parameters
        ----------
        df : pandas.DataFrame
                Input DataFrame containing a 'power' column and a datetime index.
        plr_value : float
                Performance loss ratio (% per year).
        initial_date : str or pandas.Timestamp
                Reference date corresponding to zero degradation.

        Returns
        -------
        pandas.DataFrame
                DataFrame with added 'power_without_plr_losses' column.

        Notes
        -----
        - Model:
                P_corrected = P_measured / (1 - PLR * time_elapsed)
        - Time is computed in fractional years using minute resolution.
        - Absolute value of PLR is used.
        - The function modifies the DataFrame in place.
        """

        initial_date = pd.to_datetime(initial_date)

        mins_in_year = 60*24*365
        mins_in_day = 24*60
        power_with_plr_losses = df["power"]

        date_difference = df.index - initial_date
        mins_from_beginning = date_difference.days * mins_in_day + np.floor(date_difference.seconds / 60)

        power_without_plr_losses = power_with_plr_losses / (1 - np.abs(plr_value)/100 * (mins_from_beginning / mins_in_year))

        df["power_without_plr_losses"] = power_without_plr_losses

        return df


def calculate_power_with_plr_losses(df, plr_value, col_name, save_col_name, initial_date):
        """
        Apply performance loss ratio (PLR) degradation to power values.

        The function adjusts power values by applying a linear degradation
        factor over time.

        Parameters
        ----------
        df : pandas.DataFrame
                Input DataFrame with a datetime index.
        plr_value : float
                Performance loss ratio (% per year).
        col_name : str
                Column name containing power values to be degraded.
        initial_date : str or pandas.Timestamp
                Reference date corresponding to zero degradation.

        Returns
        -------
        pandas.DataFrame
                DataFrame with updated power column including PLR losses.

        Notes
        -----
        - Model:
                P_degraded = (1 - PLR * time_elapsed) * P_original
        - Time is computed in fractional years based on minute resolution.
        - Absolute value of PLR is used.
        - The function modifies the DataFrame in place.
        """

        initial_date = pd.to_datetime(initial_date)

        mins_in_year = 60*24*365
        mins_in_day = 24*60
        power_without_plr = df[col_name]

        date_difference = df.index - initial_date
        mins_from_beginning = date_difference.days * mins_in_day + np.floor(date_difference.seconds / 60)

        power_with_plr_losses = (1 - np.abs(plr_value)/100 * (mins_from_beginning / mins_in_year)) * power_without_plr

        df[save_col_name] = power_with_plr_losses

        return df


#def aggregate_data(data, resolution, count_limit, save_data_path=""):
#        """
#        Aggregate time-series data to a coarser temporal resolution.
#
#        The function resamples the data to a specified time interval,
#        computing mean values while ensuring a minimum number of observations
#        per aggregation window.
#
#        Parameters
#        ----------
#        data : pandas.DataFrame
#                Input dataset indexed by datetime.
#        resolution : int
#                Target resolution in minutes.
#        count_limit : int
#                Minimum number of valid observations required within each
#                resampling window to compute the mean.
#        save_data_path : str, optional
#                Path to save the aggregated dataset (CSV, semicolon-separated).
#                If empty, the data is not saved.
#
#        Returns
#        -------
#        pandas.DataFrame
#                Aggregated dataset with reduced temporal resolution.
#
#        Notes
#        -----
#        - Aggregation uses:
#                mean → aggregated value
#                count → number of observations in window
#        - Windows with insufficient data (`count < count_limit`) are discarded.
#        - The 'time' column (if present) is excluded from aggregation.
#        - The function does not modify the original DataFrame.
#        """
#        
#        resolution_str = f'{resolution}min'
#        data_aggregated = data.resample(resolution_str)[data.columns.drop("time")].agg(["mean", "count"])
#        counts = data_aggregated.xs("count", axis=1, level=1)
#        means = data_aggregated.xs("mean", axis=1, level=1)
#        data_aggregated = means.where(counts >= count_limit).dropna()
#
#        if save_data_path != "":
#                data_aggregated.to_csv(save_data_path, sep=";")
#
#        return data_aggregated
#
#