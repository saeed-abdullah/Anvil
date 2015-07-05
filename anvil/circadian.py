# -*- coding: utf-8 -*-
"""
    anvil.circadian
    ~~~~~~~~~~~~~~~

    Collection of circadian utilities

    :copyright: (c) 2015 by Saeed Abdullah.

"""


import datetime
import numpy as np
import pandas as pd
import scipy

class CircadianAnalysis(object):
    """
    Utility functions for circadian analysis.
    """

    @staticmethod
    def inter_daily_stability(df, value_col,
                              hour_col='hour'):
        """
        Calculates interdaily stability (IS) from hourly data.

        Higher IS means better rhythmicity. The implementation
        follows the formula (1) from Witting et al. (pg. 565).


        Witting, W., et al. "Alterations in the circadian rest-activity rhythm
        in aging and Alzheimer's disease."
        Biological psychiatry 27.6 (1990): 563-572.

        Parameters
        ----------
        df: DataFrame.
        value_col: Column to calculate daily stability.
        hour_col: Column indicating hourly values.
        """

        hour_count = 24
        mean = df[value_col].mean()
        N = len(df)

        denom_sum_val = sum((df[value_col] - mean)**2)
        denominator = hour_count * denom_sum_val

        l = []
        for k, v in df.groupby(hour_col):
            l.append((v[value_col].mean() - mean)**2)

        nom = sum(l) * N

        return nom/denominator

    @staticmethod
    def inter_daily_stability_across_users(df, value_col,
                                           hour_col='hour',
                                           user_col='user_id'):
        """
        Utility function for calculating IS across users.
        """

        l = []
        for k, v in df.groupby(user_col):
            stability = CircadianAnalysis.inter_daily_stability(v,
                                                                value_col=value_col,
                                                                hour_col=hour_col)
            l.append({'stability': stability, user_col: k})

        return pd.DataFrame(l)

    @staticmethod
    def intra_daily_variability(df, value_col):
        """
        Calculate intra-daily variability (IV).

        High IV indicates fragmented day. The implementation
        follows the formula (1) from Witting et al. (pg. 565).


        Parameters
        ----------
        df: DataFrame. It must be sorted by date (ascending).
        value_col: Column to compute IV.
        """

        s = (df[value_col] - df[value_col].shift(1))**2
        nom = len(df) * sum(s.fillna(0).values)

        mean = df[value_col].mean()
        N = len(df)

        denom = (N - 1) * sum((df[value_col] - mean)**2)

        return nom/denom

    @staticmethod
    def intra_daily_variability_across_users(df, value_col,
                                             user_col='user_id'):
        """
        Utility function for calculating IV across users
        """

        l = []
        for k, v in df.groupby(user_col):
            variability = CircadianAnalysis.intra_daily_variability(v, value_col)
            l.append({'variability': variability, user_col: k})

        return pd.DataFrame(l)

    @staticmethod
    def sort_by_hourly_values(df, value_col,
                              hour_col='hour'):
        """
        Sorts by average hourly values.

        Performs sorting by averaging values across
        hours. The value returns here can be used
        for M10 (most active 10 hours) and L5 (least
        active 5 hours) as defined in Witting et al.

        Parameters
        ----------

        df: DataFrame.
        value_col: Column to compute average values.
        hour_col: Column denoting hours.


        Returns
        -------

        A sorted list (ascending) of tuples (h, v)
        with first element is the hour and second
        element is average value.
        """

        d = {}
        for k, v in df.groupby(hour_col):
            d[k] = v[value_col].mean()

        # sort by value
        return sorted(d.items(), key=lambda z: z[1])

    @staticmethod
    def _convert_timestamp_to_decimal(timeseries,
                                      should_convert=False):
        """
        Convert timestamp to decimal value for SRM.

        Conversion happens using hour + minute/60 formula,
        so, 08:45 would be 8.75.

        Parameters
        ----------

        timeseries: A series with timestamp or datetimes.
        should_convert: If the timeseries should be converted
                        using `pd.to_datetime` function.
        """

        if should_convert:
            timeseries = pd.to_datetime(timeseries)
        return timeseries.map(lambda z: z.hour + z.minute/60)

    @staticmethod
    def _purge_srm_outliers(series,
                            lower_limit,
                            upper_limit):
        """
        Removes outliers before calculating SRM.

        Outliers are points beyond given lower and
        upper limit, i.e., p < lower_limit or
        p > upper_limit.

        Parameters
        ----------

        series: A series containing SRM scores.
        lower_limit: Lower limit.
        upper_limit: Upper limit.

        Returns
        -------

        A new DataFrame with valid points.
        """

        filtering_f = lambda z: z >= lower_limit and z <= upper_limit
        return series[series.map(filtering_f)]

    @staticmethod
    def _srm_preprocssing(series):
        """
        Pre-processing for SRM calculation.

        This function does the followings:
        1. Converts timestamp to decimal values
        2. Removes outliers with > 1.5 * SD.

        Parameters
        ----------
        series: Series containing values

        Returns
        -------
        A new Series.
        """

        series = CircadianAnalysis._convert_timestamp_to_decimal(series)
        mean = series.mean()
        std = series.std()

        if std < 0.5:
            return series

        lower_limit = mean - 1.5 * std
        upper_limit = mean + 1.5 * std
        series = CircadianAnalysis._purge_srm_outliers(series,
                                                       lower_limit=lower_limit,
                                                       upper_limit=upper_limit)

        return series

    @staticmethod
    def _calculate_srm_hit(series,
                           lower_limit,
                           upper_limit):
        """
        Calculates number of hits.

        It is the number of values falling within
        the given range.
        """

        filtering_f = lambda z: z >= lower_limit and z <= upper_limit

        return sum(series.map(filtering_f))

    @staticmethod
    def calculate_srm(df, target_col,
                      time_col='completion_time',
                      min_samples=3):
        """
        Calculates SRM score.

        Parameters
        ----------

        df: DataFrame.
        target_col: Column with target names. Individual
                    hits would be calculated for each
                    targets.
        time_col: Column containing timestamps.
        min_samples: Minimum samples for calculating hit
                     for a given column. Default is 3 (40%
                     of a week).
        """

        hit_range = 45/60  # "hit" if falls within 45 minute

        l = []

        for k, v in df.groupby(target_col):
            series = v.loc[:, time_col]
            series = CircadianAnalysis._srm_preprocssing(series)
            if len(series) >= min_samples:

                mean = series.mean()

                lower = mean - hit_range
                upper = mean + hit_range

                hit = CircadianAnalysis._calculate_srm_hit(series,
                                                           lower_limit=lower,
                                                           upper_limit=upper)
                l.append(hit)

        return sum(l)/len(l)

    @staticmethod
    def calculate_srm_across_users(df,
                                   user_col='user_id',
                                   **srm_args):
        """
        Calculates SRM score across users.
        """

        l = []
        for k, v in df.groupby(user_col):
            srm = CircadianAnalysis.calculate_srm(v, **srm_args)
            l.append({'user_id': k, 'srm': srm})

        return pd.DataFrame(l)

    @staticmethod
    def rolling_srm_across_users(df, start_date,
                                 how_many_days,
                                 time_col='completion_time',
                                 **srm_args):
        """
        Calculates rolling SRM across days.
        """
        srm_df = None
        for i in range(how_many_days):
            s = start_date + datetime.timedelta(days=i)
            e = s + datetime.timedelta(days=7)

            w = df[df[time_col].map(lambda z: z >= s and z < e)]
            df_w = CircadianAnalysis.calculate_srm_across_users(w,
                                                                time_col=time_col,
                                                                **srm_args)
            df_w['date'] = s.date()
            srm_df = pd.concat([srm_df, df_w], ignore_index=True)

        return srm_df


class CosineCurveFitting(object):
    """
    Performs Cosine curve fitting.

    Harmonic cosine analysis is often used to model Circadian
    phenomenon. In particular, Vetter et al. [1] uses the following
    equation to model alertness:

    f(t) = a * sin(t/12 * π) + b * cos(t/12 * π) + C

    This class performs cosine curve fitting along with error reporting
    and plotting.

    """

    def __init__(self, df, y_col):
        """
        Initiates the class.

        Parameters
        ----------

        df: DataFrame with `DateTimeIndex`.

        y_col: Column name that would be used for fitting value (output).
        """

        raise NotImplemented

    @staticmethod
    def _get_formatted_df(df, y_col, bin_width_min=30):
        """
        Formats data for subsequent use.

        This function ensures that X-Axis values are in hour scale.
        """

        raise NotImplemented

    def perform_fit(self, p0=None):
        return scipy.optimize.curve_fit(CosineCurveFitting._fit_function,
                                        xdata=self.X, ydata=self.Y, p0=p0)

    @staticmethod
    def _fit_function(t, a, b):
        """
        Fit function to be used passed to curve_fit.

        """

        arg = t/12 * np.pi
        return a * np.sin(arg) + b * np.cos(arg)
