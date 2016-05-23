# -*- coding: utf-8 -*-
"""
    anvil.circadian
    ~~~~~~~~~~~~~~~

    Collection of circadian utilities

    :copyright: (c) 2015 by Saeed Abdullah.

"""


import datetime as dt
import pandas as pd


"""
Utility functions for circadian analysis.
"""


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
    df : DataFrame.
    value_col : str
        Column to calculate daily stability.
    hour_col : str
        Column indicating hourly values.

    Returns
    -------
    float
        Value indicating inter daily stability.

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


def intra_daily_variability(df, value_col):
    """
    Calculate intra-daily variability (IV).

    High IV indicates fragmented day. The implementation
    follows the formula (1) from Witting et al. (pg. 565).


    Parameters
    ----------
    df : DataFrame.
        It must be sorted by date (ascending).
    value_col : str
        Column to compute daily variability.

    Returns
    -------
    float
        Computed variability score.

    """

    s = (df[value_col] - df[value_col].shift(1))**2
    nom = len(df) * sum(s.fillna(0).values)

    mean = df[value_col].mean()
    N = len(df)

    denom = (N - 1) * sum((df[value_col] - mean)**2)

    return nom/denom


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

    df : DataFrame
    value_col : str
        Column to compute average values.
    hour_col : str
        Column denoting hours. Default is 'hour'.


    Returns
    -------
    list
        A sorted list (ascending) of tuples (h, v)
        with first element is the hour and second
        element is average value.
    """

    d = {}
    for k, v in df.groupby(hour_col):
        d[k] = v[value_col].mean()

    # sort by value
    return sorted(d.items(), key=lambda z: z[1])


def _convert_timestamp_to_decimal(timeseries,
                                  should_convert=False):
    """
    Convert timestamp to decimal value for SRM.

    Conversion happens using hour + minute/60 formula,
    so, 08:45 would be 8.75.

    Parameters
    ----------

    timeseries : Series
        A series with timestamp or datetimes.
    should_convert : bool
        If the timeseries should be converted
        using `pd.to_datetime` function.

    Returns
    -------
    Series
        Series containing converted values in decimal.
    """

    if should_convert:
        timeseries = pd.to_datetime(timeseries)
    return timeseries.map(lambda z: z.hour + z.minute/60)


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

    series: Series
        SRM scores.
    lower_limit : float
        Lower limit.
    upper_limit : float
        Upper limit.

    Returns
    -------
    Series
        A new series with valid points only.
    """

    filtering_f = lambda z: z >= lower_limit and z <= upper_limit
    return series[series.map(filtering_f)]


def _srm_preprocssing(series):
    """
    Pre-processing for SRM calculation.

    This function does the followings:
    1. Converts timestamp to decimal values
    2. Removes outliers with > 1.5 * SD.

    Parameters
    ----------
    series: Series
        Timestamps of a given SRM event.

    Returns
    -------
    Series
        A new series with timestamps converted to
        decimals and outliers purged.
    """

    series = _convert_timestamp_to_decimal(series)
    mean = series.mean()
    std = series.std()

    if std < 0.5:
        return series

    lower_limit = mean - 1.5 * std
    upper_limit = mean + 1.5 * std
    series = _purge_srm_outliers(series, lower_limit=lower_limit,
                                 upper_limit=upper_limit)

    return series


def _calculate_srm_hit(series,
                       lower_limit,
                       upper_limit):
    """
    Calculates number of hits.

    Hit is defined as the number of values z falling within
    the given range: lower_limit >= z <= upper_limit.

    Parameters
    ----------
    series : Series
       Decimal values indicating the time of event (after
       converting using `_convert_timestamp_to_decimal`).
    lower_limit : float
        Lower limit

    Returns
    -------
    int
        Number of hits.
    """

    filtering_f = lambda z: z >= lower_limit and z <= upper_limit

    return sum(series.map(filtering_f))


def calculate_srm(df, target_col,
                  time_col='completion_time',
                  min_samples=3):
    """
    Calculates SRM score.

    Parameters
    ----------

    df : DataFrame.
    target_col : str
        Column with target names. Individual hits would be
        calculated for each targets.
    time_col : str
        Column containing timestamps. Default is 'completion_time'.
    min_samples : int
        Minimum samples for calculating hit
        for a given column. Default is 3 (40%
        of a week).

    Returns
    -------
    float
        Value within [0, 7] range indicating overall SRM stability.
    """

    hit_range = 45/60  # "hit" if falls within 45 minute

    l = []

    for k, v in df.groupby(target_col):
        series = v.loc[:, time_col]
        series = _srm_preprocssing(series)
        if len(series) >= min_samples:

            mean = series.mean()

            lower = mean - hit_range
            upper = mean + hit_range

            hit = _calculate_srm_hit(series, lower_limit=lower,
                                     upper_limit=upper)
            l.append(hit)

    return sum(l)/len(l)


def _calculate_srm_across_users(df,
                                user_col='user_id',
                                **srm_args):
    """
    Calculates SRM score across users.

    Parameters
    ----------
    df : DataFrame
    user_col : str
        User id column. Default is 'user_id'.
    **srm_args
        Variable args. See `calculate_srm` for options.

    Returns
    -------
    DataFrame
        A DataFrame with user_id and srm columns.
    """

    l = []
    for k, v in df.groupby(user_col):
        srm = calculate_srm(v, **srm_args)
        l.append({'user_id': k, 'srm': srm})

    return pd.DataFrame(l)


def rolling_srm_across_users(df, start_date,
                             how_many_days,
                             time_col='completion_time',
                             **srm_args):
    """
    Calculates rolling SRM across days for given days.

    Parameters
    ----------
    df : DataFrame
    start_date : DateTime
    how_many_days : int
        Number of days for which the rolling SRM
        should be calculated. In other words, SRM
        would be calculated for each week starting
        at d where start_date <= d <= start_date + how_many_days.
    time_col : str
        Column indicating completion time. Default is 'completion_time'.
    **srm_args
        Variable args. For options, see `calculate_srm`.

    Returns
    -------
    DataFrame
        A DataFrame with user_id, srm, date columns. The date column
        indicate the first day of each week on which SRM has
        been calculated.
    """
    srm_df = None
    for i in range(how_many_days):
        s = start_date + dt.timedelta(days=i)
        e = s + dt.timedelta(days=7)

        w = df[df[time_col].map(lambda z: z >= s and z < e)]
        df_w = _calculate_srm_across_users(w, time_col=time_col, **srm_args)

        df_w['date'] = s.date()
        srm_df = pd.concat([srm_df, df_w], ignore_index=True)

    return srm_df
