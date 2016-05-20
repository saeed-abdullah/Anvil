# -*- coding: utf-8 -*-
"""
    anvil.misc
    ~~~~~~~~~~

    Collection of miscellaneous utilities

    :copyright: (c) 2016 by Saeed Abdullah.

"""

import pandas as pd


def convert_time_zone(df, column_name=None, should_localize='UTC',
                      sort_index=True,
                      to_timezone='America/New_York'):
    """
    Performs timezone conversion.

    This function does two things:

        1. Convert timezone of the given column (or index).
        2. And, create a DataFrame with the converted timestamps as index.

    Parameters
    ----------

    df : DataFrame

    column_name : str
        If a column should be used instead of the index. If the
        `column_name` is not None, then the column would be set as index
        after converting to date-time values using `pd.to_datetime`.

    sort_index: bool
        If the index of the resulting DataFrame
        should be sorted ascending order. Default is `True`.

    should_localize: str
        If the index should be localized to a specific time zone.
        If the value is `None`, then no localization is performed.
        Default is UTC.


    to_timezone : str
        The destination timezone. Default is America/New_York.


    Returns
    -------
    df : DataFrame
        DataFrame with converted timestamps as index values
    """

    if column_name is not None:
        df = df.set_index(pd.to_datetime(df[column_name]))

    if sort_index:
        df = df.sort_index()

    if should_localize is not None:
        df = df.tz_localize(should_localize)

    return df.tz_convert(to_timezone)


def get_df_slices(df, sorted_slices):
    """
    Gets DataFrame slices.

    This function performs advanced indexing
    of DataFrame based on given slices. For
    every two consecutive elements in the slice,
    the index is compared:
    slice[i] <= index < slice[i + 1] and matching
    rows are returned.

    For example, this function can be used to group
    rows in same week by having a DataFrame with
    `DateTimeIndex` and slices containing DateTime
    elements 7 days apart.

    Parameters
    ----------

    df : DataFrame

    sorted_slices : iterables
        List of sorted (ascending) slicing
        elements which are comparable against
        the given DataFrame index.


    Returns
    -------
    generator
        Rows matching to the slices. In other words,
        i-th iteration of the generator object will
        result in rows r such that:
        sorted_slices[i] >= r.index < sorted_slices[i + 1].

    Notes
    -----
        Slice elements must be sorted (ascending) and comparable
        against index of DataFrame.

    """

    for index in range(0, len(sorted_slices) - 1):
        s = sorted_slices[index]
        e = sorted_slices[index + 1]
        yield df[df.index.map(lambda z:  z >= s and z < e)]


def _sd_based_outlier_filtering(col, factor=1.5):
    """
    Performs SD based outlier filtering.

    It uses mean ± factor * SD as the threshold window. Any values
    outside of the window is considered as outlier.

    To use with different window size in outlier_filtering,
    `functools.partial` might be useful.

    Parameters
    ---------
    col : Series
        Filtering column
    factor : float
        Threshold window size. Default is 1.5

    Returns
    -------
    Series
        A Boolean Series where False indicates outlier values.
    """

    threshold = col.std() * factor
    min_val, max_val = col.mean() - threshold, col.mean() + threshold
    return col.map(lambda z: min_val < z < max_val)


def outlier_filtering(df, filtering_col,
                      filtering_f,
                      is_recursive=True):
    """
    Filters outlier from the given DataFrame.

    The filtering function takes the column as input and returns
    Boolean value for each row. Only rows with True values
    are retained.

    Parameters
    ----------

    df : DataFrame.

    filtering_col : str
        Filtering column name. It should contain comparable values

    filtering_f : function
        Filtering function. This function should take
        the filtering column and return Boolean value
        for each row with `False` values indicating outliers
        that should be discarded.

    is_recursive : bool
        If the filtering should be recursively applied
        until all values are consistent. Default is True.

    Returns
    -------
    DataFrame
        A filtered DataFrame.
    """

    col = df[filtering_col]

    df2 = df[filtering_f(col)]

    if is_recursive:
        # Check if all the values are consistent (no filtering would
        # happen in that case)
        if len(df) == len(df2):
            return df2
        else:
            return Utils.outlier_filtering(df2, filtering_col,
                                           filtering_f, is_recursive)
    else:
        return df2
