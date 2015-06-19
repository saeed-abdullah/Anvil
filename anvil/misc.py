# -*- coding: utf-8 -*-
"""
    anvil.misc
    ~~~~~~~~~~

    Collection of miscellaneous utilities

    :copyright: (c) 2015 by Saeed Abdullah.

"""
import pandas as pd


class Utils(object):

    @staticmethod
    def convert_time_zone(df, column_name=None, should_localize='UTC',
                          sort_index=True,
                          to_timezone='America/New_York'):
        """
        Converts a DataFrame to specified timezone.

        :param df: Given DataFrame.

        :param column_name: If a column should be used instead of
                            current index. If the `column_name` is
                            not None, then the column_would be set
                            as index after converting to date-time
                            values using `pd.to_datetime`.

        :param sort_index: If the index should be sorted. Default
                           is `True`.

        :param should_localize: If the index should be localized to
                                a specific time zone. The value is
                                either `None` or timezone name.

        :param to_timezone: The destination timezone. Default is
                            America/New_York.


        :returns:: A DataFrame with index converted to given timezone.
        """

        if column_name is not None:
            df = df.set_index(pd.to_datetime(df[column_name]))

        if sort_index:
            df = df.sort_index()

        if should_localize is not None:
            df = df.tz_localize(should_localize)

        return df.tz_convert(to_timezone)

    @staticmethod
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

        df: DataFrame.
        sorted_slices: List of sorted (ascending) slicing
                       elements which are comparable against
                       DataFrame index.


        Returns
        -------
        Generator producing list of rows.

        Notes
        -----

        Slice elements must be sorted (ascending) and comparable
        against index of DataFrame.

        """

        for index in range(0, len(sorted_slices) - 1):
            s = sorted_slices[index]
            e = sorted_slices[index + 1]
            yield df[df.index.map(lambda z:  z >= s and z < e)]

    @staticmethod
    def outlier_filtering(df, filtering_col,
                          filtering_factor=1.5,
                          is_recursive=True):
        """
        Filters outlier from the given DataFrame.

        The threshold value is determined as mean ± filtering_factor * SD.
        Any row with value greater or less than the threshold in filtering
        column is removed.

        :param df: DataFrame.
        :param filtering_col: Filtering column. It should contain comparable
                              values
        :param filtering_factor: Filtering factor. This determines the
                                 threshold window width. Default is 1.5.
        :param is_recursive: If the filtering should be recursively applied
                             until all values are consistent. Default is True.
        :return: A filtered DataFrame.
        """

        col = df[filtering_col]
        threshold = col.std() * filtering_factor
        min_val, max_val = col.mean() - threshold, col.mean() + threshold

        df2 = df[col.map(lambda z: min_val < z < max_val)]

        if is_recursive:
            # Check if all the values are consistent (no filtering would
            # happen in that case)
            if len(df) == len(df2):
                return df2
            else:
                return Utils.outlier_filtering(df2, filtering_col,
                                               filtering_factor, is_recursive)
        else:
            return df2
