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
    def _convert_time_zone(df, column_name=None, should_localize='UTC',
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
