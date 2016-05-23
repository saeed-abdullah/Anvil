# -*- coding: utf-8 -*-
"""
    anvil.test.utils_test
    ~~~~~~~~~~~~~~~~~~~~~

    Unit testing utils module

    :copyright: (c) 2015 by Saeed Abdullah.

"""

from anvil import utils
import datetime as dt
from functools import partial
import numpy as np
import pandas as pd
import unittest


class UtilsTest(unittest.TestCase):
    def test_convert_timezone(self):
        timezone = "America/Los_Angeles"
        rng = pd.date_range('1/1/2011', periods=14, freq='1H')
        ts = pd.DataFrame(pd.np.random.randn(len(rng)), index=rng)

        # todo: check other parameters
        converted_rng = utils.convert_time_zone(ts, should_localize="UTC",
                                                to_timezone=timezone)
        self.assertEquals(converted_rng.index.tz.zone, timezone)

    def test_get_df_slices(self):
        rng = pd.date_range('1/1/2011', periods=14, freq='D')
        ts = pd.DataFrame(pd.np.random.randn(len(rng)), index=rng)
        slices = pd.date_range('1/1/2011', periods=3, freq='7D')

        l = list(utils.get_df_slices(ts, slices))
        self.assertEquals(len(l), len(slices) - 1)

        for x in l:
            self.assertEquals(len(x), 7)

        self.assertEquals(l[0].index[0], slices[0])

    def test_outlier_filtering(self):
        l = [11171.0, 119425.0, 270.5, 250.0, 258.5]
        df = pd.DataFrame(l, columns=["x"])
        filtering_f = partial(utils._sd_based_outlier_filtering,
                              factor=1.2)
        df2 = utils.outlier_filtering(df, filtering_col="x",
                                      filtering_f=filtering_f,
                                      is_recursive=True)
        expected = sorted(l)[:-2]
        self.assertEquals(len(df2), len(expected))
        self.assertEquals(expected[0], df2.x.min())
        self.assertEquals(expected[-1], df2.x.max())

        filtering_f = partial(utils._sd_based_outlier_filtering,
                              factor=2)
        df2 = utils.outlier_filtering(df, filtering_col="x",
                                      filtering_f=filtering_f,
                                      is_recursive=True)
        self.assertEquals(len(df2), len(l))
        self.assertEquals(np.min(l), df2.x.min())
        self.assertEquals(np.max(l), df2.x.max())

        filtering_f = lambda z: z <= 250.0
        df2 = utils.outlier_filtering(df, filtering_col="x",
                                      filtering_f=filtering_f,
                                      is_recursive=True)
        self.assertEquals(len(df2), 1)
        self.assertEquals(df2.x.min(), 250.0)

    def test_get_hourly_distribution(self):
        rng = pd.date_range('1/1/2011', periods=48, freq='H')
        df = pd.DataFrame({'item': list(range(0, 48))}, index=rng)

        func = lambda z: {'avg': z.item.mean(), 'max': z.item.max()}

        r = utils.get_hourly_distribution(df, func)

        self.assertEquals(len(r), 48)
        # the first 24 values are for first date
        self.assertEquals(np.unique(r['date'].values[:24]),
                          [dt.date(2011, 1, 1)])

        # the second 24 values are for next day
        self.assertEquals(np.unique(r['date'].values[-24:]),
                          [dt.date(2011, 1, 2)])

        self.assertTrue(np.all(r.avg == r['max']))
        self.assertTrue(np.all(r.avg.values == df.item.values))

        # now repeat each value twice in a given hour

        df = df.asfreq('30Min', method='bfill')

        r = utils.get_hourly_distribution(df, func)

        self.assertEquals(len(r), 48)
        # the first 24 values are for first date
        self.assertEquals(np.unique(r['date'].values[:24]),
                          [dt.date(2011, 1, 1)])

        # the second 24 values are for next day
        self.assertEquals(np.unique(r['date'].values[-24:]),
                          [dt.date(2011, 1, 2)])

        # the values have been back filled. So, the difference
        # is always 0.5 except for the last one.
        diff = r['max'] - r.avg
        expected = [0.5] * (len(diff) - 1)
        expected.append(0)  # the last value is zero
        self.assertTrue(np.all(np.isclose(diff, expected)))
