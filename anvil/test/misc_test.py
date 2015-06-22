# -*- coding: utf-8 -*-
"""
    anvil.test.ciradian_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing circadian module

    :copyright: (c) 2015 by Saeed Abdullah.

"""

from anvil import misc
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
        converted_rng = misc.Utils.convert_time_zone(ts, should_localize="UTC",
                                                     to_timezone=timezone)
        self.assertEquals(converted_rng.index.tz.zone, timezone)

    def test_get_df_slices(self):
        rng = pd.date_range('1/1/2011', periods=14, freq='D')
        ts = pd.DataFrame(pd.np.random.randn(len(rng)), index=rng)
        slices = pd.date_range('1/1/2011', periods=3, freq='7D')

        l = list(misc.Utils.get_df_slices(ts, slices))
        self.assertEquals(len(l), len(slices) - 1)

        for x in l:
            self.assertEquals(len(x), 7)

        self.assertEquals(l[0].index[0], slices[0])

    def test_outlier_filtering(self):
        l = [11171.0, 119425.0, 270.5, 250.0, 258.5]
        df = pd.DataFrame(l, columns=["x"])
        filtering_f = partial(misc.Utils._sd_based_outlier_filtering,
                              factor=1.2)
        df2 = misc.Utils.outlier_filtering(df, filtering_col="x",
                                           filtering_f=filtering_f,
                                           is_recursive=True)
        expected = sorted(l)[:-2]
        self.assertEquals(len(df2), len(expected))
        self.assertEquals(expected[0], df2.x.min())
        self.assertEquals(expected[-1], df2.x.max())

        filtering_f = partial(misc.Utils._sd_based_outlier_filtering,
                              factor=2)
        df2 = misc.Utils.outlier_filtering(df, filtering_col="x",
                                           filtering_f=filtering_f,
                                           is_recursive=True)
        self.assertEquals(len(df2), len(l))
        self.assertEquals(np.min(l), df2.x.min())
        self.assertEquals(np.max(l), df2.x.max())

        filtering_f = lambda z: z <= 250.0
        df2 = misc.Utils.outlier_filtering(df, filtering_col="x",
                                           filtering_f=filtering_f,
                                           is_recursive=True)
        self.assertEquals(len(df2), 1)
        self.assertEquals(df2.x.min(), 250.0)
