# -*- coding: utf-8 -*-
"""
    anvil.test.ciradian_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing circadian module

    :copyright: (c) 2015 by Saeed Abdullah.

"""

from anvil import misc
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
