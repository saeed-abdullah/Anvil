# -*- coding: utf-8 -*-
"""
    anvil.test.location_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing location module

    :copyright: (c) 2015 by Saeed Abdullah.

"""
from anvil import location
import pandas as pd
import numpy as np
import unittest


class LocationUtilTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # lon, lat tuples
        locations = [(-76.501881, 42.443961),
                     (-76.502298, 42.444151),
                     (-76.502405, 42.443898),
                     (-76.504079, 42.443264),
                     (-76.505710, 42.442314),
                     (-76.506654, 42.441016),
                     (-76.480561, 42.444864),  # distance is 2 KM
                     (-76.479210, 42.445592)]

        g = []
        for l in locations:
            g.append({'longitude': l[0], 'latitude': l[1]})

        cls.location_df = pd.DataFrame(g)

    def test_do_location_clustering(self):
        clusters = location.do_location_clustering(self.location_df).labels_
        # The last cluster has only two samples
        expected_clusters = [0, 0, 0, 0, 0, 0, -1, -1]
        self.assertTrue(np.all(expected_clusters == clusters))

        # test min samples
        clusters = location.do_location_clustering(self.location_df,
                                                   min_samples=2).labels_
        # The last cluster has only two samples
        expected_clusters = [0, 0, 0, 0, 0, 0, 1, 1]
        self.assertTrue(np.all(expected_clusters == clusters))

        # test distance
        clusters = location.do_location_clustering(self.location_df,
                                                   eps=10).labels_
        # The last cluster has only two samples
        expected_clusters = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue(np.all(expected_clusters == clusters))

        clusters = location.do_location_clustering(self.location_df,
                                                   eps=0.001).labels_
        # The last cluster has only two samples
        expected_clusters = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.assertTrue(np.all(expected_clusters == clusters))

    def test_daily_location_cluster_count(self):
        df = self.location_df.copy()

        # same date
        df['date'] = pd.to_datetime('2016-05-18')
        df = df.set_index('date')
        r = location.daily_location_cluster_count(df, min_samples=2)

        self.assertEqual(len(r), 1)
        self.assertEqual(r.date.values, pd.to_datetime('2016-05-18').date())

        self.assertEqual(r.cluster.values[0], 2)

        # Across dates

        df['date'] = pd.to_datetime('2016-05-18')
        df.date.iloc[-2:] = pd.to_datetime('2016-05-19')
        df = df.set_index('date')
        r = location.daily_location_cluster_count(df, min_samples=2)

        self.assertEqual(len(r), 2)
        self.assertEqual(r.date.values[0], pd.to_datetime('2016-05-18').date())
        self.assertEqual(r.date.values[1], pd.to_datetime('2016-05-19').date())

        self.assertEqual(r.cluster.values[0], 1)
        self.assertEqual(r.cluster.values[1], 1)

        # now, if the min_samples threshold is increased, the last value should
        # be zero.

        r = location.daily_location_cluster_count(df, min_samples=3)

        self.assertEqual(len(r), 2)
        self.assertEqual(r.date.values[0], pd.to_datetime('2016-05-18').date())
        self.assertEqual(r.date.values[1], pd.to_datetime('2016-05-19').date())

        self.assertEqual(r.cluster.values[0], 1)
        self.assertEqual(r.cluster.values[1], 0)
