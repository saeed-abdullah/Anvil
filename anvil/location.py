# -*- coding: utf-8 -*-
"""
    anvil.location
    ~~~~~~~~~~~~~~

    Collection of location utilities

    :copyright: (c) 2015 by Saeed Abdullah.

"""

import geopy
import numpy as np
import pandas as pd
import scipy
import sklearn


class LocationUtil(object):
    """
    Location utility class.

    Uses code from http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    """

    @staticmethod
    def _perform_clustering(df, eps=None,
                            min_samples=None,
                            metric=None):
        """
        Performs location based clustering.

        Parameters
        ----------

        df: DataFrame with `longitude` and `latitude` columns.
        eps: Maximum distance between two points to be in the
             same cluster. Default is 1.0 (km).
        min_samples: The minimum number of points in a cluster.
                      Default is 3.
        metric: Pairwise distance calculator between two points.
                Default is the vincent distance from `geopy`.

        Note
        ----

        If you are updating either eps or metric, you probably
        want to update the other parameter as well. For example,
        if the metric is "ellipsoid", then eps should be changed
        as well (the default value 1.0 might be too large).

        Returns
        -------
        A `DBSCAN` object.

        """

        if eps is None:
            eps = 1.0  # 1.0 KM following the default metric.

        if min_samples is None:
            min_samples = 3

        c_matrix = df.as_matrix(columns=['longitude', 'latitude'])

        # Pre-computed distance matrix where (i, j) entry
        # denotes the distance between point i and j in km.
        if metric is None:
            v = scipy.spatial.distance.pdist(c_matrix,
                                             lambda x, y: geopy.distance.distance(x, y).km)
            c_matrix = scipy.spatial.distance.squareform(v)
            metric = 'precomputed'

        return sklearn.cluster.DBSCAN(eps=eps,
                                      metric=metric,
                                      min_samples=min_samples).fit(c_matrix)

    @staticmethod
    def daily_location_cluster_count(df):
        """
        Counts number of location cluster in a day.

        Parameters
        ----------
        df: DataFrame with `longitude` and `latitude` columns.

        Returns
        -------
        A DataFrame with user_id, date and cluster columns.
        """
        l = []
        for k, v in df.groupby('user_id'):
            for k1, v1 in v.groupby(lambda z: z.date()):
                clusters = LocationUtil._perform_clustering(v1,
                                                            eps=0.5,
                                                            min_samples=2).labels_
                # -1 indicates noise, so we do not want to count that
                num_clusters = len(np.unique(clusters)) - (-1 in clusters)
                l.append({'user_id': k, 'date': k1, 'cluster': num_clusters})

        return pd.DataFrame(l)
