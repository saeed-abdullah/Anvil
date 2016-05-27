# -*- coding: utf-8 -*-
"""
    anvil.location
    ~~~~~~~~~~~~~~

    Collection of location utilities

    :copyright: (c) 2015 by Saeed Abdullah.

"""

from geopy.distance import vincenty, great_circle
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import cluster


"""
Location utilities.

Uses code from http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
"""


def do_location_clustering(df, eps=None, min_samples=None,
                           metric=None, lat_c='latitude',
                           lon_c='longitude', distance_method='vincenty'):
    """
    Performs location based clustering.

    Parameters
    ----------

    df : DataFrame
        DataFrame with latitude and longitude information.

    eps : float
        Maximum distance between two points to be in the
        same cluster. Default is 1.0 (km).

    min_samples: int
        The minimum number of points in a cluster.
        Default is 3.

    metric : Pairwise distance calculator between two points.
        If `None`, the vincent distance is used. Default is
        None.

    lat_c : str
        Column name for latitude data.

    lon_c : str
        Column name for longitude data.

    distance_method : str
        Distance calculation method to use. The options are
        'vincenty' or 'great_circle'.

    Returns
    -------

    sklearn.cluster.DBSCAN
        The attribute `labels_` contains cluster label for each
        given point.


    Notes
    -----

        If you are updating either eps or metric, you probably
        want to update the other parameter as well. For example,
        if the metric is "ellipsoid", then eps should be changed
        as well (the default value 1.0 might be too large).

    """

    if eps is None:
        eps = 1.0  # 1.0 KM following the default metric.

    if min_samples is None:
        min_samples = 3

    c_matrix = df.as_matrix(columns=[lon_c, lat_c])

    if distance_method == 'vincenty':
        geodesic_distance = vincenty
    elif distance_method == 'great_circle':
        geodesic_distance = great_circle
    else:
        raise ValueError('Unknown distance method: {0}. Must be '
                         'either vincenty or great_circle')

    # Pre-computed distance matrix where (i, j) entry
    # denotes the distance between point i and j in km.
    if metric is None:
        v = spatial.distance.pdist(c_matrix,
                                   lambda x, y: geodesic_distance(x, y).km)
        c_matrix = spatial.distance.squareform(v)
        metric = 'precomputed'

    return cluster.DBSCAN(eps=eps,
                          metric=metric,
                          min_samples=min_samples).fit(c_matrix)


def daily_location_cluster_count(df, lat_c="latitude",
                                 lon_c="longitude", **kwargs):
    """
    Counts number of location cluster in a day.

    Parameters
    ----------
    df : DataFrame
        DataFrame with DateTimeIndex. The index would be used for
        grouping rows by dates.

    lat_c : str
        Column name for latitude data.

    lon_c : str
        Column name for longitude data.

    **kwargs
        Keyword arguments that will be passed to `do_location_clustering`.


    Returns
    -------
    DataFrame
        It contains date and cluster columns.

    """
    l = []
    for k, v in df.groupby(lambda z: z.date()):
        # Get cluster labels for each data points
        clusters = do_location_clustering(v, **kwargs).labels_
        # -1 indicates noise, so we do not want to count that
        num_clusters = len(np.unique(clusters)) - (-1 in clusters)
        l.append({'date': k, 'cluster': num_clusters})

    return pd.DataFrame(l)
