from analysis import np_utils
import numpy as np


def test_bfill():
    x = np.array([np.nan, 2, np.nan, 10, np.nan])
    fill_idx, unfilled = np_utils.bfill(np.isnan(x))
    result = x[fill_idx]
    np.testing.assert_equal(result, [2, 2, 10, 10, np.nan])
    np.testing.assert_equal(unfilled, [False, False, False, False, True])


def test_ffill_second_index():
    x = np.array([10, np.nan, np.nan, 20, 30, np.nan, 10])
    fill_idx, unfilled = np_utils.ffill(np.isnan(x))
    np.testing.assert_equal(fill_idx, np.array([0, 0, 0, 3, 4, 4, 6]))
    np.testing.assert_equal(
        unfilled, np.array([False, False, False, False, False, False, False])
    )


def test_ffill_forward_propogation():
    x = np.array([np.nan, np.nan, 20, 30, np.nan, 10])
    fill_idx, unfilled = np_utils.ffill(np.isnan(x))
    expected = [
        0,
        1,
        2,
        3,
        3,
        5,
    ]
    np.testing.assert_equal(
        fill_idx,
        expected,
    )
    np.testing.assert_equal(
        unfilled,
        [
            True,
            True,
            False,
            False,
            False,
            False,
        ],
    )


def test_find_nearest_sanity():
    points = np.array([10, 12, np.nan, 10, np.nan, np.nan, np.nan, np.nan, np.nan, 20])
    positions = np.arange(len(points))

    source, target = np_utils.find_nearest_valid_neighbor(
        np.isnan(points), positions, 1
    )

    points[source] = points[target]
    np.testing.assert_equal(
        points, np.array([10, 12, 12, 10, 10, np.nan, np.nan, np.nan, 20, 20])
    )


def test_find_nearest_safe():
    points = np.array([10, np.nan, 12, np.nan, np.nan, 13])
    positions = np.arange(len(points))

    source, target = np_utils.find_nearest_valid_neighbor(
        np.isnan(points), positions, 1
    )

    points[source] = points[target]
    np.testing.assert_equal(points, np.array([10, 10, 12, 12, 13, 13]))
