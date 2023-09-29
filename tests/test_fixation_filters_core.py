from analysis.fixation_filters.core import (
    GazeVector,
    ivt,
    nn_interpolator,
)
import numpy as np
import pytest


def parametrize(*test_cases):
    keys = test_cases[0].keys()

    for test_case in test_cases:
        assert test_case.keys() == keys, "All test cases must have the same keys"

    values = [test_case.values() for test_case in test_cases]
    return pytest.mark.parametrize(
        ",".join(keys),
        values,
    )


@parametrize(
    dict(
        coords=[10, 11, 12, 14, 16, 18],
        ts=[1, 2, 3, 4, 5, 6],
        mask=None,
        expected=[False, True, True, False, False, False],
        velocity_threshold=1,
    ),
    dict(
        coords=[10, 11, 12, 14, 16, 18],
        ts=[1, 2, 3, 4, 5, 6],
        mask=None,
        expected=[False, True, True, True, True, True],
        velocity_threshold=2,
    ),
    dict(
        coords=[10, 11, 12, 14, 16, 18],
        ts=[1, 2, 3, 4, 5, 6],
        mask=[True, True, False, True, True, False],
        expected=[False, True, False, True, True, False],
        velocity_threshold=2,
    ),
    dict(
        coords=[
            [1, 2, 3, 4, 5],
            [2, 4, 7, 8, 10],
        ],
        ts=[1, 2, 3, 4, 5],
        mask=None,
        expected=[False, True, False, True, True],
        velocity_threshold=3,
    ),
)
def test_ivt(coords, ts, mask, expected, velocity_threshold):
    coords = np.array(coords)
    ts = np.array(ts)
    if mask is None:
        mask = np.ones(len(ts), dtype=bool)
    gazes = GazeVector(coords, ts, mask=mask)
    result = ivt(gazes, velocity_threshold=velocity_threshold)

    np.testing.assert_array_equal(result.is_fixation, expected)


@parametrize(
    dict(
        coords=[10, np.nan, 12, np.nan, 16, 18],
        expected=[10, 10, 12, 12, 16, 18],
        radius=1,
    ),
    dict(
        coords=[10, np.nan, np.nan, np.nan, 12, np.nan, 16, 18],
        expected=[10, 10, np.nan, 12, 12, 12, 16, 18],
        radius=1,
    ),
)
def test_nn_interpolator(coords, expected, radius):
    coords = np.array(coords)
    ts = np.arange(len(coords))
    mask = ~np.isnan(coords)
    gazes = GazeVector(coords, ts, mask=mask)
    interpolation_result = nn_interpolator(gazes, radius=radius)

    result = coords.copy()
    result[interpolation_result.source] = result[interpolation_result.target]

    np.testing.assert_array_equal(result, expected)
