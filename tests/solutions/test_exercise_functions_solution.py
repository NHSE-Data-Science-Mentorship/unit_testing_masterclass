from learn_unit_testing.exercise_functions import (
    mean_squared_error,
    imputation,
)
import pytest
import numpy as np


# Data for MSE. We make sure to include nans in the ground truth and
PREDS_1 = np.array([0, 1, 2, 3, 4, 5])
PREDS_2 = np.array([0, 1, 2, 3, 4, np.nan])
PREDS_3 = np.array([0, 1, 2, 3, 4, np.nan])

GROUND_TRUTH_1 = np.array([1, 1, 1, 1, 1, 1])
GROUND_TRUTH_2 = np.array([1, 1, 1, 1, 1, 1])
GROUND_TRUTH_3 = np.array([1, 1, 1, 1, np.nan, 1])

EXPECTED_MSE_1 = 31 / 6
EXPECTED_MSE_2 = 15 / 5
EXPECTED_MSE_3 = 6 / 4

# Data for imputation. Make sure to test invalid cases and the None case.
TIMESERIES = np.array([0, 1, 2, 3, np.nan, 5])

SEASONALITY = 1
SEASONALITY_NONE = None
SEAOSNALITY_INVALID = -1

EXPECTED_IMPUTED = np.array([0.0, 1.0, 2.0, 3.0, 3.0, 5.0])
EXPECTED_IMPUTED_NONE = np.array(
    [0.0, 1.0, 2.0, 3.0, (1 + 2 + 3 + 5) / 5, 5.0]
)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (PREDS_1, GROUND_TRUTH_1, EXPECTED_MSE_1),
        (PREDS_2, GROUND_TRUTH_2, EXPECTED_MSE_2),
        (PREDS_3, GROUND_TRUTH_3, EXPECTED_MSE_3),
    ],
)
def test_mean_squared_error(y_true, y_pred, expected):
    res = mean_squared_error(y_true, y_pred)
    assert expected == res


@pytest.mark.parametrize(
    "timeseries, seasonality, expected",
    [
        (TIMESERIES, SEASONALITY, EXPECTED_IMPUTED),
        (TIMESERIES, SEASONALITY_NONE, EXPECTED_IMPUTED_NONE),
    ],
)
def test_imputation(timeseries, seasonality, expected):
    res = imputation(timeseries, seasonality)
    np.testing.assert_array_equal(expected, res)


def test_imputation_invalid():
    with pytest.raises(ValueError):
        imputation(TIMESERIES, SEAOSNALITY_INVALID)
