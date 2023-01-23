"""Test suite for skeleton module
"""
from unittest.mock import patch

import pytest

import numpy as np
import pandas as pd
from learn_unit_testing.functions import (
    Pipeline,
    add_group_mean,
    add_group_median,
    add_norm_last_value,
    preprocess,
    train_test_split,
)

# PART I: Simple testing

# Create data

DF_SIZE = 100
TO_SPLIT_DF = pd.DataFrame(
    {
        "col_a": np.random.randint(1, 100, DF_SIZE),
        "col_b": np.random.uniform(0, 1, DF_SIZE),
    }
)
TEST_FRAC = 0.2

EXPECTED_TEST_SIZE = int(TEST_FRAC * DF_SIZE)
EXPECTED_TRAIN_SIZE = DF_SIZE - EXPECTED_TEST_SIZE


# def test_train_test_split():
#     train_df, test_df = train_test_split(TO_SPLIT_DF, test_frac=TEST_FRAC)

#     assert len(train_df) == EXPECTED_TRAIN_SIZE
#     assert len(test_df) == EXPECTED_TEST_SIZE


# # PART II: parametric tests

# GROUPS_1 = ["A", "A", "A", "B", "B", "B"]
# GROUPS_2 = ["A", "A", "A", "B", "B", "B"]

# VALUES_1 = [4, 4, 1, 0, 3, 3]
# VALUES_2 = [4, 4, 1, 0, 0, 15]

# TO_PREPROCESS_DF_1 = pd.DataFrame({"group": GROUPS_1, "value": VALUES_1})
# TO_PREPROCESS_DF_2 = pd.DataFrame({"group": GROUPS_2, "value": VALUES_2})

# EXPECTED_MEANS_1 = [3, 3, 3, 2, 2, 2]
# EXPECTED_MEANS_2 = [4, 4, 4, 5, 5, 5]

# EXPECTED_MEDIANS_1 = [4, 4, 4, 3, 3, 3]
# EXPECTED_MEDIANS_2 = [3, 3, 3, 0, 0, 0]

# # We've added a second keyword argument

# MEDIAN_WEIGHT_A = 0
# MEDIAN_WEIGHT_B = 1
# MEDIAN_WEIGHT_C = -1

# EXPECTED_NORMALISED_VALUE_1A = [7, 7, 4, 2, 5, 5]
# # EXPECTED_NORMALISED_VALUE_2A = [6, 6, 3, 5, 5, 20]
# EXPECTED_NORMALISED_VALUE_2A = [7, 7, 4, 5, 5, 20]
# EXPECTED_NORMALISED_VALUE_1B = [3, 3, 0, -1, 2, 2]
# EXPECTED_NORMALISED_VALUE_2B = [3, 3, 0, 5, 5, 20]
# EXPECTED_NORMALISED_VALUE_1C = [11, 11, 8, 5, 8, 8]
# EXPECTED_NORMALISED_VALUE_2C = [11, 11, 8, 5, 5, 20]


# # We could test each of our statements 1 by 1...
# def test_preprocess_boring():
#     np.testing.assert_array_equal(
#         preprocess(TO_PREPROCESS_DF_1, median_weight=MEDIAN_WEIGHT_A)[
#             "value_normalised"
#         ],
#         EXPECTED_NORMALISED_VALUE_1A,
#     )
#     np.testing.assert_array_equal(
#         preprocess(TO_PREPROCESS_DF_2, median_weight=MEDIAN_WEIGHT_A)[
#             "value_normalised"
#         ],
#         EXPECTED_NORMALISED_VALUE_2A,
#     )
#     np.testing.assert_array_equal(
#         preprocess(TO_PREPROCESS_DF_1, median_weight=MEDIAN_WEIGHT_B)[
#             "value_normalised"
#         ],
#         EXPECTED_NORMALISED_VALUE_1B,
#     )
#     np.testing.assert_array_equal(
#         preprocess(TO_PREPROCESS_DF_2, median_weight=MEDIAN_WEIGHT_B)[
#             "value_normalised"
#         ],
#         EXPECTED_NORMALISED_VALUE_2B,
#     )
#     np.testing.assert_array_equal(
#         preprocess(TO_PREPROCESS_DF_1, median_weight=MEDIAN_WEIGHT_C)[
#             "value_normalised"
#         ],
#         EXPECTED_NORMALISED_VALUE_1C,
#     )
#     np.testing.assert_array_equal(
#         preprocess(TO_PREPROCESS_DF_2, median_weight=MEDIAN_WEIGHT_C)[
#             "value_normalised"
#         ],
#         EXPECTED_NORMALISED_VALUE_2C,
#     )


# # But that's so boring! Instead let's parametrise the test
# @pytest.mark.parametrize(
#     "df,median_weight,expected",
#     [
#         (TO_PREPROCESS_DF_1, MEDIAN_WEIGHT_A, EXPECTED_NORMALISED_VALUE_1A),
#         (TO_PREPROCESS_DF_2, MEDIAN_WEIGHT_A, EXPECTED_NORMALISED_VALUE_2A),
#         (TO_PREPROCESS_DF_1, MEDIAN_WEIGHT_B, EXPECTED_NORMALISED_VALUE_1B),
#         (TO_PREPROCESS_DF_2, MEDIAN_WEIGHT_B, EXPECTED_NORMALISED_VALUE_2B),
#         (TO_PREPROCESS_DF_1, MEDIAN_WEIGHT_C, EXPECTED_NORMALISED_VALUE_1C),
#         (TO_PREPROCESS_DF_2, MEDIAN_WEIGHT_C, EXPECTED_NORMALISED_VALUE_2C),
#     ],
# )
# def test_preprocess(df, median_weight, expected):
#     np.testing.assert_array_equal(
#         preprocess(df, median_weight=median_weight)["value_normalised"],
#         expected,
#     )


# # Part III Assertion testing

# EXPECTED_NORM_LAST_RES_1 = pd.DataFrame(
#     {
#         "time": [1, 2, 3, 4, 5],
#         "value": [0, 3, 1, 4, 6],
#         "norm_last_value": [0, 0, 1, 0.25, 0.5],
#     }
# )
# EXPECTED_NORM_LAST_RES_2 = pd.DataFrame(
#     {
#         "time": [1, 5, 3, 4, 2],
#         "value": [0, 3, 2, 4, 6],
#         "norm_last_value": [0, 0.3333333333333, 1.0, 0.25, 0],
#     }
# )
# EXPECTED_NORM_LAST_RES_3 = pd.DataFrame(
#     {
#         "time": [1, 2, 3, 4, 5],
#         "value": [0, 3, np.nan, 3, 6],
#         "norm_last_value": [0, 0, 1.0, np.nan, 0.5],
#     }
# )

# NORM_LAST_RES_DF_1 = EXPECTED_NORM_LAST_RES_1[["time", "value"]]
# NORM_LAST_RES_DF_2 = EXPECTED_NORM_LAST_RES_2[["time", "value"]]
# NORM_LAST_RES_DF_3 = EXPECTED_NORM_LAST_RES_3[["time", "value"]]


# @pytest.mark.parametrize(
#     "df,expected",
#     [
#         (NORM_LAST_RES_DF_1, EXPECTED_NORM_LAST_RES_1),
#         (NORM_LAST_RES_DF_2, EXPECTED_NORM_LAST_RES_2),
#         (NORM_LAST_RES_DF_3, EXPECTED_NORM_LAST_RES_3),
#     ],
# )
# def test_add_norm_last_value(df, expected):
#     res = add_norm_last_value(df)

#     # The check_like argument allows the check to ignore ordering
#     pd.testing.assert_frame_equal(res, expected, check_like=True)


# # Warnings and Exceptions

# EXPECTED_NORM_LAST_RES_WARN = pd.DataFrame(
#     {
#         "time": [1, 5, 3, 4, 2],
#         "value": [1, 0, 0, 0, np.nan],
#     }
# )
# EXPECTED_NORM_LAST_RES_ERROR = pd.DataFrame(
#     {
#         "time": [1, 5, 3, 4, 2],
#         "value": [np.nan, np.nan, np.nan, np.nan, np.nan],
#     }
# )


# def test_add_norm_last_value_warning_error():
#     with pytest.warns(UserWarning):
#         add_norm_last_value(EXPECTED_NORM_LAST_RES_WARN)

#     with pytest.raises(ValueError):
#         add_norm_last_value(EXPECTED_NORM_LAST_RES_ERROR)


# # PART IV Mocking

# PREPROCESSED_DF_1 = TO_PREPROCESS_DF_1.copy()
# PREPROCESSED_DF_1["value_normalised"] = EXPECTED_NORMALISED_VALUE_1B


# # Careful with the ordering1
# @patch("learn_unit_testing.functions.Pipeline.load_data", autospec=True)
# @patch("learn_unit_testing.functions.Pipeline.slow_computation", autospec=True)
# def test_pipeline(
#     slow_computation_method,
#     load_data_method,
# ):

#     # We 'hack' the load data function to return our data from before
#     load_data_method.return_value = TO_PREPROCESS_DF_1

#     # We can creat a pipeline even though the file doesn't exist!
#     pipe = Pipeline()

#     # Normally we don't need to do this, but just to be explicit:
#     pd.testing.assert_frame_equal(pipe.data, TO_PREPROCESS_DF_1)

#     # Check that the slow computation method is called with the preprocessed data
#     pipe.run()

#     slow_computation_called = slow_computation_method.call_args[0][1]
#     pd.testing.assert_frame_equal(slow_computation_called, PREPROCESSED_DF_1)
