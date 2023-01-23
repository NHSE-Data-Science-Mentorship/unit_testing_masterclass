import numpy as np


def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Calculate the mean squared error between an array of predictions and
        ground truth labels.

    Parameters
    ----------
    y_true : np.array
        Ground truth.
    y_pred : np.array
        Labels.

    Returns
    -------
    float
        MSE.
    """
    return np.nanmean((y_true - y_pred) ** 2)


def imputation(timeseries: np.ndarray, seasonality: int) -> np.ndarray:
    """
    Function that imputes missing data in timeseries
    Looks back seasonality points and replaces nan with that value.
    If no seasonality is specified, use the timeseries mean.

    Parameters
    ----------
    timeseries: np.ndarray
        numpy array with timeseries to be imputed
    seasonality: int
        lookback to impute missing values
    Returns
    ------
    new_timeseries: np.ndarray
        numpy array of same length as timeseries with missing
        values imputed
    """

    # Initialise a list, with the earyl values we won't impute.
    if seasonality is None:
        new_timeseries = []
        padding = 0
    else:
        if seasonality < 0:
            raise ValueError("seasonality should be positive!")

        new_timeseries = timeseries[:seasonality].tolist()
        padding = len(new_timeseries)

    # Loop through the timseries, performing the imputation as needed
    for item_idx, item in enumerate(timeseries[seasonality:]):
        if np.isnan(item):
            if seasonality is None:
                value = np.nanmean(timeseries)
            else:
                value = new_timeseries[item_idx + padding - seasonality]
        else:
            value = item
        new_timeseries.append(value)
    return np.array(new_timeseries)
