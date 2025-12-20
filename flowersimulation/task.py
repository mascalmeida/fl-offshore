import numpy as np
from flwr.common import NDArrays
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

from datasets import load_dataset
from pathlib import Path
from typing import Union
import pandas as pd

# This information is needed to create a correct scikit-learn model
UNIQUE_LABELS_AI4I = [0, 1]
FEATURES_AI4I = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]


def get_model_params(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklean LogisticRegression model."""
    # Validate params length to avoid IndexError and give clearer error
    expected = 2 if model.fit_intercept else 1
    if params is None or len(params) < expected:
        raise ValueError(
            f"Insufficient model parameters: expected {expected} ndarray(s), got {0 if params is None else len(params)}."
        )

    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    # scikit-learn represents binary logistic regression coef_ as shape
    # (1, n_features) while multiclass uses (n_classes, n_features).
    if n_classes <= 2:
        coef_shape = (1, n_features)
        intercept_shape = (1,)
    else:
        coef_shape = (n_classes, n_features)
        intercept_shape = (n_classes,)

    model.coef_ = np.zeros(coef_shape)
    if model.fit_intercept:
        model.intercept_ = np.zeros(intercept_shape)


def create_log_reg_and_instantiate_parameters(penalty, n_features: int = None, n_classes: int = None):
    """Create a LogisticRegression and set initial parameters.

    If `n_features` or `n_classes` are not provided, fall back to iris defaults.
    """
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
        solver="saga",
    )

    if n_features is None:
        n_features = len(FEATURES_AI4I)
    if n_classes is None:
        n_classes = len(UNIQUE_LABELS_AI4I)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=n_features, n_classes=n_classes)
    return model

# Function to load local CSV, partition it and return a pandas DataFrame for the partition

fds_ai4i = None  # Cache FederatedDataset

def load_data_ai4i(partition_id: int, num_partitions: int, data_path="ai4i2020.csv") -> pd.DataFrame:
    """Load a local CSV, partition it using IidPartitioner and return the partition as a pandas DataFrame.

    Args:
        data_path: Path to the CSV file (or list of paths) compatible with `datasets.load_dataset`.
        partition_id: Which partition to return (0-based).
        num_partitions: Number of partitions to create.

    Returns:
        A pandas DataFrame containing the rows for the requested partition.
    """
    global fds_ai4i
    if fds_ai4i is None:
        # Simple loading: use the provided data_path as-is.
        df = load_dataset("csv", data_files=str(data_path))
        # load_dataset often returns a DatasetDict; pick the 'train' split if present
        if hasattr(df, "keys"):
            if "train" in df:
                df = df["train"]
            else:
                df = next(iter(df.values()))

        fds_ai4i = IidPartitioner(num_partitions=num_partitions)
        fds_ai4i.dataset = df

    dataset = fds_ai4i.load_partition(partition_id=partition_id).with_format("pandas")[:]

    X = dataset[FEATURES_AI4I]
    y = dataset["Machine failure"]
    # Split the on-edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    return X_train.values, y_train.values, X_test.values, y_test.values