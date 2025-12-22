import numpy as np
from flwr.common import NDArrays
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from datasets import load_dataset
import pandas as pd

# server side evaluation packages
from flwr.app import ArrayRecord, MetricRecord
from sklearn.metrics import log_loss

# tunning model
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

# binary classification metrics
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# save results
from pathlib import Path

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


def create_log_reg_and_instantiate_parameters(
    penalty,
    n_features: int = None,
    n_classes: int = None,
    max_iter_per_round: int = 1,
    tol: float = 1e-3,
    C: float = 1.0,
    suppress_convergence_warning: bool = True,
):
    """
    Create a LogisticRegression and set initial parameters
    You can keep small local epochs while reducing warning noise
    """

    if suppress_convergence_warning:
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
            module=r"sklearn\.linear_model\._sag"
        )

    model = LogisticRegression(
        #penalty=penalty,
        l1_ratio=0 if penalty == "l2" else None,
        C=C,
        max_iter=max_iter_per_round,
        tol=tol,
        warm_start=True,
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

def load_data_ai4i(partition_id: int, num_partitions: int, data_path="ai4i2020.csv", split=0.8) -> pd.DataFrame:
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

    scaler = StandardScaler()

    if split == 1.0:
        # Use all data for training/testing
        X_scaled = scaler.fit_transform(X.values)

        return X_scaled, y.values
    else:
        # Split the on-edge data: 80% train, 20% test (stratified)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-split), stratify=y, random_state=42)
        X_train, X_test = X[: int(split * len(X))], X[int(split * len(X)) :] 
        y_train, y_test = y[: int(split * len(y))], y[int(split * len(y)) :]
        # Standardize features using training statistics
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)

        return X_train_scaled, y_train.values, X_test_scaled, y_test.values

def binary_classification_metrics(
    y_true,
    y_proba,
    classes,
    prefix="",
    threshold=0.5,
):
    """
    Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : ndarray of shape (n_samples, n_classes)
        Predicted class probabilities.
    classes : ndarray
        Model classes (model.classes_).
    prefix : str, optional
        Prefix for metric names (e.g. "train_", "test_").
    threshold : float, optional
        Decision threshold for positive class.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """

    # Select positive class
    pos_label = 1 if 1 in classes else classes[-1]
    pos_idx = int(np.where(classes == pos_label)[0][0])

    # Positive-class probabilities and predictions
    y_pos = y_proba[:, pos_idx]
    y_pred = (y_pos >= threshold).astype(int)

    # Core metrics
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}logloss": log_loss(y_true, y_proba, labels=classes),
    }

    # Binary metrics
    metrics.update({
        f"{prefix}precision_bin": precision_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
        f"{prefix}recall_bin": recall_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
        f"{prefix}f1_bin": f1_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
    })

    # Macro metrics
    metrics.update({
        f"{prefix}precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{prefix}recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{prefix}f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    })

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=classes).ravel()

    metrics.update({
        f"{prefix}tn": int(tn),
        f"{prefix}fp": int(fp),
        f"{prefix}fn": int(fn),
        f"{prefix}tp": int(tp),
        f"{prefix}sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # TPR
        f"{prefix}specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # TNR
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    })

    # Probabilistic metrics
    try:
        metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_pos)
    except ValueError:
        metrics[f"{prefix}roc_auc"] = float("nan")

    try:
        metrics[f"{prefix}pr_auc"] = average_precision_score(y_true, y_pos)
    except ValueError:
        metrics[f"{prefix}pr_auc"] = float("nan")

    metrics["num-examples"] = len(y_true)

    return metrics

# server side evaluation function
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    penalty = "l2"
    model = create_log_reg_and_instantiate_parameters(penalty)

    ndarrays = arrays.to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load full evaluation data
    X_total, y_total = load_data_ai4i(
        partition_id=0,
        num_partitions=1,
        split=1.0
    )

    # ONLY evaluate
    y_total_proba = model.predict_proba(X_total)

    metrics = binary_classification_metrics(
        y_true=y_total,
        y_proba=y_total_proba,
        classes=model.classes_,
        prefix="global_",
    )

    return MetricRecord(metrics)