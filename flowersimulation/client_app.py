"""flowersimulation: A Flower / sklearn app."""

import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss

from flowersimulation.task import (
    UNIQUE_LABELS_AI4I,
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    load_data_ai4i,
    set_model_params,
    binary_classification_metrics
)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # 1) Build model using current run_config
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)

    # 2) Apply global parameters received from the server
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # 3) Load local partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, y_train, _, _ = load_data_ai4i(partition_id, num_partitions)

    # 4) Fit locally
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # 5) Probabilities and thresholded predictions
    y_proba = model.predict_proba(X_train)  # shape (n_samples, n_classes)

    # 6) Binary classification metrics
    metrics = binary_classification_metrics(
        y_true=y_train,
        y_proba=y_proba,
        classes=model.classes_,
        prefix="train_",
    )

    # 6) Other metrics
    #metrics["num-examples"] = len(X_train)

    # 7) Return params and metrics
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    # Create LogisticRegression Model
    model = create_log_reg_and_instantiate_parameters(penalty)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    #_, _, X_test, y_test = load_data(partition_id, num_partitions)
    _, _, X_test, y_test = load_data_ai4i(partition_id, num_partitions)

    # Evaluate the model on local data
    y_test_proba = model.predict_proba(X_test)

    metrics = binary_classification_metrics(
        y_true=y_test,
        y_proba=y_test_proba,
        classes=model.classes_,
        prefix="test_",
    )

    #metrics["num-examples"] = len(X_test)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
