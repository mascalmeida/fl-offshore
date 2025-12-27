"""flowersimulation: A Flower / sklearn app."""

import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss
import numpy as np

from flowersimulation.task import (
    UNIQUE_LABELS_AI4I,
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    load_data_ai4i,
    set_model_params,
    binary_classification_metrics,
    set_all_seeds
)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    run_seed = context.run_config["seed"]
    partition_id = context.node_config["partition-id"]
    local_seed = run_seed + partition_id 
    
    # 2. Aplicar a seed antes de qualquer operação aleatória
    set_all_seeds(local_seed)

    failure_rate = context.run_config["failure-rate"]

    # prob failure rate definition (failure_rate para True, 1 - failure_rate para False)
    failure = np.random.choice([True, False], p=[failure_rate, 1-failure_rate])
    if failure:
        warnings.warn("Simulated failure on this client during training.")
        raise RuntimeError("Simulated client failure during training.")

    # 1) Build model using current run_config
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = create_log_reg_and_instantiate_parameters(penalty, max_iter_per_round=local_epochs, seed=local_seed)

    # 2) Apply global parameters received from the server
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # 3) Load local partition
    dataset_name = context.run_config["dataset-name"]
    num_partitions = context.node_config["num-partitions"]
    iid = context.run_config["iid"]
    X_train, y_train, _, _ = load_data_ai4i(partition_id, num_partitions, data_path=dataset_name, iid=iid, seed=local_seed)

    # 4) Fit locally
    # --- START OF FIX ---
    # 4) Train ONLY if we have at least 2 classes (0 and 1)
    # If a client has only "Normal" data, it cannot learn what "Failure" looks like.
    if len(np.unique(y_train)) < 2:
        # Skip fit(). The model keeps the global parameters we set in step 2.
        # We manually set classes_ so any subsequent metric calculation doesn't crash.
        model.classes_ = np.array([0, 1]) 
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        model.fit(X_train, y_train)
    # --- END OF FIX ---

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

    run_seed = context.run_config["seed"]
    partition_id = context.node_config["partition-id"]
    local_seed = run_seed + partition_id 
    
    # 2. Aplicar a seed antes de qualquer operação aleatória
    set_all_seeds(local_seed)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = create_log_reg_and_instantiate_parameters(penalty, max_iter_per_round=local_epochs, seed=local_seed)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    dataset_name = context.run_config["dataset-name"]
    num_partitions = context.node_config["num-partitions"]
    iid = context.run_config["iid"]
    _, _, X_test, y_test = load_data_ai4i(partition_id, num_partitions, data_path=dataset_name, iid=iid, seed=local_seed)

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
