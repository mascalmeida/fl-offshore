"""flowersimulation: A Flower / sklearn app.""" 

import joblib
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad

from pathlib import Path
from flowersimulation.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    set_model_params,
    global_evaluate
)

# Save results
import pandas as pd
from pathlib import Path

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    fl_algo: float = context.run_config["fl-algorithm"]
    fraction: float = context.run_config["fraction"]
    dataset_name: str = context.run_config["dataset-name"]

    # Create LogisticRegression Model. If AI4I data present, use its dims. 
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # Initialize strategy
    fraction_train = fraction
    fraction_evaluate = fraction
    if fl_algo == "FedAvg":
        ## FedAvg
        strategy = FedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    elif fl_algo == "FedAdagrad":
        # FedAdagrad
        strategy = FedAdagrad(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    else:
        ## FedAvg
        strategy = FedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)

    # 2) Create a wrapper function to pass dataset_name to global_evaluate
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        return global_evaluate(server_round, arrays, dataset_name)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn
    )

    # Save final model parameters
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    set_model_params(model, ndarrays)
    joblib.dump(model, "logreg_model.pkl")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # global
    global_metrics = result.evaluate_metrics_serverapp

    rows = []

    if not global_metrics:
        print("WARNING: No client-side training metrics were collected.")
    else:
        for rnd, metric_record in global_metrics.items():
            row = {"round": rnd}
            row.update(dict(metric_record))  # explicit & safe
            rows.append(row)

    df_global = pd.DataFrame(rows)
    df_global.to_csv(results_dir / "global_metrics.csv", index=False)

    # train
    train_metrics = result.train_metrics_clientapp

    rows = []

    if not train_metrics:
        print("WARNING: No client-side training metrics were collected.")
    else:
        for rnd, metric_record in train_metrics.items():
            row = {"round": rnd}
            row.update(dict(metric_record))  # explicit & safe
            rows.append(row)

    df_train = pd.DataFrame(rows)
    df_train.to_csv(results_dir / "train_metrics.csv", index=False)

    # evaluate
    evaluate_metrics = result.evaluate_metrics_clientapp

    rows = []

    if not evaluate_metrics:
        print("WARNING: No client-side evaluation metrics were collected.")
    else:
        for rnd, metric_record in evaluate_metrics.items():
            row = {"round": rnd}
            row.update(dict(metric_record))  # explicit & safe
            rows.append(row)

    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(results_dir / "eval_metrics.csv", index=False)
