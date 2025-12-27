"""flowersimulation: A Flower / sklearn app.""" 

import joblib
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad, FedAdam, FedYogi

from pathlib import Path
from flowersimulation.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    set_model_params,
    global_evaluate,
    set_all_seeds
)

# Save results
import pandas as pd
from pathlib import Path

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # 1. Set global seed at the start
    seed = context.run_config["seed"]
    set_all_seeds(seed)

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    ## Factors
    fl_algo: float = context.run_config["fl-algorithm"]
    dataset_name: str = context.run_config["dataset-name"]
    fraction: float = context.run_config["fraction"]
    failure_rate: float = context.run_config["failure-rate"]
    iid: str = context.run_config["iid"]
    
    # Build once per run
    balance_type = dataset_name.split("_")[1].replace(".csv", "")
    factors = {
        "fl_algo": fl_algo,
        "dataset_balance_type": balance_type,
        "fraction": fraction,
        "failure_rate": failure_rate,
        "iid": iid,
        "seed": seed
    }

    print(f"FL Algorithm: {fl_algo}, Dataset: {dataset_name}, Fraction: {fraction}, "
          f"Failure Rate: {failure_rate}, IID: {iid}, Seed: {seed}")
    
    # Create LogisticRegression Model. If AI4I data present, use its dims. 
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = create_log_reg_and_instantiate_parameters(penalty, max_iter_per_round=local_epochs, seed=seed)
    initial_parameters = get_model_params(model)
    # Construct ArrayRecord representation
    arrays = ArrayRecord(initial_parameters)

    # Initialize strategy
    fraction_train = fraction
    fraction_evaluate = fraction
    if fl_algo == "FedAvg":
        ## FedAvg
        strategy = FedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    elif fl_algo == "FedAdagrad":
        # FedAdagrad
        strategy = FedAdagrad(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    elif fl_algo == "FedAdam":
        # FedAdam
        strategy = FedAdam(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    elif fl_algo == "FedYogi":
        # FedYogi
        strategy = FedYogi(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    else:
        raise ValueError(f"Algorithm {fl_algo} not supported.")

    # 2) Create a wrapper function to pass dataset_name to global_evaluate
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        return global_evaluate(server_round, arrays, dataset_name, seed=seed)

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

    # -------------------------
    # GLOBAL (server-side eval)
    # -------------------------
    global_metrics = result.evaluate_metrics_serverapp
    rows = []

    if not global_metrics:
        print("WARNING: No server-side evaluation metrics were collected.")
    else:
        for rnd, metric_record in global_metrics.items():
            # base columns
            row = {"round": rnd}
            row.update(dict(metric_record))    # metrics
            row.update(factors)                # add factors
            rows.append(row)

    df_global = pd.DataFrame(rows)
    df_global.to_csv(results_dir / f"global_metrics_{fl_algo}_{balance_type}_{int(fraction*100)}_{int(failure_rate*100)}_{iid}_{seed}.csv", index=False)

    # -------------------------
    # TRAIN (client-side train)
    # -------------------------
    train_metrics = result.train_metrics_clientapp
    rows = []

    if not train_metrics:
        print("WARNING: No client-side training metrics were collected.")
    else:
        for rnd, metric_record in train_metrics.items():
            row = {"round": rnd}
            row.update(dict(metric_record))
            row.update(factors)
            rows.append(row)

    df_train = pd.DataFrame(rows)
    df_train.to_csv(results_dir / f"train_metrics_{fl_algo}_{balance_type}_{int(fraction*100)}_{int(failure_rate*100)}_{iid}_{seed}.csv", index=False)

    # -------------------------
    # EVALUATE (client-side eval)
    # -------------------------
    evaluate_metrics = result.evaluate_metrics_clientapp
    rows = []

    if not evaluate_metrics:
        print("WARNING: No client-side evaluation metrics were collected.")
    else:
        for rnd, metric_record in evaluate_metrics.items():
            row = {"round": rnd}
            row.update(dict(metric_record))
            row.update(factors)
            rows.append(row)

    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(results_dir / f"eval_metrics_{fl_algo}_{balance_type}_{int(fraction*100)}_{int(failure_rate*100)}_{iid}_{seed}.csv", index=False)
