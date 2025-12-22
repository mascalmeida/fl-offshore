"""flowersimulation: A Flower / sklearn app.""" 

import joblib
from flwr.app import ArrayRecord, Context
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

    # Create LogisticRegression Model. If AI4I data present, use its dims. 
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # Initialize strategy
    fraction_train = 1.0
    fraction_evaluate = 1.0
    ## FedAvg
    #strategy = FedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
    # FedAdagrad
    strategy = FedAdagrad(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate
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
