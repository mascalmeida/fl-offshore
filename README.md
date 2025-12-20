---
tags: [tabular, fds]
dataset: [ai4i2020]
framework: [scikit-learn]
---

# Federated Learning with scikit-learn and Flower

This repository contains code to run federated training of a scikit-learn `LogisticRegression` model with Flower on the AI4I dataset. The code includes dataset partitioning, server and client apps, and utilities for model parameters and evaluation.

AI4I, also called ai4i2020, has sensor readings and labels for machine failure. Typical features are air temperature, process temperature, rotational speed, torque and tool wear. The data is often used for predictive maintenance research for rotating machinery. These problems are similar to failure prediction tasks on offshore systems such as FPSO vessels. The examples here help test federated training and evaluation in edge settings for predictive maintenance.

## Quick setup

Requirements
- Python 3.10 or newer
- Dependencies listed in `pyproject.toml`

Install

```bash
pip install -e .
```

Or install core packages

```bash
pip install "flwr[simulation]>=1.24.0" "flwr-datasets[vision]>=0.5.0" scikit-learn>=1.6.1
```

Run the simulation

```bash
flwr run .
```

To change runtime options pass `--run-config` flags to `flwr run`.

The server saves the final model to `logreg_model.pkl`.

## Project layout
- `pyproject.toml` - project metadata and run config
- `ai4i2020.csv` - dataset file expected in the project root
- `flowersimulation/task.py` - model helpers and data partitioning
- `flowersimulation/server_app.py` - server app and FedAvg orchestration
- `flowersimulation/client_app.py` - client app with train and evaluate handlers

## Notes on data and experiments
- `load_data_ai4i` uses `flwr_datasets.partitioner.IidPartitioner` to create IID partitions from `ai4i2020.csv` and returns train and test splits for a partition.
- The code sets initial model parameters so clients can return arrays even before local fitting.

## Development
- Use Flower simulation to run multi-node experiments locally and tune partitions and run options to reproduce experiments.
- Add unit tests for `get_model_params`, `set_model_params`, and `load_data_ai4i` to improve reproducibility.

## License
- Apache-2.0, see `pyproject.toml` for details