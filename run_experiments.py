import itertools
import subprocess
import time
import sys

# ==========================================
# 1. Definition of Factors and Levels (2^5)
# ==========================================
# Define your 5 factors and their 2 levels here
doe_factors = {
    "fl-algorithm": ["FedAvg", "FedAdagrad", "FedAdam", "FedYogi"], # Factor 1: FL Algorithm
    "dataset-name": ["ai4i2020_balanced.csv", "ai4i2020_imbalanced.csv"], # Factor 2: Balancing
    "fraction": [0.39, 1],             # Factor 3: Client participation
    "failure-rate": [0, 0.3],          # Factor 4: Robustness (0% or 30% failure)
    "iid": [1, 0],                     # Factor 5: Heterogeneity
}

# Fixed configurations that do not vary in the DOE (optional)
fixed_config = {
    "num-server-rounds": 3,  # Example: fix at 10 rounds for all
    "local-epochs": 1,    # Example: fix at 1 and 5 local epochs
    "penalty": "l2"
}

# List of Seeds to replicate the experiment (e.g., 3 runs per combination)
# If you want only one deterministic run, leave only [42]
#seeds = [428956419, 1954324947, 1145661099, 1835732737, 794161987, 1329531353, 200496737, 633816299, 1410143363, 1282538739]
seeds = [1954324947]

# ==========================================
# 2. Generate Combinations
# ==========================================
keys, values = zip(*doe_factors.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

total_runs = len(combinations) * len(seeds)
print(f"Total combinations (2^5): {len(combinations)}")
print(f"Total runs (including seeds): {total_runs}")

# ==========================================
# 3. Execution Loop
# ==========================================
# Helper function to format values
def format_arg(key, value):
    if isinstance(value, str):
        # If it is a string, put double quotes around the value
        return f'{key}="{value}"'
    else:
        # If it is a number or boolean, leave without quotes
        return f"{key}={value}"

current_run = 1

for seed in seeds:
    for config in combinations:
        print(f"\n=============================================")
        print(f"Running {current_run}/{total_runs} | Seed: {seed}")
        print(f"Config: {config}")
        print(f"=============================================")

        # Assembles the argument string for --run-config
        # Format: key=value key2=value2
        config_args = []
        
        # Adds DOE factors
        for k, v in config.items():
            config_args.append(format_arg(k, v))
        
        # Adds fixed configs
        for k, v in fixed_config.items():
            config_args.append(format_arg(k, v))

        # Adds the current seed
        config_args.append(f"seed={seed}")

        # Joins everything into a string
        run_config_str = " ".join(config_args)

        # Final command: flwr run . --run-config "..."
        # Note: dataset-name and other string values must be handled carefully in the shell,
        # but Flower usually handles simple strings well.
        cmd = [
            "flwr", "run", ".", 
            "--run-config", run_config_str
        ]

        try:
            # Executes the command and waits for it to finish
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing run {current_run}: {e}")
            # Decide if you want 'break' or 'continue'
            # continue 
        except KeyboardInterrupt:
            print("Execution interrupted by user.")
            sys.exit()

        current_run += 1
        
        # Optional: Short pause to clear OS resources if necessary
        time.sleep(2)

print("\nDesign of Experiments successfully completed!")