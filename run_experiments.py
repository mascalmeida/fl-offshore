import itertools
import subprocess
import time
import sys

# ==========================================
# 1. Definition of Factors and Levels (2*2^5)
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
    "num-server-rounds": 100,  # Example: fix at 10 rounds for all
    "local-epochs": 1,    # Example: fix at 1 and 5 local epochs
    "penalty": "l2"
}

# List of Seeds to replicate the experiment (e.g., 3 runs per combination)
# If you want only one deterministic run, leave only [42]
seeds = [428956419, 1954324947, 1145661099, 1835732737, 794161987, 1329531353, 200496737, 633816299, 1410143363, 1282538739]

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

# --- CONFIGURAÇÃO DE RETOMADA ---
START_FROM_RUN = 319  # Ajustado para o crash
# --------------------------------

print(f"--> Iniciando script.")
print(f"--> Regra 1: Pular execuções anteriores a {START_FROM_RUN}")
print(f"--> Regra 2: Pular TODAS configurações com fraction = 0.39")

for seed in seeds:
    for config in combinations:
        
        # --- LÓGICA DE PULO (SKIP) ---
        
        # 1. Pular rodadas passadas
        if current_run < START_FROM_RUN:
            current_run += 1
            continue 

        # 2. Pular configurações com fraction == 0.39 (Nova regra)
        if config['fraction'] == 0.39:
            # Descomente a linha abaixo se quiser ver no log o que está sendo pulado
            print(f"Skipping run {current_run} (Fraction 0.39 ignored)...") 
            current_run += 1
            continue

        # -----------------------------

        print(f"\n=============================================")
        print(f"Running {current_run}/{total_runs} | Seed: {seed}")
        print(f"Config: {config}")
        print(f"=============================================")

        # Monta a string de argumentos
        config_args = []
        for k, v in config.items():
            config_args.append(format_arg(k, v))
        for k, v in fixed_config.items():
            config_args.append(format_arg(k, v))
        config_args.append(f"seed={seed}")

        run_config_str = " ".join(config_args)

        cmd = [
            "flwr", "run", ".", 
            "--run-config", run_config_str
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro na execução da run {current_run}: {e}")
            # Se quiser que ele continue tentando as próximas mesmo com erro, descomente abaixo:
            # time.sleep(10) 
            # current_run += 1
            # continue
        except KeyboardInterrupt:
            print("Execução interrompida pelo usuário.")
            sys.exit()

        current_run += 1
        
        # Aumentei o tempo de descanso para evitar o erro do Ray no Windows
        print("Resfriando processos (5s)...")
        time.sleep(5) 

print("\nDesign of Experiments successfully completed!")