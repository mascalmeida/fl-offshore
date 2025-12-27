import itertools
import subprocess
import time
import sys

# ==========================================
# 1. Definição dos Fatores e Níveis (2^5)
# ==========================================
# Defina aqui seus 5 fatores e os 2 níveis de cada um
doe_factors = {
    "fl-algorithm": ["FedAvg", "FedAdagrad", "FedAdam", "FedYogi"], # Fator 1: Algoritmo de FL
    #"dataset-name": ["ai4i2020_balanced.csv", "ai4i2020_imbalanced.csv"], # Fator 2: Balanceamento
    #"fraction": [0.39, 1],             # Fator 3: Participação dos clientes
    #"failure-rate": [0, 0.3],         # Fator 4: Robustez (0% ou 30% falha)
    #"iid": [1, 0],                      # Fator 5: Heterogeneidade
}

# Configurações fixas que não variam no DOE (opcional)
fixed_config = {
    "num-server-rounds": 3,  # Exemplo: fixar em 10 rounds para todos
    "local-epochs": 1,    # Exemplo: fixar em 1 e 5 épocas locais
    "penalty": "l2"
}

# Lista de Seeds para replicar o experimento (ex: 3 execuções por combinação)
# Se quiser apenas uma rodada determinística, deixe apenas [42]
#seeds = [428956419, 1954324947, 1145661099, 1835732737, 794161987, 1329531353, 200496737, 633816299, 1410143363, 1282538739]
seeds = [1954324947]

# ==========================================
# 2. Gerar Combinações
# ==========================================
keys, values = zip(*doe_factors.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

total_runs = len(combinations) * len(seeds)
print(f"Total de combinações (2^5): {len(combinations)}")
print(f"Total de execuções (incluindo seeds): {total_runs}")

# ==========================================
# 3. Loop de Execução
# ==========================================
# Função auxiliar para formatar valores
def format_arg(key, value):
    if isinstance(value, str):
        # Se for string, coloca aspas duplas em volta do valor
        return f'{key}="{value}"'
    else:
        # Se for número ou booleano, deixa sem aspas
        return f"{key}={value}"

current_run = 1

for seed in seeds:
    for config in combinations:
        print(f"\n=============================================")
        print(f"Executando {current_run}/{total_runs} | Seed: {seed}")
        print(f"Config: {config}")
        print(f"=============================================")

        # Monta a string de argumentos para o --run-config
        # Formato: chave=valor chave2=valor2
        config_args = []
        
        # Adiciona fatores do DOE
        for k, v in config.items():
            config_args.append(format_arg(k, v))
        
        # Adiciona configs fixas
        for k, v in fixed_config.items():
            config_args.append(format_arg(k, v))

        # Adiciona a seed atual
        config_args.append(f"seed={seed}")

        # Junta tudo em uma string
        run_config_str = " ".join(config_args)

        # Comando final: flwr run . --run-config "..."
        # Nota: O dataset-name e outros valores string devem ser tratados com cuidado no shell,
        # mas o Flower geralmente lida bem com strings simples.
        cmd = [
            "flwr", "run", ".", 
            "--run-config", run_config_str
        ]

        try:
            # Executa o comando e espera terminar
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro na execução da run {current_run}: {e}")
            # Decida se quer 'break' ou 'continue'
            # continue 
        except KeyboardInterrupt:
            print("Execução interrompida pelo usuário.")
            sys.exit()

        current_run += 1
        
        # Opcional: Pausa curta para limpar recursos do SO se necessário
        time.sleep(2)

print("\nPlanejamento de Experimentos concluído com sucesso!")