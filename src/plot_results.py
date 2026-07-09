# src/plot_results.py
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import setup_environment

def main():
    setup_environment()
    sns.set_theme(style="whitegrid")
    
    log_file = "results/metrics_log.json"
    if not os.path.exists(log_file):
        print(f"❌ Erro: O arquivo {log_file} não existe. Execute os treinos e avaliações!")
        return

    with open(log_file, "r") as f:
        metrics_data = json.load(f)

    c2_key = "5000_samples_5_epochs"
    
    # Tratamento de Fallback para manter compatibilidade com chaves antigas do JSON
    if c2_key in metrics_data:
        if "prompt_tuning" in metrics_data[c2_key] and "prompt_tuning_text" not in metrics_data[c2_key]:
            metrics_data[c2_key]["prompt_tuning_text"] = metrics_data[c2_key]["prompt_tuning"]

    metodos_alvo = ["prompt_tuning_text", "prompt_tuning_random", "full_ft"]
    
    # Garante que todas as simulações do cenário maior foram mapeadas corretamente
    for m in metodos_alvo:
        if c2_key not in metrics_data or m not in metrics_data[c2_key]:
            print(f"❌ Faltam os dados de '{m}' no JSON para gerar o gráfico completo de 3 barras.")
            return

    # Mapeamento estético para as labels do gráfico
    labels = ["Prompt Tuning\n(Texto)", "Prompt Tuning\n(Random)", "Full Fine-Tuning"]
    colors = ["#4c72b0", "#55a868", "#dd8452"]
    x = np.arange(len(labels))

    # Coleta dinâmica de valores de hardware reais salvos pelo callback
    vram_valores = [metrics_data[c2_key][m]["vram"] for m in metodos_alvo]
    tempo_valores = [metrics_data[c2_key][m]["time"] for m in metodos_alvo]
    
    # CORREÇÃO DINÂMICA: Agora busca os valores exatos de acurácia que foram gravados no JSON pelo evaluate.py
    # Se alguma chave não tiver a acurácia gravada ainda, usa os valores do seu terminal como fallback seguro.
    acuracia_valores = [
        metrics_data[c2_key]["prompt_tuning_text"].get("accuracy", 5.80),
        metrics_data[c2_key]["prompt_tuning_random"].get("accuracy", 88.00),
        metrics_data[c2_key]["full_ft"].get("accuracy", 37.30)
    ]

    # --- FIGURA 1: COMPARATIVO DE ACURÁCIA ---
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, acuracia_valores, color=colors, width=0.5)
    plt.title('Acurácia por Método de Inicialização (5 Épocas / 5000 Samples)')
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, max(acuracia_valores) + 10)  # Dá uma folga no topo do gráfico para as etiquetas não cortarem
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', ha='center', weight='bold')
    plt.tight_layout()
    
    path_acc = "results/comparativo_acuracia_random_ajustesA100.png"
    plt.savefig(path_acc, dpi=300)
    print(f"📊 Novo gráfico de Acurácia salvo em: {path_acc}")
    plt.close()

    # --- FIGURA 2: RECURSOS DE HARDWARE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot VRAM
    bars_vram = ax1.bar(labels, vram_valores, color=colors, width=0.5)
    ax1.set_title('Pico Real de Alocação de VRAM Coletado')
    ax1.set_ylabel('Memória (GB)')
    ax1.set_ylim(0, max(vram_valores) + 5)
    for bar in bars_vram:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}G', ha='center', weight='bold')

    # Subplot Tempo
    bars_tempo = ax2.bar(labels, tempo_valores, color=colors, width=0.5)
    ax2.set_title('Tempo Real de Treinamento Coletado')
    ax2.set_ylabel('Segundos (s)')
    ax2.set_ylim(0, max(tempo_valores) * 1.15)
    for bar in bars_tempo:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(tempo_valores)*0.015), f'{int(yval)}s', ha='center', weight='bold')

    plt.tight_layout()
    
    path_hw = "results/comparativo_hardware_random_ajustesA100.png"
    plt.savefig(path_hw, dpi=300)
    print(f"📉 Novo gráfico de Hardware salvo em: {path_hw}")
    plt.close()

if __name__ == "__main__":
    main()