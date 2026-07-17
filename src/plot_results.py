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

    # Chave calibrada para o dataset completo
    c2_key = "25000_samples_5_epochs"
    
    # Mapeamento estrito dos 4 métodos que queremos comparar no cenário realista de Edge
    metodos_alvo = ["prompt_tuning_random", "prompt_tuning_text", "lora", "full_ft"]
    
    for m in metodos_alvo:
        if c2_key not in metrics_data or m not in metrics_data[c2_key]:
            print(f"❌ Faltam os dados de '{m}' no JSON para gerar o gráfico comparativo.")
            print(f"Dica: Certifique-se de que rodou o evaluate.py após os treinos das 4 abordagens.")
            return

    # Labels estéticas e paleta de cores equilibrada para o quarteto
    labels = [
        "Prompt Tuning\n(Random - 100t)", 
        "Prompt Tuning\n(Texto - 20t)", 
        "LoRA\n(PEFT)", 
        "Full Fine-Tuning"
    ]
    colors = ["#55a868", "#4c72b0", "#c44e52", "#dd8452"]
    x = np.arange(len(labels))

    # Coleta de métricas reais de hardware salvos pelo callback do train.py
    vram_valores = [metrics_data[c2_key][m]["vram"] for m in metodos_alvo]
    tempo_valores = [metrics_data[c2_key][m]["time"] for m in metodos_alvo]
    
    # Coleta dinâmica das acurácias calculadas no evaluate.py
    acuracia_valores = [
        metrics_data[c2_key]["prompt_tuning_random"].get("accuracy", 0.0),
        metrics_data[c2_key]["prompt_tuning_text"].get("accuracy", 0.0),
        metrics_data[c2_key]["lora"].get("accuracy", 0.0),
        metrics_data[c2_key]["full_ft"].get("accuracy", 0.0)
    ]

    # --- FIGURA 1: COMPARATIVO DE ACURÁCIA (4 BARRAS) ---
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, acuracia_valores, color=colors, width=0.45)
    plt.title('Acurácia Comparativa por Método de Ajuste (5 Épocas / Full Dataset 25k)')
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, max(acuracia_valores) + 12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', ha='center', weight='bold')
    plt.tight_layout()
    
    path_acc = "results/comparativo_acuracia_grok.png"
    plt.savefig(path_acc, dpi=300)
    print(f"📊 Gráfico quadri-barra de Acurácia salvo em: {path_acc}")
    plt.close()

    # --- FIGURA 2: RECURSOS DE HARDWARE (4 BARRAS) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Subplot VRAM
    bars_vram = ax1.bar(labels, vram_valores, color=colors, width=0.45)
    ax1.set_title('Pico Real de Alocação de VRAM Coletado')
    ax1.set_ylabel('Memória (GB)')
    ax1.set_ylim(0, max(vram_valores) + 3)
    for bar in bars_vram:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval:.2f}G', ha='center', weight='bold')

    # Subplot Tempo
    bars_tempo = ax2.bar(labels, tempo_valores, color=colors, width=0.45)
    ax2.set_title('Tempo Real de Treinamento Coletado')
    ax2.set_ylabel('Segundos (s)')
    ax2.set_ylim(0, max(tempo_valores) * 1.15)
    for bar in bars_tempo:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(tempo_valores)*0.015), f'{int(yval)}s', ha='center', weight='bold')

    plt.tight_layout()
    
    path_hw = "results/comparativo_hardware_grok.png"
    plt.savefig(path_hw, dpi=300)
    print(f"📉 Gráfico quadri-barra de Hardware salvo em: {path_hw}")
    plt.close()

if __name__ == "__main__":
    main()