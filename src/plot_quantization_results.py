# src/plot_quantization_results.py
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    log_file = "results/quantization_metrics.json"
    if not os.path.exists(log_file):
        print(f"❌ Arquivo {log_file} não encontrado! Execute evaluate_quantized.py primeiro.")
        return

    with open(log_file, "r") as f:
        data = json.load(f)

    methods = list(data.keys())
    accuracies = [data[m]["acc"] for m in methods]
    vram_usage = [data[m]["vram"] for m in methods]
    execution_times = [data[m]["time"] for m in methods]

    sns.set_theme(style="whitegrid")

    # --- Gráfico 1: Acurácia Comparativa pós-Quantização ---
    plt.figure(figsize=(9, 6))
    colors = ["#c94c4c", "#2b90d9", "#42b883"]
    bars = plt.bar(methods, accuracies, color=colors, width=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.title("Acurácia Comparativa com Compressão Extrema (LoRA + Binarização 1-Bit / KV 4-Bit)", fontsize=12, pad=15)
    plt.ylabel("Acurácia (%)", fontsize=11)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig("results/quantization_accuracy.png", dpi=300)
    plt.close()

    # --- Gráfico 2: VRAM e Tempo de Execução ---
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = '#2b90d9'
    ax1.set_xlabel('Método de Compressão', fontsize=11)
    ax1.set_ylabel('Pico de VRAM (GB)', color=color, fontsize=11)
    bars_vram = ax1.bar(methods, vram_usage, color=color, width=0.35, align='center', label='VRAM (GB)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(vram_usage) * 1.3)

    for bar in bars_vram:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} GB", 
                 ha='center', va='bottom', fontweight='bold', color=color)

    plt.title("Consumo de VRAM durante a Inferência Comprimida", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("results/quantization_hardware.png", dpi=300)
    plt.close()

    print("📊 Gráficos de quantização gerados em results/quantization_accuracy.png e results/quantization_hardware.png!")

if __name__ == "__main__":
    main()