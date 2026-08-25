# src/plot_training_vram.py
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    log_file = "results/training_vram_comparison.json"
    if not os.path.exists(log_file):
        print(f"❌ Arquivo {log_file} não encontrado!")
        return

    with open(log_file, "r") as f:
        data = json.load(f)

    methods = list(data.keys())
    vram_vals = [data[m]["peak_vram_gb"] for m in methods]
    time_vals = [data[m]["time_seconds"] for m in methods]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#c94c4c", "#2b90d9"]
    bars = ax.bar(methods, vram_vals, color=colors, width=0.45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f} GB", 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    diff = ((vram_vals[0] - vram_vals[1]) / vram_vals[0]) * 100
    plt.title(f"Pico de Memória VRAM Durante o Fine-Tuning (Economia de {diff:.1f}%)", fontsize=12, pad=15)
    plt.ylabel("Pico de VRAM Alocada (GB)", fontsize=11)
    plt.ylim(0, max(vram_vals) * 1.3)
    plt.tight_layout()

    out_plot = "results/training_vram_reduction.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print(f"📊 Gráfico de VRAM de Treinamento salvo em {out_plot}")

if __name__ == "__main__":
    main()