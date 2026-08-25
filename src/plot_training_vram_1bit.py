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

    # Ordenar por conveniência visual
    methods = ["LoRA Padrão (BF16)", "QLoRA (Base 4-bit NF4)", "QLoRA (Base 1-bit HQQ)"]
    labels = ["LoRA\n(16-bit)", "QLoRA\n(4-bit NF4)", "HQQ-LoRA\n(1-bit)"]
    vram_vals = [data[m]["peak_vram_gb"] for m in methods]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#c94c4c", "#42b883", "#e28743"]
    bars = ax.bar(labels, vram_vals, color=colors, width=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} GB", 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.title("Comparativo de VRAM: O Paradoxo da Ativação no Treinamento", fontsize=13, pad=15)
    plt.ylabel("Pico de VRAM Alocada (GB)", fontsize=11)
    plt.ylim(0, 6.0)
    plt.tight_layout()

    out_plot = "results/training_vram_3_bars.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print(f"📊 Gráfico final salvo em {out_plot}")

if __name__ == "__main__":
    main()