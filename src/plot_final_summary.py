# src/plot_final_summary.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    sns.set_theme(style="whitegrid")
    
    methods = ["BF16 Original", "4-Bit Real (NF4)", "1-Bit (Bit-Level Sim)"]
    accuracies = [98.32, 96.86, 99.70]
    vram_usage = [2.29, 0.95, 0.26]  # 0.26 GB teórico com packing
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Eixo 1: Acurácia
    color_acc = "#2b90d9"
    bars_acc = ax1.bar(x - width/2, accuracies, width, label="Acurácia (%)", color=color_acc)
    ax1.set_ylabel("Acurácia (%)", color=color_acc, fontsize=11, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 115)
    
    for bar in bars_acc:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", 
                 ha='center', va='bottom', fontweight='bold', color=color_acc, fontsize=10)
        
    # Eixo 2: Consumo de VRAM
    ax2 = ax1.twinx()
    color_vram = "#c94c4c"
    bars_vram = ax2.bar(x + width/2, vram_usage, width, label="Pico de VRAM (GB)", color=color_vram)
    ax2.set_ylabel("Pico de VRAM (GB)", color=color_vram, fontsize=11, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=color_vram)
    ax2.set_ylim(0, 3.0)
    
    for bar in bars_vram:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} GB", 
                 ha='center', va='bottom', fontweight='bold', color=color_vram, fontsize=10)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11, fontweight="bold")
    
    plt.title("Compensação entre Acurácia e Consumo de VRAM (LoRA + Técnicas de Quantização)", fontsize=12, pad=15)
    fig.tight_layout()
    plt.savefig("results/final_tradeoff_comparison.png", dpi=300)
    plt.close()
    
    print("📊 Gráfico comparativo gerado em: results/final_tradeoff_comparison.png")

if __name__ == "__main__":
    main()