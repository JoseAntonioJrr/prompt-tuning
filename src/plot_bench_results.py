# src/plot_bench_results.py
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    sns.set_theme(style="whitegrid")
    
    models = ["FP16", "Q4_K_M (4-bit)", "Q2_K (2-bit)"]
    sizes_mb = [2099.2, 636.18, 411.41]       # 2.05 GiB em MiB
    speed_tg = [8.08, 20.58, 50.75]           # Geração de texto (tokens/s)
    speed_pp = [433.44, 212.75, 381.75]       # Processamento de prompt (tokens/s)
    
    # --- Gráfico 1: Throughput de Geração vs. Tamanho do Modelo ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color_speed = "#2b90d9"
    bars = ax1.bar(models, speed_tg, color=color_speed, width=0.4, label="Tokens/s (Geração)")
    ax1.set_ylabel("Throughput de Geração (tokens/s)", color=color_speed, fontsize=11, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color_speed)
    ax1.set_ylim(0, 65)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f} t/s\n({yval/8.08:.1f}x)", 
                 ha='center', va='bottom', fontweight='bold', color=color_speed)
                 
    plt.title("Impacto da Quantização Física na Velocidade de Inferência (tg128)", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig("results/gguf_inference_speed.png", dpi=300)
    plt.close()
    
    # --- Gráfico 2: Comparativo de Tamanho Físico em Disco ---
    plt.figure(figsize=(8, 5))
    colors_size = ["#c94c4c", "#e28743", "#42b883"]
    bars_size = plt.bar(models, sizes_mb, color=colors_size, width=0.45)
    
    for bar in bars_size:
        yval = bar.get_height()
        reduction = (1 - (yval / sizes_mb[0])) * 100
        label = f"{yval:.1f} MB" if reduction == 0 else f"{yval:.1f} MB\n(-{reduction:.1f}%)"
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 25, label, 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
                 
    plt.title("Redução Física do Modelo em Disco / Flash Memory", fontsize=12, pad=15)
    plt.ylabel("Tamanho do Modelo (MiB)", fontsize=11)
    plt.ylim(0, 2400)
    plt.tight_layout()
    plt.savefig("results/gguf_model_footprint.png", dpi=300)
    plt.close()
    
    print("📊 Gráficos gerados com sucesso:")
    print("  -> results/gguf_inference_speed.png")
    print("  -> results/gguf_model_footprint.png")

if __name__ == "__main__":
    main()