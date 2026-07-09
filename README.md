# Benchmark de Fine-Tuning Eficiente: Prompt Tuning vs Full Fine-Tuning com SLM

Este repositório apresenta um estudo comparativo e pedagógico entre as técnicas de **Full Fine-Tuning** e **Prompt Tuning (PEFT)** utilizando o modelo de linguagem de pequena escala (SLM) **TinyLlama-1.1B** aplicado à tarefa de classificação de sentimentos no dataset IMDB.

O projeto foi validado e otimizado utilizando uma GPU de alta performance **NVIDIA A100 (80GB VRAM)**, demonstrando os impactos práticos de hiperparâmetros como tamanho do lote (*Batch Size*) e métodos de inicialização de *soft prompts* na convergência e eficiência de hardware.

---

## 🛠️ Como Executar o Projeto

### 1. Pré-requisitos e Instalação

O projeto utiliza o gerenciador de pacotes de alta performance **uv**. Certifique-se de tê-lo instalado no ambiente Linux.

```bash

# Sincronize e instale as dependências usando o uv
uv sync

### 2. Executando os Treinamentos

Os scripts aceitam argumentos dinâmicos para controlar o método de treino, amostragem e parâmetros de inicialização.
Bash

# Treinamento 1: Prompt Tuning com inicialização por Texto
python src/train.py --method prompt_tuning --samples 5000 --epochs 5 --init text

# Treinamento 2: Prompt Tuning com inicialização Aleatória (Melhor Performance)
python src/train.py --method prompt_tuning --samples 5000 --epochs 5 --init random

# Treinamento 3: Full Fine-Tuning de todas as camadas
python src/train.py --method full_ft --samples 5000 --epochs 5

3. Avaliação de Métricas (Logits)

Para avaliar os checkpoints gerados de forma analítica e extrair os scores de acurácia baseados na probabilidade dos tokens de saída ("positive" vs "negative"):
Bash

python src/evaluate.py

4. Geração Automática de Gráficos Comparativos

O ecossistema conta com um script automatizado que lê os registros de hardware salvos em tempo real (results/metrics_log.json) e gera os gráficos de barras agrupadas:
Bash

python src/plot_results.py

Os gráficos serão exportados diretamente para o diretório results/ com o sufixo _ajustesA100.png.
📂 Estrutura do Repositório
Plaintext

├── src/
│   ├── train.py          # Script de treinamento com Callbacks de monitoramento de VRAM
│   ├── evaluate.py       # Pipeline de avaliação analítica via Logits brutos
│   └── plot_results.py   # Gerador dinâmico de gráficos comparativos (Matplotlib/Seaborn)
├── results/              # JSON de logs e saídas gráficas geradas
├── pyproject.toml        # Configuração de dependências do projeto via uv
└── README.md             # Documentação técnica do ecossistema