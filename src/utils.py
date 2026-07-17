# src/utils.py
import os
from datasets import load_dataset

def setup_environment():
    """Cria a estrutura de pastas do projeto automaticamente."""
    directories = ["data", "results", "outputs", "src"]
    for folder in directories:
        os.makedirs(folder, exist_ok=True)
    print("📁 Estrutura de pastas verificada/criada com sucesso!")

def get_imdb_dataset(tokenizer, max_length=256, num_samples=5000):
    """Baixa o dataset IMDB do Hugging Face e tokeniza para Causal LM."""
    print(f"📥 Carregando {num_samples} exemplos do IMDB Dataset via Hugging Face...")
    
    # Carrega fatias pequenas para treino e validação rápidos usando o caminho canônico
    train_data = load_dataset("stanfordnlp/imdb", split=f"train[:{num_samples}]")
    val_data = load_dataset("stanfordnlp/imdb", split=f"test[:{int(num_samples * 0.2)}]")
    
    def tokenize_fn(examples):
        inputs = []
        # Combina a instrução, o texto da review e a resposta (rótulo verbalizado final)
        for text, label in zip(examples["text"], examples["label"]):
            instruction = "Classifique o sentimento desta revisão de filme como positivo ou negativo."
            label_str = "positive" if label == 1 else "negative"
            inputs.append(f"{instruction}\nReview: {text}\nSentiment: {label_str}")
            
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        # Para modelos autoregressivos, os labels são espelhos dos inputs
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    print("🧹 Tokenizando dados...")
    tokenized_train = train_data.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
    tokenized_val = val_data.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
    
    return tokenized_train, tokenized_val, val_data