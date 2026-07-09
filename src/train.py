# src/train.py
import argparse
import torch
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from utils import setup_environment, get_imdb_dataset

class HardwareMonitorCallback(TrainerCallback):
    def __init__(self, method_name, num_samples, num_epochs):
        self.method_name = method_name
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        print(f"⏱️  [Monitor] Iniciando contagem de tempo e VRAM para {self.method_name}...")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        print(f"🏁 [Monitor] Treino Concluído! Tempo Real: {total_time:.2f}s | Pico de VRAM: {peak_vram:.2f} GB")
        
        log_file = "results/metrics_log.json"
        data = {}
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try: data = json.load(f)
                except: data = {}
                
        key = f"{self.num_samples}_samples_{self.num_epochs}_epochs"
        if key not in data: data[key] = {}
        if self.method_name not in data[key]: data[key][self.method_name] = {}
        
        data[key][self.method_name]["time"] = total_time
        data[key][self.method_name]["vram"] = peak_vram
        
        with open(log_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Métricas reais salvas em {log_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de SLM com Prompt Tuning ou Full Fine-Tuning")
    parser.add_argument("--method", type=str, required=True, choices=["prompt_tuning", "full_ft"])
    parser.add_argument("--samples", type=int, default=5000, help="Quantidade de amostras")
    parser.add_argument("--epochs", type=int, default=5, help="Quantidade de épocas")
    parser.add_argument("--init", type=str, default="text", choices=["text", "random"], help="Inicialização do Prompt Tuning")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🤖 Carregando o modelo base: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    if args.method == "prompt_tuning":
        if args.init == "text":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=20,
                prompt_tuning_init_text="Classifique o sentimento desta revisão de filme como positivo ou negativo:",
                tokenizer_name_or_path=model_id,
            )
            output_dir = "outputs/prompt_tuning_checkpoint"
            method_name = "prompt_tuning_text"
            learning_rate = 3e-2
        else:
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.RANDOM,
                num_virtual_tokens=100,  
                tokenizer_name_or_path=model_id,
            )
            output_dir = "outputs/prompt_tuning_random_checkpoint"
            method_name = "prompt_tuning_random"
            learning_rate = 5e-3  # Taxa ideal e estável para atualizar apenas os soft-tokens
            
        model = get_peft_model(model, peft_config)
    else:
        # CORREÇÃO DIDÁTICA: Reduzido drasticamente para evitar colapso de pesos com Batch 64
        learning_rate = 1e-5  
        output_dir = "outputs/full_ft_checkpoint"
        method_name = "full_ft"

    train_dataset, val_dataset, _ = get_imdb_dataset(tokenizer, num_samples=args.samples)

    training_args = TrainingArguments(
        output_dir=f"results/{method_name}",
        learning_rate=learning_rate,
        per_device_train_batch_size=64,  # Mantido em 64 para extrair a paralelização da A100
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,  
        save_strategy="no",
        bf16=True,
    )

    monitor_callback = HardwareMonitorCallback(method_name, args.samples, args.epochs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[monitor_callback]
    )

    print(f"\n🚀 Iniciando o treinamento usando: {method_name}...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    print(f"✅ Treinamento concluído e pesos salvos em {output_dir}")

if __name__ == "__main__":
    main()