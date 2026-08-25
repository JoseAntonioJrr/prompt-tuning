# src/train_comparison.py
import torch
import time
import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import setup_environment, get_imdb_dataset

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

def run_training(use_qlora: bool, train_dataset, tokenizer, output_dir: str):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if use_qlora:
        print("\n⚡ [Modo QLoRA] Carregando modelo base quantizado em 4-bit (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("\n📦 [Modo LoRA BF16] Carregando modelo base em precisão normal (BF16)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # 100 steps são suficientes para estabilizar o pico de VRAM
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("🚀 Iniciando treinamento...")
    trainer.train()

    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return peak_vram, total_time

def main():
    setup_environment()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("📥 Carregando dataset...")
    train_dataset, _, _ = get_imdb_dataset(tokenizer, num_samples=25000)

    results = {}

    # --- 1. Treino LoRA Padrão (BF16) ---
    vram_bf16, time_bf16 = run_training(
        use_qlora=False, 
        train_dataset=train_dataset, 
        tokenizer=tokenizer, 
        output_dir="outputs/train_lora_bf16"
    )
    results["LoRA Padrão (BF16)"] = {"peak_vram_gb": vram_bf16, "time_seconds": time_bf16}
    print(f"\n📊 Resultado LoRA BF16 -> Pico VRAM: {vram_bf16:.2f} GB | Tempo: {time_bf16:.1f}s")

    # --- 2. Treino QLoRA (Base Quantizada 4-bit) ---
    vram_qlora, time_qlora = run_training(
        use_qlora=True, 
        train_dataset=train_dataset, 
        tokenizer=tokenizer, 
        output_dir="outputs/train_qlora_4bit"
    )
    results["QLoRA (Base 4-bit NF4)"] = {"peak_vram_gb": vram_qlora, "time_seconds": time_qlora}
    print(f"\n📊 Resultado QLoRA 4-bit -> Pico VRAM: {vram_qlora:.2f} GB | Tempo: {time_qlora:.1f}s")

    out_file = "results/training_vram_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Métricas de treinamento salvas em {out_file}")

if __name__ == "__main__":
    main()