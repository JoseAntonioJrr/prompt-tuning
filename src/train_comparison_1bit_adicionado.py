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

def binarize_base_model(model):
    """Binariza os pesos dos módulos lineares para 1-bit {-γ, +γ} e congela."""
    print("⚡ Binarizando pesos do modelo base para 1-bit...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                w = module.weight.data
                gamma = torch.mean(torch.abs(w), dim=-1, keepdim=True)
                w_bin = gamma * torch.sign(w)
                module.weight.data.copy_(w_bin)
                module.weight.requires_grad = False

def run_training(mode: str, train_dataset, tokenizer, output_dir: str):
    """
    mode: 'bf16', '4bit_qlora', '1bit_hqq_lora'
    """
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

    if mode == "4bit_qlora":
        print("\n⚡ [Modo QLoRA (4-bit NF4)] Carregando modelo quantizado...")
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
        model = get_peft_model(model, lora_config)

    elif mode == "1bit_hqq_lora":
        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        
        print("\n⚡ [Modo 1-Bit Real (HQQ)] Carregando modelo base...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("⚡ Empacotando pesos fisicamente para 1-bit (Manual Bit-Packing)...")
        quant_config = BaseQuantizeConfig(nbits=1, group_size=64)
        
        # Função recursiva para injetar 1-bit apenas nas camadas corretas, fugindo do erro do rotary_emb
        def replace_linear_with_hqq(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear) and name != "lm_head":
                    # Substitui a camada linear pelo HQQ empacotado e deleta os pesos originais
                    setattr(module, name, HQQLinear(child, quant_config, compute_dtype=torch.bfloat16))
                else:
                    replace_linear_with_hqq(child)
                    
        replace_linear_with_hqq(model)
        
        # Prepara a base congelada para receber os gradientes do LoRA
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            
        model = get_peft_model(model, lora_config)

    else:  # bf16
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
        max_steps=100,
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

    # 1. LoRA Padrão (BF16)
    vram_bf16, time_bf16 = run_training(
        mode="bf16", 
        train_dataset=train_dataset, 
        tokenizer=tokenizer, 
        output_dir="outputs/train_lora_bf16"
    )
    results["LoRA Padrão (BF16)"] = {"peak_vram_gb": vram_bf16, "time_seconds": time_bf16}
    print(f"\n📊 Resultado LoRA BF16 -> Pico VRAM: {vram_bf16:.2f} GB | Tempo: {time_bf16:.1f}s")

    # 2. QLoRA (Base 4-bit)
    vram_qlora, time_qlora = run_training(
        mode="4bit_qlora", 
        train_dataset=train_dataset, 
        tokenizer=tokenizer, 
        output_dir="outputs/train_qlora_4bit"
    )
    results["QLoRA (Base 4-bit NF4)"] = {"peak_vram_gb": vram_qlora, "time_seconds": time_qlora}
    print(f"\n📊 Resultado QLoRA 4-bit -> Pico VRAM: {vram_qlora:.2f} GB | Tempo: {time_qlora:.1f}s")

    # 3. QLoRA (Base 1-bit HQQ)
    vram_1bit, time_1bit = run_training(
        mode="1bit_hqq_lora", 
        train_dataset=train_dataset, 
        tokenizer=tokenizer, 
        output_dir="outputs/train_lora_1bit_hqq"
    )
    results["QLoRA (Base 1-bit HQQ)"] = {"peak_vram_gb": vram_1bit, "time_seconds": time_1bit}
    print(f"\n📊 Resultado QLoRA 1-bit -> Pico VRAM: {vram_1bit:.2f} GB | Tempo: {time_1bit:.1f}s")

    out_file = "results/training_vram_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Métricas completas salvas em {out_file}")

if __name__ == "__main__":
    main()