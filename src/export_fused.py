# src/export_fused.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    lora_path = "outputs/lora_checkpoint"
    output_dir = "outputs/fused_lora_model"

    print("📦 Carregando modelo base e LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu"
    )
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("🔀 Fundindo pesos (merge_and_unload)...")
    fused_model = peft_model.merge_and_unload()

    print(f"💾 Salvando modelo fundido em {output_dir}...")
    fused_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Modelo fundido salvo com sucesso!")

if __name__ == "__main__":
    main()