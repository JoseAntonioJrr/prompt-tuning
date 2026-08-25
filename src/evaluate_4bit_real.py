# src/evaluate_4bit_real.py
import torch
import time
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import setup_environment, get_imdb_dataset

def main():
    setup_environment()
    model_path = "outputs/fused_lora_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo fundido não encontrado em {model_path}!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]
    
    _, _, raw_val_data = get_imdb_dataset(tokenizer, num_samples=25000)

    # Configuração de Quantização Real 4-bit (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("\n📦 Carregando modelo em 4-bit Real na GPU (BitsAndBytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    correct = 0
    total = len(raw_val_data)

    print("🧠 Rodando avaliação analítica na GPU...")
    with torch.no_grad():
        for idx, item in enumerate(raw_val_data):
            prompt = f"Review: {item['text']}\nSentiment:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            pos_score = next_token_logits[pos_id].item()
            neg_score = next_token_logits[neg_id].item()
            
            predicted = "positive" if pos_score > neg_score else "negative"
            target = "negative" if item["label"] == 0 else "positive"
            
            if predicted == target:
                correct += 1

    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    accuracy = (correct / total) * 100

    print("\n" + "="*45)
    print(f"🎯 4-Bit Real (NF4): {accuracy:.2f}% de Acurácia")
    print(f"⏱️ Tempo Total: {total_time:.2f}s")
    print(f"💾 Pico de VRAM Real: {peak_vram:.2f} GB")
    print("="*45)

    results = {
        "4-Bit Real NF4": {
            "accuracy": accuracy,
            "vram_gb": peak_vram,
            "time_seconds": total_time
        }
    }
    with open("results/real_4bit_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()