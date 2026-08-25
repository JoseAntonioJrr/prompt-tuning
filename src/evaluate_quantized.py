# src/evaluate_quantized.py
import torch
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import setup_environment, get_imdb_dataset
from quantization import binarize_fused_model

# Importação protegida e compatível da configuração do KV Cache Quantizado
HAS_KV_QUANT = False
QuantizedCacheConfig = None

try:
    from transformers import QuantizedCacheConfig
    HAS_KV_QUANT = True
except ImportError:
    try:
        from transformers import HqqQuantizedCacheConfig as QuantizedCacheConfig
        HAS_KV_QUANT = True
    except ImportError:
        HAS_KV_QUANT = False

def evaluate_model(model, tokenizer, raw_val_data, cache_config=None):
    correct = 0
    total = len(raw_val_data)
    
    pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    print("🧠 Rodando avaliação analítica baseada em Logits...")
    
    for idx, item in enumerate(raw_val_data):
        prompt = f"Review: {item['text']}\nSentiment:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            if cache_config is not None:
                outputs = model(**inputs, past_key_values=cache_config)
            else:
                outputs = model(**inputs)
                
            next_token_logits = outputs.logits[0, -1, :]
            pos_score = next_token_logits[pos_id].item()
            neg_score = next_token_logits[neg_id].item()
            
        predicted_sentiment = "positive" if pos_score > neg_score else "negative"
        target = "negative" if item["label"] == 0 else "positive"
            
        if predicted_sentiment == target:
            correct += 1
            
    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    accuracy = (correct / total) * 100
    
    return accuracy, total_time, peak_vram

def main():
    setup_environment()
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    lora_path = "outputs/lora_checkpoint"
    
    if not os.path.exists(lora_path):
        print(f"❌ Checkpoint do LoRA não encontrado em {lora_path}!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    _, _, raw_val_data = get_imdb_dataset(tokenizer, num_samples=25000)

    results = {}

    # --- 1. BF16 Original ---
    print("\n📦 Carregando Fused LoRA (BF16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    fused_model = peft_model.merge_and_unload()
    
    acc_bf16, time_bf16, vram_bf16 = evaluate_model(fused_model, tokenizer, raw_val_data)
    results["BF16 Original"] = {"acc": acc_bf16, "time": time_bf16, "vram": vram_bf16}
    print(f"🎯 Fused LoRA (BF16): {acc_bf16:.2f}% | Tempo: {time_bf16:.2f}s | VRAM: {vram_bf16:.2f} GB")

    # --- 2. Binarizado 1-bit ---
    print("\n⚡ Aplicando Binarização de 1-bit nos Pesos...")
    quantized_model = binarize_fused_model(fused_model)
    acc_1bit, time_1bit, vram_1bit = evaluate_model(quantized_model, tokenizer, raw_val_data)
    results["1-Bit Pesos"] = {"acc": acc_1bit, "time": time_1bit, "vram": vram_1bit}
    print(f"🎯 1-Bit Pesos: {acc_1bit:.2f}% | Tempo: {time_1bit:.2f}s | VRAM: {vram_1bit:.2f} GB")

    # --- 3. Binarizado 1-bit + KV Cache 4-bit (Tentativa de Execução) ---
    if HAS_KV_QUANT and QuantizedCacheConfig is not None:
        try:
            print("\n⚡ Ativando KV Cache Quantizado em 4-bits...")
            kv_config = QuantizedCacheConfig(backend="hqq", nbits=4, axis=0)
            acc_kv4, time_kv4, vram_kv4 = evaluate_model(quantized_model, tokenizer, raw_val_data, cache_config=kv_config)
            results["1-Bit + KV 4-Bit"] = {"acc": acc_kv4, "time": time_kv4, "vram": vram_kv4}
            print(f"🎯 1-Bit + KV Cache 4-Bit: {acc_kv4:.2f}% | Tempo: {time_kv4:.2f}s | VRAM: {vram_kv4:.2f} GB")
        except Exception as e:
            print(f"⚠️ Não foi possível aplicar KV Cache 4-bit ({e}).")
            print("💡 Seguindo com a avaliação de 1-bit preservada perfeitamente!")
    else:
        print("\nℹ️ Recurso experimental de KV Cache 4-bit ignorado (versão do transformers não possui a classe no ambiente local).")

    # Salva os resultados das métricas de quantização
    log_file = "results/quantization_metrics.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Métricas salvas com sucesso em {log_file}")

if __name__ == "__main__":
    main()