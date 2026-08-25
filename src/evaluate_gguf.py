# src/evaluate_gguf.py
import time
import json
import os
from llama_cpp import Llama
from utils import setup_environment, get_imdb_dataset
from transformers import AutoTokenizer

def evaluate_gguf_model(model_path, raw_val_data, pos_id, neg_id):
    print(f"\n🧠 Avaliando modelo GGUF: {os.path.basename(model_path)}...")
    
    # n_gpu_layers=-1 aloca todas as camadas na GPU via CUDA
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=512,
        verbose=False,
        logits_all=False
    )
    
    correct = 0
    total = len(raw_val_data)
    
    start_time = time.time()
    
    for idx, item in enumerate(raw_val_data):
        if idx % 1000 == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"  Progresso: {idx}/{total} ({elapsed:.1f}s)")
            
        prompt = f"Review: {item['text']}\nSentiment:"
        tokens = llm.tokenize(prompt.encode('utf-8'))
        
        # Limita tokens para respeitar o contexto de treino
        if len(tokens) > 1500:
            tokens = tokens[:1500]
            
        llm.reset()
        llm.eval(tokens)
        logits = llm._scores[-1]
        
        pos_score = logits[pos_id]
        neg_score = logits[neg_id]
        
        predicted = "positive" if pos_score > neg_score else "negative"
        target = "negative" if item["label"] == 0 else "positive"
        
        if predicted == target:
            correct += 1
            
    total_time = time.time() - start_time
    accuracy = (correct / total) * 100
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    return accuracy, total_time, file_size_mb

def main():
    setup_environment()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.pad_token = tokenizer.eos_token
    
    pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]
    
    _, _, raw_val_data = get_imdb_dataset(tokenizer, num_samples=25000)
    
    models = {
        "GGUF FP16": "outputs/fused_model_fp16.gguf",
        "GGUF 4-Bit (Q4_K_M)": "outputs/fused_model_q4_k_m.gguf",
        "GGUF 2-Bit (Q2_K)": "outputs/fused_model_q2_k.gguf"
    }
    
    results = {}
    
    for name, path in models.items():
        if os.path.exists(path):
            acc, t_time, size_mb = evaluate_gguf_model(path, raw_val_data, pos_id, neg_id)
            results[name] = {
                "accuracy": acc, 
                "time_seconds": t_time, 
                "size_mb": size_mb
            }
            print(f"🎯 {name} -> Acurácia: {acc:.2f}% | Tamanho: {size_mb:.1f} MB | Tempo: {t_time:.2f}s")
            
    output_log = "results/gguf_real_quantization_metrics.json"
    with open(output_log, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n💾 Resultados salvos em {output_log}")

if __name__ == "__main__":
    main()