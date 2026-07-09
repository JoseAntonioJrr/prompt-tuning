# src/evaluate.py
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import setup_environment, get_imdb_dataset

def evaluate_model(model, tokenizer, raw_val_data):
    correct = 0
    total = len(raw_val_data)
    
    pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]
    
    print("🧠 Rodando avaliação analítica baseada em Logits...")
    for idx, item in enumerate(raw_val_data):
        prompt = f"Review: {item['text']}\nSentiment:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            pos_score = next_token_logits[pos_id].item()
            neg_score = next_token_logits[neg_id].item()
            
        predicted_sentiment = "positive" if pos_score > neg_score else "negative"
        target = "negative" if item["label"] == 0 else "positive"
            
        if predicted_sentiment == target:
            correct += 1
            
    accuracy = (correct / total) * 100
    return accuracy

def main():
    setup_environment()
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    _, _, raw_val_data = get_imdb_dataset(tokenizer, num_samples=5000)

    results = {}
    checkpoints = {
        "Prompt Tuning (Texto)": "outputs/prompt_tuning_checkpoint",
        "Prompt Tuning (Random)": "outputs/prompt_tuning_random_checkpoint",
        "Full Fine-Tuning": "outputs/full_ft_checkpoint"
    }

    for name, path in checkpoints.items():
        if os.path.exists(path):
            print(f"\n🧐 Carregando e avaliando {name}...")
            if "outputs/full_ft_checkpoint" in path:
                model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
                model = PeftModel.from_pretrained(base_model, path)
            
            acc = evaluate_model(model, tokenizer, raw_val_data)
            results[name] = acc
            
            log_file = "results/metrics_log.json"
            if os.path.exists(log_file):
                with open(log_file, "r") as f: data = json.load(f)
                key = "5000_samples_5_epochs"
                method_key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                if key in data and method_key in data[key]:
                    data[key][method_key]["accuracy"] = acc
                    with open(log_file, "w") as f: json.dump(data, f, indent=4)

            if "outputs/full_ft_checkpoint" in path: del model
            else: del model, base_model
            torch.cuda.empty_cache()

    print("\n📊 ========================================")
    print("🏁 RESULTADO COMPARATIVO DE ACURÁCIA FINAL")
    print("============================================")
    for method, acc in results.items():
        print(f"🔹 {method}: {acc:.2f}% de Acurácia")
    print("============================================\n")

if __name__ == "__main__":
    main()