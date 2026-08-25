# src/quantization.py
import torch
import torch.nn as nn

def binarize_fused_model(model):
    """
    Aplica a Binarização de 1-bit (W = gamma * sign(W)) no modelo fundido (Fused LoRA).
    Preserva a lm_head e os embeddings para manter a resolução de busca de logits.
    """
    print("⚡ [Quantization] Iniciando binarização de 1-bit nos pesos...")
    count = 0
    
    for name, module in model.named_modules():
        # Aplica a binarização apenas nas camadas Lineares de atenção (q, k, v, o) e MLP (gate, up, down)
        if isinstance(module, nn.Linear):
            # Protege a lm_head de saída
            if "lm_head" in name:
                continue
                
            with torch.no_grad():
                w = module.weight.data
                # Scale gamma: Média do módulo absoluto dos pesos da matriz
                gamma = torch.mean(torch.abs(w))
                
                # Projeção Binária em {-1, +1}
                w_bin = torch.sign(w)
                w_bin[w_bin == 0] = 1.0  # Trata zeros pontuais
                
                # Atribui o peso binarizado e escalado
                module.weight.data = gamma * w_bin
                count += 1
                
    print(f"✅ [Quantization] Binarização concluída com sucesso em {count} camadas lineares!")
    return model