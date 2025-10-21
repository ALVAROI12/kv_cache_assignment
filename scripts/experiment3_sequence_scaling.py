import torch
import json
import time
import numpy as np
from run_model import load_model, MemoryTracker
import matplotlib.pyplot as plt
import pandas as pd

def measure_with_sequence_length(model, tokenizer, seq_length, max_new_tokens=50, device="cuda"):
    """Generate with specific input sequence length"""
    tracker = MemoryTracker()
    
    # Create prompt of desired length
    base_text = "The quick brown fox jumps over the lazy dog. "
    prompt = base_text * (seq_length // 10 + 1)
    
    # Tokenize and truncate to exact length
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs['input_ids'] = inputs['input_ids'][:, :seq_length]
    inputs['attention_mask'] = inputs['attention_mask'][:, :seq_length]
    
    actual_input_length = inputs['input_ids'].shape[1]
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    mem_before = tracker.get_memory()
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, use_cache=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Actual measurement
    latencies = []
    num_runs = 3
    
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        latencies.append(end - start)
    
    mem_after = tracker.get_memory()
    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    tokens_generated = outputs.shape[1] - actual_input_length
    mean_latency = np.mean(latencies)
    
    # Estimate KV cache size
    # For distilgpt2: 6 layers, 768 hidden_dim, 2 bytes (FP16)
    n_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    total_seq_len = actual_input_length + tokens_generated
    
    # KV cache: 2 (K and V) * layers * seq_len * hidden_size * bytes
    estimated_cache_mb = (2 * n_layers * total_seq_len * hidden_size * 2) / (1024**2)
    
    results = {
        "input_seq_length": actual_input_length,
        "tokens_generated": tokens_generated,
        "total_seq_length": total_seq_len,
        "mean_latency_sec": mean_latency,
        "latency_per_token_ms": (mean_latency / tokens_generated * 1000) if tokens_generated > 0 else 0,
        "tokens_per_sec": tokens_generated / mean_latency if mean_latency > 0 else 0,
        "peak_gpu_mem_mb": peak_gpu_mem,
        "estimated_cache_mb": estimated_cache_mb,
        "n_layers": n_layers,
        "hidden_size": hidden_size
    }
    
    return results

def experiment_sequence_scaling(model_name="distilgpt2", max_new_tokens=50):
    """Test different sequence lengths"""
    
    model, tokenizer, device = load_model(model_name)
    
    sequence_lengths = [32, 64, 128, 256, 512]
    results = []
    
    print(f"\n{'='*60}")
    print("SEQUENCE LENGTH SCALING EXPERIMENT")
    print(f"{'='*60}")
    
    for seq_len in sequence_lengths:
        print(f"\nSequence length: {seq_len}")
        
        result = measure_with_sequence_length(
            model, tokenizer, seq_len,
            max_new_tokens=max_new_tokens,
            device=device
        )
        
        results.append(result)
        
        print(f"  Latency per token: {result['latency_per_token_ms']:.2f} ms")
        print(f"  Estimated cache size: {result['estimated_cache_mb']:.2f} MB")
        print(f"  Peak GPU memory: {result['peak_gpu_mem_mb']:.2f} MB")
    
    # Save results
    with open('logs/experiment3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    df = pd.DataFrame(results)
    df.to_csv('logs/experiment3_summary.csv', index=False)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cache memory vs sequence length
    axes[0, 0].plot(df['total_seq_length'], df['estimated_cache_mb'], 'o-', linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Total Sequence Length')
    axes[0, 0].set_ylabel('Estimated KV Cache (MB)')
    axes[0, 0].set_title('KV Cache Size vs Sequence Length')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Show linear relationship
    z = np.polyfit(df['total_seq_length'], df['estimated_cache_mb'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['total_seq_length'], p(df['total_seq_length']), "--", alpha=0.5, label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.2f}')
    axes[0, 0].legend()
    
    # Latency per token
    axes[0, 1].plot(df['total_seq_length'], df['latency_per_token_ms'], 's-', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Total Sequence Length')
    axes[0, 1].set_ylabel('Latency per Token (ms)')
    axes[0, 1].set_title('Generation Speed vs Sequence Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Peak GPU memory
    axes[1, 0].plot(df['total_seq_length'], df['peak_gpu_mem_mb'], '^-', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Total Sequence Length')
    axes[1, 0].set_ylabel('Peak GPU Memory (MB)')
    axes[1, 0].set_title('Total Memory vs Sequence Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Throughput
    axes[1, 1].plot(df['total_seq_length'], df['tokens_per_sec'], 'd-', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Total Sequence Length')
    axes[1, 1].set_ylabel('Throughput (tokens/s)')
    axes[1, 1].set_title('Throughput vs Sequence Length')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/experiment3_sequence_scaling.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to plots/experiment3_sequence_scaling.png")
    
    return results, df

if __name__ == "__main__":
    results, df = experiment_sequence_scaling(max_new_tokens=50)
    
    print(f"\n{'='*60}")
    print("THEORETICAL VALIDATION")
    print(f"{'='*60}")
    
    # Check if cache scales linearly
    from scipy.stats import pearsonr
    corr, _ = pearsonr(df['total_seq_length'], df['estimated_cache_mb'])
    print(f"Cache-Sequence correlation: {corr:.4f} (should be ~1.0 for linear)")
    
    # Check complexity
    print(f"\nCache size increases {df['estimated_cache_mb'].iloc[-1]/df['estimated_cache_mb'].iloc[0]:.2f}x")
    print(f"Sequence length increases {df['total_seq_length'].iloc[-1]/df['total_seq_length'].iloc[0]:.2f}x")
    print("These should match for O(S) scaling")