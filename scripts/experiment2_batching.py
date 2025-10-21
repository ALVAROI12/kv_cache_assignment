import torch
import json
import time
import numpy as np
from run_model import load_model, MemoryTracker
import matplotlib.pyplot as plt
import pandas as pd

def generate_batch(model, tokenizer, prompts, max_new_tokens=50, device="cuda"):
    """Generate for multiple prompts in parallel"""
    tracker = MemoryTracker()
    
    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    batch_size = len(prompts)
    input_length = inputs.input_ids.shape[1]
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    mem_before = tracker.get_memory()
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=5,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Multiple runs for stable timing
    latencies = []
    num_runs = 5
    
    for run in range(num_runs):
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
    
    # Calculate metrics
    tokens_generated = (outputs.shape[1] - input_length) * batch_size
    
    results = {
        "batch_size": batch_size,
        "tokens_generated": tokens_generated,
        "latency_mean": np.mean(latencies),
        "latency_p50": np.percentile(latencies, 50),
        "latency_p95": np.percentile(latencies, 95),
        "latency_std": np.std(latencies),
        "throughput_tokens_per_sec": tokens_generated / np.mean(latencies),
        "peak_gpu_mem_mb": peak_gpu_mem,
        "gpu_mem_delta_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"]
    }
    
    return results

def experiment_batching(model_name="distilgpt2", max_new_tokens=50):
    """Test different batch sizes"""
    
    model, tokenizer, device = load_model(model_name)
    
    base_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances,",
        "Scientists have recently discovered",
        "The impact of climate change on"
    ]
    
    batch_sizes = [1, 2, 4]
    results = []
    
    print(f"\n{'='*60}")
    print("BATCHING EXPERIMENT")
    print(f"{'='*60}")
    
    for batch_size in batch_sizes:
        prompts = base_prompts[:batch_size]
        
        print(f"\nBatch size: {batch_size}")
        
        result = generate_batch(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            device=device
        )
        
        results.append(result)
        
        print(f"  P50 latency: {result['latency_p50']:.3f}s")
        print(f"  P95 latency: {result['latency_p95']:.3f}s")
        print(f"  Throughput: {result['throughput_tokens_per_sec']:.2f} tokens/s")
        print(f"  Peak GPU memory: {result['peak_gpu_mem_mb']:.2f} MB")
    
    # Save results
    with open('logs/experiment2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    df = pd.DataFrame(results)
    df.to_csv('logs/experiment2_summary.csv', index=False)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latency
    axes[0, 0].plot(df['batch_size'], df['latency_p50'], 'o-', label='P50', linewidth=2)
    axes[0, 0].plot(df['batch_size'], df['latency_p95'], 's-', label='P95', linewidth=2)
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('Latency (seconds)')
    axes[0, 0].set_title('Latency vs Batch Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Throughput
    axes[0, 1].plot(df['batch_size'], df['throughput_tokens_per_sec'], 'o-', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Throughput (tokens/s)')
    axes[0, 1].set_title('Throughput vs Batch Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory
    axes[1, 0].plot(df['batch_size'], df['peak_gpu_mem_mb'], 'o-', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Peak GPU Memory (MB)')
    axes[1, 0].set_title('Memory Usage vs Batch Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficiency (throughput per MB)
    efficiency = df['throughput_tokens_per_sec'] / df['peak_gpu_mem_mb']
    axes[1, 1].plot(df['batch_size'], efficiency, 'o-', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Tokens/s per MB')
    axes[1, 1].set_title('Memory Efficiency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/experiment2_batching.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to plots/experiment2_batching.png")
    
    return results, df

if __name__ == "__main__":
    results, df = experiment_batching(max_new_tokens=50)
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    print(f"Throughput scales {df['throughput_tokens_per_sec'].iloc[-1]/df['throughput_tokens_per_sec'].iloc[0]:.2f}x")
    print(f"Memory scales {df['peak_gpu_mem_mb'].iloc[-1]/df['peak_gpu_mem_mb'].iloc[0]:.2f}x")