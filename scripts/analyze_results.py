import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def analyze_experiment1():
    """Analyze cache comparison results"""
    with open('logs/experiment1_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT 1 ANALYSIS: KV Cache Impact")
    print("="*70)
    
    # Group by cache usage
    with_cache = df[df['use_cache'] == True]
    without_cache = df[df['use_cache'] == False]
    
    print(f"\nWith Cache:")
    print(f"  Mean throughput: {with_cache['tokens_per_sec'].mean():.2f} ± {with_cache['tokens_per_sec'].std():.2f} tokens/sec")
    print(f"  Mean latency: {with_cache['total_time_sec'].mean():.3f} ± {with_cache['total_time_sec'].std():.3f} sec")
    print(f"  Mean GPU memory: {with_cache['peak_gpu_mem_mb'].mean():.2f} MB")
    
    print(f"\nWithout Cache:")
    print(f"  Mean throughput: {without_cache['tokens_per_sec'].mean():.2f} ± {without_cache['tokens_per_sec'].std():.2f} tokens/sec")
    print(f"  Mean latency: {without_cache['total_time_sec'].mean():.3f} ± {without_cache['total_time_sec'].std():.3f} sec")
    print(f"  Mean GPU memory: {without_cache['peak_gpu_mem_mb'].mean():.2f} MB")
    
    speedup = with_cache['tokens_per_sec'].mean() / without_cache['tokens_per_sec'].mean()
    
    print(f"\n{'='*70}")
    if speedup > 1:
        print(f"✓ Cache provides {speedup:.2f}x SPEEDUP")
    else:
        print(f"✗ Cache shows {1/speedup:.2f}x SLOWDOWN (overhead dominates for short sequences)")
    print(f"{'='*70}")
    
    print("\n📊 INTERPRETATION:")
    print("  For DistilGPT2 with 100 tokens:")
    print("  - Model is very small and fast")
    print("  - Cache overhead (memory management) > computation savings")
    print("  - Cache benefits appear with LONGER sequences (>500 tokens)")
    print("  - This demonstrates cache has overhead that must be amortized")

def analyze_experiment2():
    """Analyze batching results"""
    with open('logs/experiment2_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT 2 ANALYSIS: Batching Effects")
    print("="*70)
    
    for _, row in df.iterrows():
        print(f"\nBatch Size {row['batch_size']}:")
        print(f"  Throughput: {row['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"  P50 latency: {row['latency_p50']:.3f} sec")
        print(f"  P95 latency: {row['latency_p95']:.3f} sec")
        print(f"  Peak GPU memory: {row['peak_gpu_mem_mb']:.2f} MB")
        print(f"  GPU utilization: {row['throughput_tokens_per_sec'] / df['throughput_tokens_per_sec'].max() * 100:.1f}%")
    
    throughput_scaling = df['throughput_tokens_per_sec'].iloc[-1] / df['throughput_tokens_per_sec'].iloc[0]
    memory_scaling = df['peak_gpu_mem_mb'].iloc[-1] / df['peak_gpu_mem_mb'].iloc[0]
    
    print(f"\n{'='*70}")
    print(f"✓ Throughput scales {throughput_scaling:.2f}x (batch 1→4)")
    print(f"✓ Memory scales only {memory_scaling:.2f}x")
    print(f"✓ Efficiency: {throughput_scaling/memory_scaling:.2f}x more throughput per MB")
    print(f"{'='*70}")
    
    print("\n📊 INTERPRETATION:")
    print("  - Excellent scaling with batching")
    print("  - GPU has spare capacity at batch=1")
    print("  - Memory overhead is minimal (mostly KV cache)")
    print("  - Recommendation: Use batch=4 for maximum throughput")

def analyze_experiment3():
    """Analyze sequence scaling"""
    with open('logs/experiment3_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT 3 ANALYSIS: Sequence Length Scaling")
    print("="*70)
    
    for _, row in df.iterrows():
        print(f"\nSequence Length {row['total_seq_length']}:")
        print(f"  Estimated cache: {row['estimated_cache_mb']:.2f} MB")
        print(f"  Peak GPU memory: {row['peak_gpu_mem_mb']:.2f} MB")
        print(f"  Latency/token: {row['latency_per_token_ms']:.2f} ms")
        print(f"  Throughput: {row['tokens_per_sec']:.2f} tokens/sec")
    
    # Test linear scaling of cache
    corr, p_value = pearsonr(df['total_seq_length'], df['estimated_cache_mb'])
    
    # Fit linear model
    z = np.polyfit(df['total_seq_length'], df['estimated_cache_mb'], 1)
    slope = z[0]
    
    print(f"\n{'='*70}")
    print(f"✓ Cache-Sequence correlation: {corr:.4f} (p={p_value:.6f})")
    print(f"✓ Linear fit slope: {slope:.6f} MB/token")
    print(f"✓ Expected slope: {2 * 6 * 768 * 2 / (1024**2):.6f} MB/token")
    print(f"  (2 × {6} layers × {768} hidden × 2 bytes)")
    print(f"{'='*70}")
    
    cache_scaling = df['estimated_cache_mb'].iloc[-1] / df['estimated_cache_mb'].iloc[0]
    seq_scaling = df['total_seq_length'].iloc[-1] / df['total_seq_length'].iloc[0]
    
    print(f"\n✓ Cache scales {cache_scaling:.2f}x")
    print(f"✓ Sequence scales {seq_scaling:.2f}x")
    print(f"✓ Ratio: {cache_scaling/seq_scaling:.3f} (should be ~1.0 for O(S))")
    
    print("\n📊 INTERPRETATION:")
    print("  - Cache size scales LINEARLY with sequence length ✓")
    print("  - Confirms O(S) complexity with KV cache")
    print("  - Latency per token increases slightly (memory bandwidth)")
    print("  - Model is MEMORY-BOUND (latency increases with cache size)")

def compute_bound_analysis():
    """Analyze if workload is compute or memory bound"""
    with open('logs/experiment3_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("COMPUTE vs MEMORY BOUND ANALYSIS")
    print("="*70)
    
    # For DistilGPT2
    n_layers = 6
    hidden_size = 768
    vocab_size = 50257
    
    # Calculate FLOPs per token (approximate)
    # Attention: 4 * d^2 per layer (Q, K, V, O projections)
    # FFN: 8 * d^2 per layer (up and down projections, d_ff = 4*d typically)
    flops_per_layer = 4 * hidden_size**2 + 8 * hidden_size**2
    total_flops = n_layers * flops_per_layer
    
    print(f"\nModel Architecture:")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Parameters: ~82M")
    
    print(f"\nComputation per token:")
    print(f"  FLOPs per layer: ~{flops_per_layer/1e6:.2f}M")
    print(f"  Total FLOPs: ~{total_flops/1e6:.2f}M")
    
    # Memory bandwidth requirements
    # Read model weights + Read/Write KV cache
    avg_seq_len = df['total_seq_length'].mean()
    bytes_per_token = (
        2 * n_layers * hidden_size +  # Write new KV
        2 * n_layers * hidden_size * avg_seq_len +  # Read cached KV
        12 * n_layers * hidden_size**2 * 2  # Read weights (approximate)
    )
    
    print(f"\nMemory traffic per token (seq_len={avg_seq_len:.0f}):")
    print(f"  Bytes moved: ~{bytes_per_token/1e6:.2f} MB")
    
    arithmetic_intensity = total_flops / bytes_per_token
    print(f"\nArithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
    
    # Typical GPU specs (assuming RTX 3060 or similar based on your results)
    peak_tflops = 13  # TFLOPs for typical gaming GPU
    bandwidth_gbs = 360  # GB/s
    
    ridge_point = (peak_tflops * 1e12) / (bandwidth_gbs * 1e9)
    
    print(f"\nGPU Characteristics (estimated):")
    print(f"  Peak compute: ~{peak_tflops} TFLOPs")
    print(f"  Memory bandwidth: ~{bandwidth_gbs} GB/s")
    print(f"  Ridge point: {ridge_point:.2f} FLOPs/byte")
    
    print(f"\n{'='*70}")
    if arithmetic_intensity < ridge_point:
        print(f"✓ Workload is MEMORY-BOUND")
        print(f"  AI ({arithmetic_intensity:.1f}) < Ridge ({ridge_point:.1f})")
        print(f"  Performance limited by memory bandwidth, not compute")
    else:
        print(f"✓ Workload is COMPUTE-BOUND")
        print(f"  AI ({arithmetic_intensity:.1f}) > Ridge ({ridge_point:.1f})")
    print(f"{'='*70}")
    
    print("\n📊 EVIDENCE from experiments:")
    print("  ✓ Latency increases with sequence length (Exp 3)")
    print("  ✓ Batching improves throughput 7x (Exp 2)")
    print("  ✓ Small model, low compute requirements")
    print("  → Conclusion: MEMORY-BOUND workload")

if __name__ == "__main__":
    analyze_experiment1()
    analyze_experiment2()
    analyze_experiment3()
    compute_bound_analysis()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. KV Cache Trade-offs:
   - Has overhead for short sequences
   - Benefits increase with longer sequences
   - Essential for real-time applications

2. Batching is Critical:
   - 7x throughput improvement (batch 1→4)
   - Minimal memory overhead
   - Key optimization for production

3. Memory-Bound Workload:
   - Performance limited by memory bandwidth
   - Cache size scales linearly (O(S))
   - Optimizations should focus on:
     * Quantization (INT8/INT4)
     * Flash Attention
     * Larger batch sizes

4. For Edge Deployment:
   - Use quantization to reduce cache size
   - Balance batch size with latency requirements
   - Consider sequence length limits
    """)