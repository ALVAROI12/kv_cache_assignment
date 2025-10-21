import torch
import json
from run_model import load_model, generate_text, MemoryTracker
import matplotlib.pyplot as plt
import pandas as pd

def experiment_cache_comparison(model_name="distilgpt2", max_new_tokens=100):
    """Compare generation with and without KV cache"""
    
    model, tokenizer, device = load_model(model_name)
    
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Scientists have discovered that"
    ]
    
    results = []
    
    for use_cache in [True, False]:
        cache_type = "with_cache" if use_cache else "without_cache"
        print(f"\n{'='*60}")
        print(f"Running experiments {cache_type}")
        print(f"{'='*60}")
        
        for idx, prompt in enumerate(test_prompts):
            print(f"\nPrompt {idx+1}: {prompt[:50]}...")
            
            result = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                device=device
            )
            
            result['experiment'] = cache_type
            result['prompt_idx'] = idx
            results.append(result)
            
            print(f"  Tokens/sec: {result['tokens_per_sec']:.2f}")
            print(f"  Total time: {result['total_time_sec']:.3f}s")
            print(f"  Peak GPU mem: {result['peak_gpu_mem_mb']:.2f} MB")
    
    # Save results
    with open('logs/experiment1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Calculate averages
    summary = df.groupby('experiment').agg({
        'tokens_per_sec': ['mean', 'std'],
        'total_time_sec': ['mean', 'std'],
        'peak_gpu_mem_mb': ['mean', 'std']
    }).round(3)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary)
    
    summary.to_csv('logs/experiment1_summary.csv')
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Tokens per second
    df.boxplot(column='tokens_per_sec', by='experiment', ax=axes[0])
    axes[0].set_title('Throughput Comparison')
    axes[0].set_ylabel('Tokens/second')
    axes[0].set_xlabel('')
    
    # Total time
    df.boxplot(column='total_time_sec', by='experiment', ax=axes[1])
    axes[1].set_title('Total Generation Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xlabel('')
    
    # Memory usage
    df.boxplot(column='peak_gpu_mem_mb', by='experiment', ax=axes[2])
    axes[2].set_title('Peak GPU Memory')
    axes[2].set_ylabel('Memory (MB)')
    axes[2].set_xlabel('')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('plots/experiment1_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to plots/experiment1_comparison.png")
    
    return results, df

if __name__ == "__main__":
    results, df = experiment_cache_comparison(max_new_tokens=100)
    
    # Calculate speedup
    with_cache = df[df['experiment'] == 'with_cache']['tokens_per_sec'].mean()
    without_cache = df[df['experiment'] == 'without_cache']['tokens_per_sec'].mean()
    speedup = with_cache / without_cache
    
    print(f"\n{'='*60}")
    print(f"KV Cache provides {speedup:.2f}x speedup in token generation")
    print(f"{'='*60}")