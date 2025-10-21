import torch
import time
import psutil
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class MemoryTracker:
    """Track CPU and GPU memory usage"""
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        
    def get_memory(self):
        cpu_mem = self.process.memory_info().rss / 1024**2  # MB
        gpu_mem = 0
        if self.gpu_available:
            gpu_mem = torch.cuda.memory_allocated() / 1024**2  # MB
        return {"cpu_mb": cpu_mem, "gpu_mb": gpu_mem}

def load_model(model_name="distilgpt2"):
    """Load model and tokenizer"""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, max_new_tokens=50, use_cache=True, device="cuda"):
    """Generate text with timing and memory tracking"""
    tracker = MemoryTracker()
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    # Track initial memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = tracker.get_memory()
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    
    # Track final memory
    mem_after = tracker.get_memory()
    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    
    # Calculate metrics
    total_time = end_time - start_time
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
    time_per_token = total_time / tokens_generated if tokens_generated > 0 else 0
    
    results = {
        "prompt": prompt,
        "generated_text": generated_text,
        "use_cache": use_cache,
        "input_tokens": input_length,
        "output_tokens": tokens_generated,
        "total_time_sec": total_time,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": time_per_token * 1000,
        "cpu_mem_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mem_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
        "peak_gpu_mem_mb": peak_gpu_mem
    }
    
    return results

if __name__ == "__main__":
    # Test basic generation
    model, tokenizer, device = load_model("distilgpt2")
    
    prompt = "The future of artificial intelligence is"
    
    print("\n" + "="*50)
    print("Testing basic generation...")
    print("="*50)
    
    result = generate_text(model, tokenizer, prompt, max_new_tokens=30, use_cache=True, device=device)
    
    print(f"\nPrompt: {result['prompt']}")
    print(f"\nGenerated: {result['generated_text']}")
    print(f"\nMetrics:")
    print(f"  Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"  ms/token: {result['ms_per_token']:.2f}")
    print(f"  GPU memory: {result['peak_gpu_mem_mb']:.2f} MB")