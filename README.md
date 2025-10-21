# Assignment 2: KV-Cache in LLMs

## Setup

### Requirements
```bash
pip install torch transformers accelerate matplotlib pandas numpy psutil pynvml scipy
```

### Hardware Used
- CPU: [Your CPU model]
- RAM: [Your RAM size]
- GPU: [Your GPU model if available]

## Running Experiments

### All at once
```bash
cd scripts
python run_all_experiments.py
```

### Individual experiments
```bash
# Experiment 1: Cache comparison
python experiment1_cache_comparison.py

# Experiment 2: Batching
python experiment2_batching.py

# Experiment 3: Sequence scaling
python experiment3_sequence_scaling.py
```

## Key Findings

### Experiment 1: KV Cache Impact
- **Speedup**: X.Xx faster with cache
- **Memory overhead**: XX MB for cache
- **Conclusion**: Cache trades memory for speed

### Experiment 2: Batching
- Throughput scales X.Xx with batch size
- Memory scales linearly
- Optimal batch size: X for this hardware

### Experiment 3: Sequence Scaling
- Cache size scales O(S) as expected
- Latency per token remains relatively constant
- Confirms memory-bound behavior

## File Structure
```
├── scripts/
│   ├── run_model.py
│   ├── experiment1_cache_comparison.py
│   ├── experiment2_batching.py
│   ├── experiment3_sequence_scaling.py
│   └── run_all_experiments.py
├── logs/
│   ├── experiment1_results.json
│   ├── experiment2_results.json
│   └── experiment3_results.json
├── plots/
│   ├── experiment1_comparison.png
│   ├── experiment2_batching.png
│   └── experiment3_sequence_scaling.png
└── README.md
```# kv_cache_assignment
