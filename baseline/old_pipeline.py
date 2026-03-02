"""
Legacy architecture reference — original platform constraints.

Documents the original system's limitations:
  - Sequential execution: 1 client at a time
  - Fixed resources: 4 vCPU, 32 GB RAM (shared across all tasks)
  - Always retrain: no model reuse mechanism
  - Simulation: 3 parallel processes via joblib on shared hardware
  - Max throughput: ~50 clients/day

This file is for documentation. The comparison script (compare.py) simulates
both architectures side by side.
"""

CONSTRAINTS = {
    "max_vcpu": 4,
    "max_ram_gb": 32,
    "execution_model": "sequential",
    "simulation_processes": 3,
    "model_strategy": "always_retrain",
    "estimated_throughput_per_day": 50,
    "step_durations_min": {
        "data_prep":  {"avg": 3,  "max": 11},
        "split":      {"avg": 1,  "max": 3},
        "training":   {"avg": 7,  "max": 23},
        "simulation": {"avg": 12, "max": 20},
    },
}
