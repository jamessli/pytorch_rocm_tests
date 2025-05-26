**# PyTorch CE Microbenchmarking Tests**

Basic PyTorch Tests module for the CE microbenchmarking suite, designed to validate **MI300X/MI325X** out-of-the-box performance.

**## Overview
**
This package provides a collection of PyTorch-based microbenchmarks that form the core “Tests” module of the larger CE microbenchmarking suite. It exercises https://github.com/ROCm/pytorch-micro-benchmarking to test key neural networks and workloads to measure baseline performance on AMD’s MI300X and MI325X GPUs, right out of the box.

**## Prerequisites**

- **Python 3.8+**  
- **PyTorch** compiled with ROCm support (compatible with ROCm 5.5 or later)  
- **ROCm** platform installed and accessible  
- `git`  

Optionally, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

**## Installation**
Clone the microbenchmarks repo

```
git clone https://github.com/AMD-AIG-AIMA/CE_microbenchmarks
cd ce-pytorch-tests
```

**## Install Python dependencies**

```
git clone https://github.com/AMD-AIG-AIMA/CE_microbenchmarks.git
cd CE_microbenchmarks
git submodule update --init --recursive
Note: This module leverages ROCm’s PyTorch Micro-Benchmarking as its backbone.
```

**## Configuration**
All models, batch sizes, precisions, and other runtime parameters are controlled via a JSON config file.

Copy the sample template:
```
cp sample.json my_config.json
Edit my_config.json to adjust:

models (e.g. resnet50, bert)
batch_size
precision (fp32, fp16, etc.)
number_of_iterations
warmup_iterations
```

**## Example snippet:**
```
{
  "models": ["resnet50", "bert"],
  "batch_size": 32,
  "precision": "fp16",
  "warmup_iterations": 10,
  "benchmark_iterations": 50
}
```
**## Usage**
With your config file prepared, invoke the runner script:
```
python run_microbenchmarks.py --config my_config.json
```
The script will parse your config, launch each model’s benchmark on the available GPU(s), emit per-model throughput, latency, and memory-utilization logs

Results will be saved under ./results/<timestamp>/.

**## Directory Structure**
```
ce-pytorch-tests/
├── run_microbenchmarks.py     # Main entrypoint
├── sample.json                # Configuration template
├── benchmarks/                # Model-specific benchmark scripts
│   ├── resnet50.py
│   └── bert.py
├── utils/                     # Helper functions (logging, device setup)
│   └── logger.py
├── requirements.txt
└── README.md
```

AMD AIG AIMA CE Microbenchmarks: https://github.com/AMD-AIG-AIMA/CE_microbenchmarks
ROCm PyTorch Micro-Benchmarking: https://github.com/ROCm/pytorch-micro-benchmarking
