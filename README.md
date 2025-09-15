# Techtonic-AI – Ultra-Optimized TinyLlama Server

This repository contains the latest version of a highly optimized TinyLlama-based chat server.  

## What’s New Compared to Last Week

- **Much lighter footprint**:  
  • Cache size reduced and managed via an LRU-style strategy.  
  • Shorter max history, shorter truncation of user input.  
  • Reduced thread count on CPU runs.  

- **Better memory management**:  
  • Automatic cleanup when RAM usage exceeds thresholds (e.g. > 80-85 %).  
  • GPU/CPU cache clearance when needed.  
  • Cleaner response trimming to avoid large generated outputs.

- **More adaptive response generation**:  
  • Different modes depending on system load (emergency / short input / normal).  
  • Predefined quick replies for common inputs.  
  • Dynamic parameters (`max_tokens`, `top_k`, sampling vs greedy) based on input length and load.

- **Performance testing added**:  
  • Quick validation scenarios.  
  • A more comprehensive performance test suite measuring response time, CPU & memory peaks.  
  • Reports & logs to track pass rate vs targets.

## Usage

1. Clone this repository.  
2. Install dependencies:  
   ```bash
   pip install torch transformers psutil
