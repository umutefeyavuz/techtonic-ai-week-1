# Techtonic-AI – Ultra-Optimized TinyLlama Server

This repository contains the latest version of a highly optimized TinyLlama-based chat server.  

- **Better memory management**:  
  • Automatic cleanup when RAM usage exceeds thresholds (e.g. > 80-85 %).  
  • GPU/CPU cache clearance when needed.  
  • Cleaner response trimming to avoid large generated outputs.

- **Performance testing optimized**:  
  • Quick validation scenarios.  
  • A more comprehensive performance test suite measuring response time, CPU & memory peaks.  
  • Reports & logs to track pass rate vs targets.

## Usage

1. Clone this repository.  
2. Install dependencies:  
   ```bash
   pip install torch transformers psutil
