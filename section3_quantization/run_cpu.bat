@echo off
REM Full CPU benchmark for this laptop (Intel UHD 620, no CUDA).
REM Leg 1 (fp32) ~15 min; leg 2 (GGUF Q4_K_M) downloads ~1 GB once, then runs faster.

python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --precision fp16 --out results_fp16.json
if errorlevel 1 exit /b 1

python benchmark.py --precision gguf-q4 --out results_gguf.json
if errorlevel 1 exit /b 1

python benchmark.py --compare results_fp16.json results_gguf.json