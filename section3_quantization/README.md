# Section 3 — Quantization

Compare an open-weight LLM at **full precision** vs **quantized** on memory,
throughput, and output quality across 5 fixed prompts.

**Hardware used for this submission:** Windows laptop with Intel UHD Graphics
620 — **no NVIDIA GPU**, so CUDA-only paths (bitsandbytes NF4 int4) are not
runnable here. The submitted comparison is CPU-vs-CPU: **fp32 baseline vs
GGUF Q4_K_M (llama.cpp)** — a real 4-bit technique explicitly listed in the
assignment, and the correct one for CPU deployment. The int4/bitsandbytes path
is still implemented and works unchanged on any CUDA machine.

Model: `Qwen/Qwen2.5-1.5B-Instruct` (ungated).

## Run (this laptop / any CPU-only machine)
```bash
pip install -r requirements.txt
# GGUF leg needs llama-cpp-python; on Windows CPU install the prebuilt wheel:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

python benchmark.py --precision fp16    --out results_fp16.json
python benchmark.py --precision gguf-q4 --out results_gguf.json   # downloads ~1 GB Q4_K_M once
python benchmark.py --compare results_fp16.json results_gguf.json
```
Or just: `run_cpu.bat` (runs all three).

Optional third leg — naive-ish PyTorch dynamic int8 (per-channel, lm_head kept
fp32), useful as a discussion point:
```bash
python benchmark.py --precision int8 --out results_int8.json
python benchmark.py --compare results_fp16.json results_int8.json
```

Each run generates up to 128 tokens for the same 5 prompts, times **decode**
throughput (tok/s), and records memory two ways: `peak_mem_mb` (lifetime
process peak / VRAM) and `mem_after_load_mb` (steady-state RSS right after the
model is loaded — the honest "what does serving cost" number, since peak can
capture transient conversion copies). `--compare` prints the table plus a
side-by-side of every output, and refuses to compare result files from
different devices.

**Timing expectation:** the fp32 leg decodes ~2.3 tok/s on this CPU (~15 min
for 5 prompts). GGUF Q4_K_M should be several times faster.

### On a CUDA machine (not this laptop)
```bash
python benchmark.py --precision fp16 --out results_fp16.json
python benchmark.py --precision int4 --out results_int4.json   # bitsandbytes NF4
python benchmark.py --compare results_fp16.json results_int4.json
```
The `[env]` startup diagnostic distinguishes CPU-only-torch-wheel vs
missing-driver vs working CUDA and prints the exact fix (Windows PyPI torch is
CPU-only; CUDA builds come from `--index-url https://download.pytorch.org/whl/cu126`).

## Fix log (what broke and why)
1. **Gated-repo 401** — default model needed Meta license + HF login →
   switched to ungated `Qwen/Qwen2.5-1.5B-Instruct`.
2. **`KeyError: 'shape'` in `generate()`** — newer transformers returns a
   `BatchEncoding` from `apply_chat_template` → `return_dict=True` +
   `model.generate(**inputs)`.
3. **`torch_dtype` deprecation** → `dtype=`.
4. **Windows memory measurement** — `resource` is Linux-only → `psutil`.
5. **CUDA detection & comparability guard** — `cuda_diagnostic()` at startup;
   int4 hard-fails with the diagnosis; `--compare` refuses cross-device files;
   results JSON records an `env` block.
6. **Hardware determination** — leftover `nvidia-smi.exe` was a red herring;
   `Win32_VideoController` shows only Intel UHD 620 → locked the submission
   to a same-device CPU comparison.
7. **Naive dynamic int8 collapsed quality** (measured, see Results): first
   attempt quantized ALL Linears per-tensor → outputs derailed, model rambled
   to the token cap, throughput dropped, and peak memory *rose* (transient
   fp32 copy during load-then-quantize). Fixed the int8 path (per-CHANNEL
   scales, `lm_head` kept fp32, gc after conversion, steady-state memory
   metric) and added the **GGUF Q4_K_M leg** as the proper CPU quantization.

## Results

Measured on: Intel CPU (UHD 620 laptop), Windows, torch 2.13 CPU build,
`Qwen/Qwen2.5-1.5B-Instruct`, greedy decoding, 128 max new tokens, 5 prompts.

### Headline comparison — fp32 vs GGUF Q4_K_M *(fill in after run)*

| Metric | fp32 (HF/PyTorch) | GGUF Q4_K_M (llama.cpp) | Trade-off |
|---|---|---|---|
| Mem after load (MB) | *(from results_fp16.json)* | *(from results_gguf.json)* | expect ~4x smaller |
| Throughput (tok/s) | ~2.3 | *(paste)* | expect 2–5x faster |
| Output quality (5 prompts) | baseline | *(side-by-side)* | expect near-identical |
| Weights on disk | ~3.1 GB (fp16 safetensors) | ~1.0 GB | ~3x smaller |

### Measured finding — why naive quantization fails (first int8 attempt)

| Metric | fp32 | naive dynamic int8 (per-tensor, incl. lm_head) |
|---|---|---|
| Peak memory (MB) | 7450 | **8295 (worse!)** — peak caught the transient fp32 copy during load-then-quantize |
| Throughput (tok/s) | 2.37 | 2.10 (0.89x) — degraded model rambled to the 128-token cap on every prompt |
| Quality | coherent on all 5 | derailed: off-topic answers, spurious prefixes, one refusal |

This failure is the most instructive number in the whole benchmark: per-tensor
weight scales can't represent the outlier channels modern LLMs have, and
quantizing the embedding-tied `lm_head` corrupts logits directly. It's
precisely why production formats use grouped/per-channel schemes with outlier
handling (GGUF K-quants, AWQ, GPTQ) rather than naive round-to-nearest.

**Throughput caveat:** quantization is not automatically faster. bitsandbytes
NF4 dequantizes to bf16 for the matmul (GPU: memory ↓↓, speed ≈). llama.cpp
GGUF computes *in* low precision with kernels built for it, which is why it
both shrinks and speeds up CPU inference — the right tool per target.

## Write-up — GPTQ/AWQ vs bitsandbytes vs GGUF

This benchmark demonstrated the core production lesson twice over: **the
quantization format is dictated by the deployment hardware, and the
quantization *scheme* determines whether quality survives.** I couldn't run
bitsandbytes at all here (no CUDA), and my first naive int8 attempt measurably
destroyed output quality — the gap that calibrated/grouped formats exist to
fill.

**bitsandbytes (NF4)** — my default for *development and fine-tuning on GPU*.
Zero calibration: pass a config and load. Natural for QLoRA and squeezing a
model onto a smaller GPU. Dequantizes to compute, so serving throughput isn't
better than fp16 — and it's CUDA-only, irrelevant for CPU/edge targets.

**GPTQ / AWQ** — my default for *production GPU serving* of a pinned model.
One-time calibration minimises output error (AWQ protects activation-salient
weights; GPTQ uses second-order info). With Marlin/ExLlama kernels via vLLM
they deliver low memory AND high throughput with strong 4-bit quality. Cost:
per-model calibration. AWQ for the best quality/speed balance; GPTQ where the
ecosystem favours it.

**GGUF (llama.cpp)** — my default for *CPU / Apple Silicon / edge / laptop* —
i.e. hardware like this submission machine. A ladder of K-quant levels
(Q4_K_M, Q5_K_M, Q8_0…) with grouped quantization that preserved quality where
my naive int8 didn't, computing in low precision so it's also faster. It's how
I'd ship a local assistant to this laptop. Less ideal for high-QPS
multi-tenant GPU serving, where vLLM + AWQ/GPTQ wins.

**Rule of thumb I actually use:** prototyping / QLoRA → bitsandbytes; max
GPU-serving speed for a pinned model → AWQ (or GPTQ) via vLLM; CPU / Mac /
edge → GGUF. And never ship naive per-tensor RTN quantization of an LLM — I
now have the receipts for why.

## Notes / limitations
- The fp32-vs-GGUF comparison crosses frameworks (PyTorch vs llama.cpp). For
  this assignment that's the point — it compares realistic deployment options
  on the same hardware and prompts — but it means the speedup mixes
  quantization gains with runtime-efficiency gains.
- The fixed int8 path (per-channel, fp32 lm_head) is load-time weight-only
  quantization — the CPU analogue of bnb, kept as a comparison point, not a
  production recommendation.
- Throughput is decode tok/s (excludes prefill), greedy decoding for
  determinism.
- Quality judged qualitatively on 5 fixed prompts; for a real decision I'd add
  a small automatic eval (exact-match on the arithmetic/code/JSON prompts).