"""
benchmark.py
------------
Run an open-weight LLM at full precision (bf16/fp16) and quantized, and
measure the trade-off:
  * memory footprint (peak VRAM, or RSS on CPU)
  * throughput (tokens/sec, decode)
  * output quality on 5 fixed prompts (saved for side-by-side reading)

Default model: Qwen/Qwen2.5-1.5B-Instruct (ungated). meta-llama models work
too if you accept the license and `hf auth login`.

Usage:
  python benchmark.py --precision fp16 --out results_fp16.json
  python benchmark.py --precision int4 --out results_int4.json     # CUDA GPU
  python benchmark.py --precision int8 --out results_int8.json     # CPU OK
  python benchmark.py --compare results_fp16.json results_int4.json

Notes:
  * int4 = bitsandbytes NF4, needs a CUDA GPU.
  * int8 = PyTorch dynamic quantization of Linear layers, works on CPU —
    use it if you have no NVIDIA GPU. (GGUF/llama.cpp is the other CPU
    option, see README.)
  * We measure DECODE tokens/sec (generated tokens / generation time).
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompts import PROMPTS

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128


def cuda_diagnostic(require: bool = False) -> dict:
    """Explain WHY CUDA is or isn't usable, so results are never silently CPU.

    The three cases look identical from `torch.cuda.is_available()` but need
    different fixes, so we distinguish them:
      1. torch is a CPU-only build (torch.version.cuda is None)  -> reinstall torch
      2. CUDA build but no visible GPU/driver                    -> driver/hardware
      3. CUDA works                                              -> proceed
    """
    import shutil, subprocess
    env = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,      # None => CPU-only wheel
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "nvidia_smi_gpu": None,
    }
    # Ask the driver directly, independent of torch
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                env["nvidia_smi_gpu"] = out.stdout.strip().splitlines()[0]
        except Exception:
            pass

    if env["cuda_available"]:
        env["gpu_name"] = torch.cuda.get_device_name(0)
        print(f"[env] CUDA OK: {env['gpu_name']} "
              f"(torch {env['torch_version']}, cuda {env['torch_cuda_build']})")
        return env

    # CUDA not available -- diagnose why
    if env["torch_cuda_build"] is None:
        msg = (
            "torch in this venv is a CPU-ONLY build (torch.version.cuda is None).\n"
            f"  Installed: torch {env['torch_version']}\n"
        )
        if env["nvidia_smi_gpu"]:
            msg += (
                f"  BUT nvidia-smi sees a GPU: {env['nvidia_smi_gpu']}\n"
                "  Fix: reinstall the CUDA build of torch:\n"
                "    pip uninstall -y torch\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
                "  (check the CUDA version in the top-right of `nvidia-smi`; if it\n"
                "   shows < 12.6 use cu121 or cu124 instead)\n"
            )
        else:
            msg += (
                "  and nvidia-smi found no NVIDIA GPU/driver on this machine.\n"
                "  Use `--precision int8` (CPU) or the GGUF path instead of int4.\n"
            )
    else:
        msg = (
            f"torch has a CUDA build (cuda {env['torch_cuda_build']}) but no GPU is "
            "visible.\n  Check the NVIDIA driver is installed (`nvidia-smi`) and, in "
            "WSL, that GPU passthrough is enabled.\n"
        )

    if require:
        raise SystemExit("[env] CUDA REQUIRED but unusable:\n" + msg)
    print("[env] WARNING: running on CPU.\n" + msg)
    return env


def _load(model_id: str, precision: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if precision == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4, the recommended 4-bit type
            bnb_4bit_use_double_quant=True,      # quantize the quant constants too
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto"
        )
    elif precision == "int8":
        # CPU-friendly: PyTorch dynamic int8 quantization.
        # Two lessons learned from the first (naive) attempt on this model:
        #   * per-TENSOR weight scales collapse quality on modern LLMs, which
        #     have outlier channels -> use per-CHANNEL scales;
        #   * quantizing lm_head (weight-tied to the embeddings) corrupts the
        #     output logits directly -> keep it in fp32 and quantize only the
        #     transformer body.
        import warnings
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch.ao deprecation chatter
            from torch.ao.quantization import (
                per_channel_dynamic_qconfig, quantize_dynamic,
            )
            body = getattr(model, "model", None)  # transformer w/o lm_head
            if body is not None:
                model.model = quantize_dynamic(
                    body, {torch.nn.Linear: per_channel_dynamic_qconfig},
                    dtype=torch.qint8,
                )
            else:  # architecture without a .model attribute: quantize all
                model = quantize_dynamic(
                    model, {torch.nn.Linear: per_channel_dynamic_qconfig},
                    dtype=torch.qint8,
                )
        gc.collect()  # drop the fp32 weight copies before memory measurement
    else:  # fp16 / bf16 full precision
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()
    return tok, model


def _mem_footprint_mb() -> float:
    """Peak device memory in MB (GPU if available, else process RSS)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    try:
        import resource  # Linux/macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:  # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6


def _mem_current_mb() -> float:
    """CURRENT process RSS in MB (not peak) — steady-state after model load.

    Peak RSS is misleading for load-then-quantize flows (e.g. dynamic int8
    loads fp32 first, so peak captures BOTH copies transiently). Current RSS
    after load reflects what the model actually occupies while serving.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def run(model_id: str, precision: str, out_path: str) -> None:
    # int4 hard-requires CUDA; fp16 warns loudly if it falls back to CPU so
    # the baseline is measured on the SAME device as the quantized run.
    env = cuda_diagnostic(require=(precision == "int4"))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tok, model = _load(model_id, precision)
    results_mem_after_load = round(_mem_current_mb(), 1)
    try:
        device = next(model.parameters()).device
    except StopIteration:  # fully dynamically-quantized modules expose no params
        device = torch.device("cpu")

    results = {"model": model_id, "precision": precision, "env": env,
               "samples": []}
    total_gen_tokens = 0
    total_gen_time = 0.0

    for prompt in PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        # return_dict=True -> {"input_ids", "attention_mask"} (BatchEncoding).
        # Newer transformers return this dict by default, so we make it explicit
        # and unpack it with ** into generate() instead of passing positionally.
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
        n_prompt = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        gen_ids = out[0][n_prompt:]
        n_gen = int(gen_ids.shape[0])
        text = tok.decode(gen_ids, skip_special_tokens=True)

        total_gen_tokens += n_gen
        total_gen_time += dt
        results["samples"].append(
            {"prompt": prompt, "output": text, "gen_tokens": n_gen,
             "seconds": round(dt, 3)}
        )
        print(f"  done ({n_gen} tok, {dt:.1f}s): {prompt[:60]}...")

    results["peak_mem_mb"] = round(_mem_footprint_mb(), 1)
    results["mem_after_load_mb"] = results_mem_after_load
    results["tokens_per_sec"] = round(total_gen_tokens / total_gen_time, 2)
    results["device"] = str(device)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{precision}] peak_mem={results['peak_mem_mb']} MB  "
          f"throughput={results['tokens_per_sec']} tok/s  -> {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "*q4_k_m*.gguf"   # ~1.0 GB, the standard 4-bit K-quant


def run_gguf(out_path: str) -> None:
    """Quantized leg for CPU machines: GGUF Q4_K_M via llama-cpp-python.

    Unlike dynamic int8 (per-tensor RTN, no outlier handling — collapses
    quality on modern LLMs), K-quant GGUF uses grouped/block quantization and
    computes IN low precision, so it keeps quality AND runs faster on CPU.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise SystemExit(
            "llama-cpp-python is not installed. On Windows CPU install the "
            "prebuilt wheel:\n"
            "  pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
            "(plain `pip install llama-cpp-python` builds from source and "
            "needs a C++ compiler)."
        )

    env = cuda_diagnostic(require=False)
    print(f"[gguf] loading {GGUF_REPO} :: {GGUF_FILE} (downloads ~1 GB on "
          "first run, then cached)")
    llm = Llama.from_pretrained(
        repo_id=GGUF_REPO, filename=GGUF_FILE,
        n_ctx=1024, verbose=False,
    )
    mem_after_load = round(_mem_current_mb(), 1)

    results = {"model": f"{GGUF_REPO} ({GGUF_FILE})", "precision": "gguf-q4_k_m",
               "env": env, "samples": []}
    total_gen_tokens = 0
    total_gen_time = 0.0

    for prompt in PROMPTS:
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_NEW_TOKENS, temperature=0.0,  # greedy, like the HF leg
        )
        dt = time.perf_counter() - t0

        text = out["choices"][0]["message"]["content"]
        n_gen = int(out["usage"]["completion_tokens"])
        total_gen_tokens += n_gen
        total_gen_time += dt
        results["samples"].append(
            {"prompt": prompt, "output": text, "gen_tokens": n_gen,
             "seconds": round(dt, 3)}
        )
        print(f"  done ({n_gen} tok, {dt:.1f}s): {prompt[:60]}...")

    results["peak_mem_mb"] = round(_mem_footprint_mb(), 1)
    results["mem_after_load_mb"] = mem_after_load
    results["tokens_per_sec"] = round(total_gen_tokens / total_gen_time, 2)
    results["device"] = "cpu"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gguf-q4_k_m] peak_mem={results['peak_mem_mb']} MB  "
          f"after_load={mem_after_load} MB  "
          f"throughput={results['tokens_per_sec']} tok/s  -> {out_path}")


def compare(fp_path: str, q_path: str) -> None:
    fp = json.load(open(fp_path, encoding="utf-8"))
    q = json.load(open(q_path, encoding="utf-8"))

    # Guard: comparing CPU numbers against GPU numbers is meaningless.
    dev_fp = fp.get("device", "?").split(":")[0]
    dev_q = q.get("device", "?").split(":")[0]
    if dev_fp != dev_q:
        raise SystemExit(
            f"REFUSING to compare: {fp_path} was measured on '{dev_fp}' but "
            f"{q_path} on '{dev_q}'. Both runs must use the same device — "
            f"re-run the '{dev_fp if dev_fp == 'cpu' else dev_q}' one after "
            "fixing your CUDA setup (see cuda_diagnostic output / README)."
        )

    print(f"\n[both runs on {dev_fp}"
          + (f", GPU: {fp.get('env', {}).get('gpu_name')}" if dev_fp == "cuda" else "")
          + "]")
    print(f"\n| Metric | {fp['precision']} | {q['precision']} | change |")
    print("|---|---|---|---|")
    mem_change = f"{(1 - q['peak_mem_mb']/fp['peak_mem_mb'])*100:.0f}% smaller"
    spd_change = f"{q['tokens_per_sec']/fp['tokens_per_sec']:.2f}x"
    print(f"| Peak memory (MB) | {fp['peak_mem_mb']} | {q['peak_mem_mb']} | {mem_change} |")
    if "mem_after_load_mb" in fp and "mem_after_load_mb" in q:
        steady = f"{(1 - q['mem_after_load_mb']/fp['mem_after_load_mb'])*100:.0f}% smaller"
        print(f"| Mem after load (MB) | {fp['mem_after_load_mb']} | "
              f"{q['mem_after_load_mb']} | {steady} |")
    print(f"| Throughput (tok/s) | {fp['tokens_per_sec']} | {q['tokens_per_sec']} | {spd_change} |")
    print("\nQualitative (read side by side):")
    for a, b in zip(fp["samples"], q["samples"]):
        print("\nPROMPT:", a["prompt"][:70], "...")
        print(f"  {fp['precision']}:", a["output"][:120].replace("\n", " "))
        print(f"  {q['precision']}:", b["output"][:120].replace("\n", " "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--precision", choices=["fp16", "int4", "int8", "gguf-q4"])
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--compare", nargs=2, metavar=("FP", "QUANT"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    elif args.precision == "gguf-q4":
        run_gguf(args.out)
    elif args.precision:
        run(args.model, args.precision, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()