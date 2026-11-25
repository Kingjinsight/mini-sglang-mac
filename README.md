# mini-sglang-mac

A native Apple Silicon LLM serving engine. It runs on [MLX](https://github.com/ml-explore/mlx) and Metal, and is adapted from [mini-sglang](https://github.com/sgl-project/mini-sglang).

I built this to understand how a modern inference engine works by making one run well on my Mac. I kept mini-sglang's scheduler, radix prefix cache, chunked prefill, overlap scheduling and OpenAI-compatible server, replaced its CUDA engine with an MLX one, and wrote a Metal paged-attention kernel for decoding. Everything CUDA-only (FlashInfer, CUDA graphs, tensor parallelism, MoE kernels) is gone, which took the package from about 10k lines of Python, CUDA and C++ to 4.4k lines of Python.

## Results

All numbers are measured on a MacBook with an M5 Pro (48 GB), serving Qwen3-0.6B in BF16.

**Decode throughput and time per output token**, compared with my first MLX port (short prompts, 256 output tokens per request):

| Batch size | First port | Now | TPOT before | TPOT now |
|---:|---:|---:|---:|---:|
| 1 | 165 tok/s | 166 tok/s | 6.1 ms | 6.0 ms |
| 8 | 832 tok/s | 1,155 tok/s | 9.6 ms | 6.9 ms |
| 32 | 1,069 tok/s | 3,597 tok/s | 29.9 ms | 8.9 ms |
| 128 | 1,521 tok/s | 7,206 tok/s | 84.2 ms | 17.8 ms |

At batch size 1 the time goes into reading the 1.2 GB of weights, so there is nothing to gain. The larger the batch, the more of each step is spent reading the KV cache, which is what the new attention kernel speeds up.

## What I did

- **MLX engine.** Llama, Qwen2, Qwen3 and Mistral models (Mistral is untested) in BF16 or MLX-quantized form, a paged KV cache, and a sampler, all driven by the existing scheduler.
- **Metal paged decode kernel.** It reads keys and values in place through a GPU copy of the page table, loads each KV row once per GQA group, and splits long contexts across threadgroups when the batch is small. It is 4–5× faster than gathering the KV cache and calling MLX's attention, and reaches 249 GB/s.
- **MLX pitfalls.** Writes into the KV pool only happen in place when no other step holds the old array, so each step waits for the previous one before it is launched. MLX's buffer cache is unbounded, so varying-size activations grew the process to 35 GB until I capped it. Every layer passes the whole KV pool to the kernel, which made MLX cut a command buffer per layer and block `async_eval`; raising `MLX_MAX_MB_PER_BUFFER` fixed overlap scheduling.
- **Scheduler.** Admission now reserves only part of each request's output budget and preempts the newest requests when the next decode step does not fit. I also fixed two overlap-scheduling bugs inherited from upstream: an extra token after EOS, and length-limited requests finishing one step early.

## Quick start

Requires an Apple Silicon Mac, macOS 14 or later, and Python 3.10+.

```bash
git clone https://github.com/Kingjinsight/mini-sglang-mac.git
cd mini-sglang-mac && pip install -e .
```

Serve an OpenAI-compatible API on port 1919:

```bash
python -m macsgl --model Qwen/Qwen3-0.6B
```

Chat in the terminal:

```bash
python -m macsgl --model mlx-community/Llama-3.2-1B-Instruct-4bit --shell
```

Or call it from Python:

```python
from macsgl.core import SamplingParams
from macsgl.llm import LLM

llm = LLM("Qwen/Qwen3-0.6B")
print(llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))[0]["text"])
```

The GPU shares RAM with the rest of the system, so the KV cache takes half of the currently available memory by default. Change it with `--memory-ratio` or set an exact size with `--num-pages`.
