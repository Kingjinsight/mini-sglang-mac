# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
from random import randint, seed

from macsgl.core import SamplingParams
from macsgl.llm import LLM


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # align the hyperparameters; 400 pages x 256 tokens is ~11 GiB of KV cache for Qwen3-0.6B
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        page_size=256,
        num_page_override=400,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # warm up
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
