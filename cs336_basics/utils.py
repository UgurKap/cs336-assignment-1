from pathlib import Path
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cs336_basics.bpe_tokenizer import Tokenizer


def sample_from_txt(input_file_path: str, num_samples: int, eot_token: str = "<|endoftext|>") -> list[str]:
    p = Path(input_file_path).absolute()
    samples = []
    sample_limit = min(100, num_samples)
    with open(p, "rb") as f:
        sample = []
        for line in f.readlines():
            cur_line = line.decode("utf-8")
            if eot_token in cur_line:
                prev_sample, next_sample = cur_line.split(eot_token)
                sample.append(prev_sample)
                samples.append("".join(sample))
                if len(samples) == sample_limit:
                    break
                sample = [next_sample]
            else:
                sample.append(cur_line)

    random.shuffle(samples)
    return samples[:num_samples]


def benchmark_tokenizer(tokenizer: "Tokenizer", samples: list[str], special_token: str, label: str):
    begin = time.time()
    input_text = special_token.join(samples)
    encoded_input = tokenizer.encode(input_text)
    end = time.time()

    num_special = len(samples) - 1
    special_len = num_special * len(special_token.encode("utf-8"))
    in_bytes = len(input_text.encode("utf-8")) - special_len
    out_tokens = len(encoded_input) - num_special
    comp_ratio = in_bytes / out_tokens
    time_elapsed = end - begin
    throughput = out_tokens / time_elapsed

    print(f"=== {label} ===")
    print(f"Input Bytes: {in_bytes}, Encoded length: {out_tokens}, Compression Ratio: {comp_ratio}")
    print(
        f"Processing time: {time_elapsed} seconds, Throughput: {throughput} token/s (Processing {in_bytes / time_elapsed} bytes/s)"
    )
