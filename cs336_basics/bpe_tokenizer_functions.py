import regex as re
from multiprocessing import Pool
from collections import Counter
import cloudpickle
import sys

try:
    from .pretokenization import find_chunk_boundaries
except ImportError:
    from pretokenization import find_chunk_boundaries

MAX_MERGE_NUM = 20000
MAX_VOCAB_SIZE = 1000


def init_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    special_tokens = [token.encode("utf-8") for token in special_tokens]
    vocabulary_tokens = [bytes([i]) for i in range(256)]
    vocab = {i: t for i, t in enumerate(special_tokens + vocabulary_tokens)}
    return vocab


def merge_pairs(sequence: tuple[int], old_ids: tuple[int], new_id: int) -> tuple[int]:
    new_sequence = list(sequence)
    num_inserts = 0
    i = 0
    while i < (len(sequence) - 1):
        if (sequence[i] == old_ids[0]) and (sequence[i + 1] == old_ids[1]):
            new_sequence[(i - num_inserts) : (i - num_inserts + 2)] = [new_id]
            num_inserts += 1
            i += 1
        i += 1
    return tuple(new_sequence)

def parallel_pretokenize_chunk(args:tuple[str, int, int, list[str]]) -> dict[tuple[int], int]:
    input_path, start, end, special_tokens = args
    print(f"Worker starting chunk {start}-{end}", flush=True)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    frequency_table_bytes = dict()

    with open(input_path, "rb") as f:
        f.seek(start)
        text_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        print(f"Chunk {start}-{end}: splitting on special tokens", file=sys.stderr, flush=True)
        pieces = re.split("|".join(re.escape(s_token) for s_token in special_tokens), text_chunk)

        print(f"Chunk {start}-{end}: processing {len(pieces)} pieces", file=sys.stderr, flush=True)
        for piece_idx, piece in enumerate(pieces):
            if piece_idx % 10000 == 0:
                print(f"Chunk {start}-{end}: piece {piece_idx}/{len(pieces)}", file=sys.stderr, flush=True)
            for match in re.finditer(PAT, piece):
                catch = (match.group(0)).encode("utf-8")
                frequency_table_bytes[catch] = frequency_table_bytes.get(catch, 0) + 1
    
    print(f"Worker finished chunk {start}-{end}", flush=True)
    return frequency_table_bytes


def count_frequencies(
    input_path: str, num_processes: int, byte_to_id: dict[int, int], special_tokens: list[str]
) -> dict[list[int], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(num_processes) as pool:
        async_result = pool.map_async(parallel_pretokenize_chunk, chunk_args)
        freq_dicts = async_result.get(timeout=300)

    frequency_table_bytes = Counter()
    for d in freq_dicts:
        frequency_table_bytes.update(d)
    del freq_dicts
    return {tuple(byte_to_id[bytes([b])] for b in k): v for k, v in frequency_table_bytes.items()}


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4, max_merge_num: int = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if max_merge_num is None:
        max_merge_num = vocab_size
    vocab = init_vocabulary(special_tokens=special_tokens)
    byte_to_id = {v: k for k, v in vocab.items()}
    merges = list()

    frequency_table_token_ids = count_frequencies(
        input_path=input_path, num_processes=num_processes, byte_to_id=byte_to_id, special_tokens=special_tokens
    )

    seq_id_to_sequence = {}
    seq_id_to_freq = {}
    pair_to_seq_ids = {}
    pair_dict = {}

    for seq_id, token_sequence in enumerate(frequency_table_token_ids.keys()):
        seq_id_to_sequence[seq_id] = token_sequence
        seq_id_to_freq[seq_id] = frequency_table_token_ids[token_sequence]
        for i in range(len(token_sequence) - 1):
            pair = (token_sequence[i], token_sequence[i + 1])
            pair_dict[pair] = pair_dict.get(pair, 0) + frequency_table_token_ids[token_sequence]
            if pair not in pair_to_seq_ids:
                pair_to_seq_ids[pair] = set()
            pair_to_seq_ids[pair].add(seq_id)

    del frequency_table_token_ids

    merge_num = 0
    while (len(vocab) < vocab_size) and (merge_num < max_merge_num):
        new_id = len(vocab)
        merged_pair = max(pair_dict.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
        merges.append(merged_pair)
        vocab[new_id] = vocab[merged_pair[0]] + vocab[merged_pair[1]]
        byte_to_id[vocab[new_id]] = new_id

        if merged_pair in pair_to_seq_ids:
            affected_seq_ids = list(pair_to_seq_ids[merged_pair])
            for seq_id in affected_seq_ids:
                old_sequence = seq_id_to_sequence[seq_id]
                new_sequence = merge_pairs(old_sequence, merged_pair, new_id)
                if len(old_sequence) != len(new_sequence):
                    freq = seq_id_to_freq[seq_id]

                    for i in range(len(old_sequence) - 1):
                        old_pair = (old_sequence[i], old_sequence[i + 1])
                        pair_dict[old_pair] -= freq
                        if old_pair in pair_to_seq_ids:
                            pair_to_seq_ids[old_pair].discard(seq_id)
                            if len(pair_to_seq_ids[old_pair]) == 0:
                                del pair_to_seq_ids[old_pair]

                    seq_id_to_sequence[seq_id] = new_sequence

                    for i in range(len(new_sequence) - 1):
                        new_pair = (new_sequence[i], new_sequence[i + 1])
                        pair_dict[new_pair] = pair_dict.get(new_pair, 0) + freq
                        if new_pair not in pair_to_seq_ids:
                            pair_to_seq_ids[new_pair] = set()
                        pair_to_seq_ids[new_pair].add(seq_id)

        merge_num += 1

    return vocab, [(vocab[a], vocab[b]) for (a, b) in merges]

def main():
    vocab, merges = train_bpe(input_path="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", vocab_size=10_000, special_tokens=["<|endoftext|>"], num_processes=6)
    print("Training complete, saving the vocabulary and list of merges to disk now")
    with open("vocab.pickle", "wb") as f:
        cloudpickle.dump(vocab, f)
    with open("merges.pickle", "wb") as f:
        cloudpickle.dump(merges, f)

if __name__ == "__main__":
    main()