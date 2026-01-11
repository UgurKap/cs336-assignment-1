import regex as re
from .pretokenization import find_chunk_boundaries
from multiprocessing import Pool
from collections import Counter

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


def get_chunks(input_path: str, num_processes: int, special_tokens: list[str]) -> list[str]:
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks += re.split(
                "|".join(re.escape(s_token) for s_token in special_tokens),
                f.read(end - start).decode("utf-8", errors="ignore"),
            )
    return chunks


def parallel_pretokenize(chunk) -> dict[tuple[int], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    frequency_table_bytes = dict()
    for match in re.finditer(PAT, chunk):
        catch = (match.captures()[0]).encode("utf-8")
        if catch == b"<|endoftext|>":
            continue
        frequency_table_bytes[catch] = frequency_table_bytes.get(catch, 0) + 1
    return frequency_table_bytes


def count_frequencies(
    input_path: str, num_processes: int, byte_to_id: dict[int, int], special_tokens: list[str]
) -> dict[list[int], int]:
    chunks = get_chunks(input_path=input_path, num_processes=num_processes, special_tokens=special_tokens)
    with Pool(num_processes) as pool:
        freq_dicts = pool.map(parallel_pretokenize, chunks)
    frequency_table_bytes = Counter()
    for d in freq_dicts:
        frequency_table_bytes.update(d)
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

    pair_dict = {}
    for token_sequence in frequency_table_token_ids.keys():
        for i in range(len(token_sequence) - 1):
            pair_dict[(token_sequence[i], token_sequence[i + 1])] = (
                pair_dict.get((token_sequence[i], token_sequence[i + 1]), 0) + frequency_table_token_ids[token_sequence]
            )

    merge_num = 0
    while (len(vocab) < vocab_size) and (merge_num < max_merge_num):
        new_id = len(vocab)
        merged_pair = max(pair_dict.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
        merges.append(merged_pair)
        vocab[new_id] = vocab[merged_pair[0]] + vocab[merged_pair[1]]
        byte_to_id[vocab[new_id]] = new_id
        keys_to_replace = []

        for token_sequence in frequency_table_token_ids.keys():
            new_tuple = merge_pairs(token_sequence, merged_pair, new_id)
            if len(token_sequence) != len(new_tuple):
                keys_to_replace.append((token_sequence, new_tuple))

        for old_key, new_key in keys_to_replace:
            num_changes = frequency_table_token_ids.pop(old_key)
            frequency_table_token_ids[new_key] = num_changes
            for i in range(len(old_key) - 1):
                pair_dict[(old_key[i], old_key[i + 1])] -= num_changes
            for i in range(len(new_key) - 1):
                pair_dict[(new_key[i], new_key[i + 1])] = pair_dict.get((new_key[i], new_key[i + 1]), 0) + num_changes

        merge_num += 1

    return vocab, [(vocab[a], vocab[b]) for (a, b) in merges]
