from typing import Self
from collections.abc import Iterable, Iterator
import pickle
from pathlib import Path
import regex as re
import numpy as np
from math import inf


try:
    from .utils import sample_from_txt, benchmark_tokenizer
except ImportError:
    from utils import sample_from_txt, benchmark_tokenizer


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}  # For reverse look-up (i.e. byte to token_id)
        self.merges = merges  # This will have pairs like (b' uğu', b'r')
        self.merges_ranks = {
            (self.byte_to_id[t[0]], self.byte_to_id[t[1]]): i for i, t in enumerate(self.merges)
        }  # (token1, token2): rank (merge order)
        if special_tokens is not None:
            self.special_tokens = [
                s_token.encode("utf-8") for s_token in special_tokens
            ]  # We encode the special tokens with utf-8 as well, because I want to ensure dict[int, bytes]
        else:
            self.special_tokens = []

        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)  # Longest special token first
        for s_token in self.special_tokens:
            if s_token not in self.byte_to_id:
                self.byte_to_id[s_token] = len(vocab)
                self.vocab[len(vocab)] = s_token
        self.regex_pattern = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # Capture pattern from GPT-2
        )

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        vocab_path = Path(vocab_filepath).absolute()
        merges_path = Path(merges_filepath).absolute()
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if len(self.special_tokens) > 0:
            pieces = re.split(
                "(" + "|".join(re.escape(s_token.decode("utf-8")) for s_token in self.special_tokens) + ")", text
            )
        else:
            pieces = [text]

        encoded_sequence = list()

        for piece in pieces:
            if (encoded := piece.encode("utf-8")) in self.special_tokens:
                encoded_sequence += [self.byte_to_id[encoded]]
            else:
                for match in re.finditer(self.regex_pattern, piece):
                    catch = (match.group(0)).encode("utf-8")
                    chunk_tokens = [self.byte_to_id[bytes([c])] for c in catch]
                    chunk_pairs = []

                    if len(chunk_tokens) > 1:  # if chunk_pairs is only 1, just use the byte token, no need for merges
                        for i in range(len(chunk_tokens) - 1):
                            pair = (chunk_tokens[i], chunk_tokens[i + 1])
                            rank = self.merges_ranks.get(pair, inf)
                            chunk_pairs.append((pair, rank))

                        # Merge loop
                        while True:
                            min_idx = min(range(len(chunk_pairs)), key=lambda x: chunk_pairs[x][1])
                            pair, rank = chunk_pairs[min_idx]
                            if rank == inf:
                                break

                            left = chunk_pairs[: max(0, min_idx - 1)]  # can be empty, can have elements
                            right = chunk_pairs[min_idx + 2 :]  # can be empty, can have elements

                            # token after the merge
                            new_token = self.byte_to_id.get(self.vocab[pair[0]] + self.vocab[pair[1]], None)

                            # if there is something to the left
                            if min_idx > 0:
                                prev_pair_idx = min_idx - 1
                                new_pair = (chunk_pairs[prev_pair_idx][0][0], new_token)
                                left.append((new_pair, self.merges_ranks.get(new_pair, inf)))

                            if min_idx < len(chunk_pairs) - 1:
                                next_pair_idx = min_idx + 1
                                new_pair = (new_token, chunk_pairs[next_pair_idx][0][1])
                                right.insert(0, (new_pair, self.merges_ranks.get(new_pair, inf)))

                            if len(pair_list := left + right) == 0:
                                chunk_pairs = [((new_token, 0), -1)]
                                break
                            else:
                                chunk_pairs = pair_list

                        chunk_tokens = []
                        for pair in chunk_pairs:
                            chunk_tokens.append(pair[0][0])
                        if chunk_pairs[-1][1] != -1:
                            chunk_tokens.append(chunk_pairs[-1][0][1])

                    encoded_sequence.extend(chunk_tokens)

        return encoded_sequence

    def encode_iterable(self, iterable: Iterable[str], chunk_size: int = 1024 * 1024) -> Iterator[int]:
        buffer = ""
        for item in iterable:
            buffer += item
            if len(buffer) >= chunk_size:
                yield from self.encode(buffer)
                buffer = ""

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        comb = []
        for id in ids:
            comb += self.vocab[id]
        return bytes(comb).decode("utf-8", errors="replace")

    def __len__(self):
        return len(self.vocab)

    def merge_pairs(self, sequence: list[int], old_tokens: tuple[bytes]) -> list[int]:
        old_token_ids = (self.byte_to_id[old_tokens[0]], self.byte_to_id[old_tokens[1]])
        new_token_id = self.byte_to_id[old_tokens[0] + old_tokens[1]]
        new_sequence = []
        i = 0
        while i < len(sequence):
            if (i < len(sequence) - 1) and (sequence[i] == old_token_ids[0]) and (sequence[i + 1] == old_token_ids[1]):
                new_sequence.append(new_token_id)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        return new_sequence


def main():
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/owt_vocab.pickle",
        merges_filepath="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/owt_merges.pickle",
        special_tokens=["<|endoftext|>"],
    )
    tiny_tokenizer = Tokenizer.from_files(
        vocab_filepath="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tiny_vocab.pickle",
        merges_filepath="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tiny_merges.pickle",
        special_tokens=["<|endoftext|>"],
    )

    # Benchmarking

    owt_samples = sample_from_txt(
        "/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/owt_valid.txt", num_samples=10
    )
    tiny_samples = sample_from_txt(
        "/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", num_samples=10
    )

    benchmark_tokenizer(owt_tokenizer, owt_samples, "<|endoftext|>", "OWT")
    print()
    benchmark_tokenizer(tiny_tokenizer, tiny_samples, "<|endoftext|>", "TinyStories")
    print()
    benchmark_tokenizer(owt_tokenizer, tiny_samples, "<|endoftext|>", "OWT Tokenizer on TinyStories Dataset")
    print()
    benchmark_tokenizer(tiny_tokenizer, owt_samples, "<|endoftext|>", "TinyStories Tokenizer on OWT Dataset")

    # Process the training and validation sets
    owt_train_out = []
    with open(Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/owt_train.txt").absolute()) as f:
        for _id in owt_tokenizer.encode_iterable(f):
            owt_train_out.append(_id)
    np.save(
        Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/owt_train_tokens.npy").absolute(),
        np.array(owt_train_out, dtype=np.uint16),
    )
    del owt_train_out

    owt_valid_out = []
    with open(Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/owt_valid.txt").absolute()) as f:
        for _id in owt_tokenizer.encode_iterable(f):
            owt_valid_out.append(_id)
    np.save(
        Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/owt_valid_tokens.npy").absolute(),
        np.array(owt_valid_out, dtype=np.uint16),
    )
    del owt_valid_out

    tiny_train_out = []
    with open(
        Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt").absolute()
    ) as f:
        for _id in tiny_tokenizer.encode_iterable(f):
            tiny_train_out.append(_id)
    np.save(
        Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/tiny_train_tokens.npy").absolute(),
        np.array(tiny_train_out, dtype=np.uint16),
    )
    del tiny_train_out

    tiny_valid_out = []
    with open(
        Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt").absolute()
    ) as f:
        for _id in tiny_tokenizer.encode_iterable(f):
            tiny_valid_out.append(_id)
    np.save(
        Path("/home/ugurkap/stanford-cs336-assignments/assignment1-basics/data/tiny_valid_tokens.npy").absolute(),
        np.array(tiny_valid_out, dtype=np.uint16),
    )


if __name__ == "__main__":
    main()
