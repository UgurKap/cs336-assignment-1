from typing import Self
from collections.abc import Iterable, Iterator
import pickle
from pathlib import Path
import regex as re
from copy import deepcopy

import os, json
from functools import lru_cache
import tiktoken


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}  # For reverse look-up (i.e. byte to token_id)
        self.merges = merges  # This will have pairs like (b' uğu', b'r')
        if special_tokens is not None:
            self.special_tokens = [
                s_token.encode("utf-8") for s_token in special_tokens
            ]  # We encode the special tokens with utf-8 as well, because I want to ensure dict[int, bytes]
        else:
            self.special_tokens = []

        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True) # Longest special token first
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
        pieces = re.split(
            "(" + "|".join(re.escape(s_token.decode("utf-8")) for s_token in self.special_tokens) + ")", text
        )
        encoded_sequence = list()

        for piece in pieces:
            if (encoded := piece.encode("utf-8")) in self.special_tokens:
                encoded_sequence += [self.byte_to_id[encoded]]
            else:
                chunk_tokens = list()
                for match in re.finditer(self.regex_pattern, piece):
                    catch = (match.group(0)).encode("utf-8")
                    chunk_tokens += [self.byte_to_id[bytes([c])] for c in catch]                                
                for merge_pair in self.merges:
                    chunk_tokens = self.merge_pairs(chunk_tokens, merge_pair)
                encoded_sequence.extend(chunk_tokens)

        return encoded_sequence

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            yield from self.encode(it)

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
        new_sequence = deepcopy(sequence)
        num_inserts = 0
        i = 0
        while i < (len(sequence) - 1):
            if (sequence[i] == old_token_ids[0]) and (sequence[i + 1] == old_token_ids[1]):
                new_sequence[(i - num_inserts) : (i - num_inserts + 2)] = [new_token_id]
                num_inserts += 1
                i += 1
            i += 1
        return new_sequence


def main():
    @lru_cache
    def gpt2_bytes_to_unicode() -> dict[int, str]:
        """
        Returns a mapping between every possible byte (an integer from 0 to 255) to a
        printable unicode string character representation. This function is taken
        from the GPT-2 code.

        For example, `chr(0)` is `\x00`, which is an unprintable character:

        >>> chr(0)
        '\x00'
        >>> print(chr(0))

        As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
        The bytes that are visually printable keep their original string representation [1].
        For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
        Note in particular that the space character `chr(32)` becomes `d[32]`, which
        returns 'Ġ'.

        For unprintable characters, the function shifts takes the integer representing
        the Unicode code point of that character (returned by the Python `ord`) function
        and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
        ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
        string representation of the space.

        This function can simplify the BPE implementation and makes it slightly easier to
        manually inspect the generated merges after they're serialized to a file.
        """
        # These 188 integers can used as-is, since they are not whitespace or control characters.
        # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        # now get the representations of the other 68 integers that do need shifting
        # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
        # Get printable representations of the remaining integers 68 integers.
        n = 0
        for b in range(2**8):
            if b not in bs:
                # If this integer isn't in our list of visually-representable
                # charcters, then map it to the next nice character (offset by 256)
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        characters = [chr(n) for n in cs]
        d = dict(zip(bs, characters))
        return d

    def get_tokenizer_from_vocab_merges_path(
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return Tokenizer(vocab, merges, special_tokens)

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tests/fixtures/gpt2_vocab.json", merges_path="/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tests/fixtures/gpt2_merges.txt")
    corpus_path = "/home/ugurkap/stanford-cs336-assignments/assignment1-basics/tests/fixtures/address.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents

    

if __name__ == "__main__":
    main()
