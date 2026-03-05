from typing import Optional, Iterable, Iterator, overload, BinaryIO
from dataclasses import dataclass
from collections import defaultdict, Counter
from functools import partial
import base64
import json
import multiprocessing
import os
import re
import regex
from tqdm import tqdm


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def get_chunk(input_path: str, desired_num_chunks: int) -> list[str]:
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    return chunks


PRE_TOKENIZE_REGEX = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


@dataclass
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


class BPETokenizer:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, params: BPETokenizerParams) -> None: ...

    def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
        if params:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab = {}
            self.merges = []

    def pre_tokenize(self, text: str) -> Counter[tuple[bytes, ...]]:
        token_freqs = Counter()

        for match in regex.finditer(PRE_TOKENIZE_REGEX, text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in token_bytes)
            token_freqs[byte_tuple] += 1

        return token_freqs

    @staticmethod
    def _pre_tokenize_chunk(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
        split_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = split_pattern.split(text)
        tokenizer = BPETokenizer()
        token_freqs = Counter()
        for chunk in chunks:
            chunk_freqs = tokenizer.pre_tokenize(chunk)
            token_freqs.update(chunk_freqs)
        return token_freqs

    def _count_pair_freqs(self, token_freqs: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], int]:
        pair_freqs = defaultdict(int)
        for word, freq in token_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _most_frequent_pair(self, pair_freqs: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        return max(pair_freqs, key=lambda x: (pair_freqs[x], x))

    def _apply_merge(self, token_freqs: Counter[tuple[bytes]], left: bytes, right: bytes) -> Counter[tuple[bytes]]:
        new_freqs = Counter()
        merged = left + right
        for word, freq in token_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] += freq
        return new_freqs

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        import time

        print(f"[Train] input={input_path}, target_vocab_size={vocab_size}, special_tokens={special_tokens}")
        t_start = time.time()

        self.vocab = {}
        self.merges = []

        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        next_id = offset + 256
        print(f"[Train] Initial vocab: {len(self.vocab)} ({len(special_tokens)} special + 256 byte)")

        num_workers = multiprocessing.cpu_count()
        t_chunk = time.time()
        chunks = get_chunk(input_path, num_workers)
        print(f"[Train] Chunked into {len(chunks)} parts ({time.time() - t_chunk:.2f}s)")

        t_pretok = time.time()
        pre_tokenize_fn = partial(BPETokenizer._pre_tokenize_chunk, special_tokens=special_tokens)

        with multiprocessing.Pool() as pool:
            results = list(pool.imap(pre_tokenize_fn, chunks))

        token_freqs = Counter()
        for counter in results:
            token_freqs.update(counter)
        print(f"[Train] Pre-tokenized {len(token_freqs)} unique sequences ({time.time() - t_pretok:.2f}s)")

        num_merges = vocab_size - len(self.vocab)
        print(f"[Train] Merging ({num_merges} merges)...")
        t_merge = time.time()
        with tqdm(total=num_merges, desc="Training BPE") as pbar:
            while len(self.vocab) < vocab_size:
                pair_freqs = self._count_pair_freqs(token_freqs)
                if not pair_freqs:
                    print(f"[Train] No pairs left, stopped at vocab_size={len(self.vocab)}")
                    break
                left, right = self._most_frequent_pair(pair_freqs)
                token_freqs = self._apply_merge(token_freqs, left, right)
                self.vocab[next_id] = left + right
                self.merges.append((left, right))
                next_id += 1
                pbar.update(1)

        print(f"[Train] Merging done ({time.time() - t_merge:.2f}s)")
        print(f"[Train] Complete in {time.time() - t_start:.2f}s — vocab_size={len(self.vocab)}, merges={len(self.merges)}")
        return self.vocab, self.merges

    def save(self, vocab_filepath: str, merges_filepath: str) -> None:
        vocab_data = {v.decode("latin1"): k for k, v in self.vocab.items()}
        with open(vocab_filepath, "w") as vf:
            json.dump(vocab_data, vf, ensure_ascii=False)

        merges_data = [
            [base64.b64encode(a).decode("ascii"), base64.b64encode(b).decode("ascii")]
            for a, b in self.merges
        ]
        with open(merges_filepath, "w") as mf:
            json.dump(merges_data, mf)

    def get_params(self) -> BPETokenizerParams:
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.id_to_bytes = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {v: k for k, v in self.id_to_bytes.items()}

        self.special_tokens_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self.special_tokens_set = set(self.special_tokens_bytes)

        for token in self.special_tokens_bytes:
            if token not in self.bytes_to_id:
                new_id = len(self.id_to_bytes)
                self.id_to_bytes[new_id] = token
                self.bytes_to_id[token] = new_id

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, "r") as vf:
            vocab_data = json.load(vf)
            vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

        merges = []
        with open(merges_filepath, "r") as mf:
            merges_data = json.load(mf)
            for a_b64, b_b64 in merges_data:
                merges.append((base64.b64decode(a_b64), base64.b64decode(b_b64)))

        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str) -> list[str]:
        return regex.findall(PRE_TOKENIZE_REGEX, text)

    def _apply_merges(self, token: bytes) -> list[bytes]:
        word = [bytes([b]) for b in token]

        while True:
            current_pairs = set((word[i], word[i + 1]) for i in range(len(word) - 1))
            mergeable_pairs = [(self.merge_ranks[p], p) for p in current_pairs if p in self.merge_ranks]
            if not mergeable_pairs:
                break

            _, best_pair = min(mergeable_pairs)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text: str) -> list[int]:
        result = []
        special_pattern = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
        split_pattern = re.compile(f"({special_pattern})") if special_pattern else None

        segments = re.split(split_pattern, text) if split_pattern else [text]

        for segment in tqdm(segments, desc="Encoding segments"):
            if segment == "":
                continue
            segment_bytes = segment.encode("utf-8")
            if segment_bytes in self.special_tokens_set:
                result.append(self.bytes_to_id[segment_bytes])
            else:
                for token in self._pre_tokenize(segment):
                    for merged in self._apply_merges(token.encode("utf-8")):
                        result.append(self.bytes_to_id[merged])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.id_to_bytes[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Train or encode with a BPE tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a BPE tokenizer")
    train_parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    train_parser.add_argument("--vocab-size", type=int, required=True, help="Target vocabulary size")
    train_parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens")
    train_parser.add_argument("--vocab-output", type=str, default="vocab.json", help="Output path for vocab file")
    train_parser.add_argument("--merges-output", type=str, default="merges.txt", help="Output path for merges file")

    encode_parser = subparsers.add_parser("encode", help="Encode a text file using a trained BPE tokenizer")
    encode_parser.add_argument("--input", type=str, required=True, help="Path to input text file to encode")
    encode_parser.add_argument("--vocab", type=str, required=True, help="Path to vocab JSON file")
    encode_parser.add_argument("--merges", type=str, required=True, help="Path to merges file")
    encode_parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens")
    encode_parser.add_argument("--output", type=str, required=True, help="Output .bin file for token IDs (loadable via np.memmap)")

    args = parser.parse_args()

    if args.command == "train":
        tokenizer = BPETokenizer()
        tokenizer.train(args.input, args.vocab_size, args.special_tokens)
        tokenizer.save(args.vocab_output, args.merges_output)
        print(f"[Train] Saved vocab ({len(tokenizer.vocab)} tokens) -> {args.vocab_output}")
        print(f"[Train] Saved merges ({len(tokenizer.merges)} merges) -> {args.merges_output}")

    elif args.command == "encode":
        import numpy as np

        tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
        print(f"[Encode] Loaded tokenizer (vocab={args.vocab}, merges={args.merges})")

        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[Encode] Read {len(text)} characters from {args.input}")

        t_start = time.time()
        token_ids = tokenizer.encode(text)
        print(f"[Encode] Encoded {len(token_ids)} tokens ({time.time() - t_start:.2f}s)")

        token_ids_np = np.array(token_ids, dtype=np.uint32)
        token_ids_np.tofile(args.output)
        print(f"[Encode] Saved uint32 array -> {args.output}")
        print(f"[Encode] Load with: np.memmap('{args.output}', dtype=np.uint32, mode='r')")
