"""Tatoeba English–French pairs compiled by manythings.org.

The file is tab-separated: English, French, then a CC BY 2.0 attribution.
Per-sentence credits stay in that third column; training only reads the sentences.
"""

from __future__ import annotations

import random
import re
import urllib.request
import zipfile
from collections import Counter

import torch

from attention.paths import CORPUS, DATA_URL, ZIP_PATH

_UNICODE_SPACES = re.compile(r"[\u00a0\u202f\u2007\u2009\u200b]+")
_PUNCT = re.compile(r"([?.!,;:\"()\[\]/\\])")
# Drop symbols we do not treat as tokens, after hyphens and ellipses have been split off.
_DROP = re.compile(r"[^\w\s'?.!,;:\"()\[\]/\\]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, keep apostrophes as their own token, and split punctuation."""
    text = text.strip().lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = _UNICODE_SPACES.sub(" ", text)
    text = text.replace("—", " ").replace("–", " ").replace("…", " ").replace("-", " ")
    text = _DROP.sub(" ", text)
    text = _PUNCT.sub(r" \1 ", text)
    text = text.replace("'", " ' ")
    return [token for token in text.split() if token]


def detokenize(tokens: list[str]) -> str:
    """Join model tokens into a readable sentence."""
    text = " ".join(tokens)
    text = text.replace(" ' ", "'")
    text = re.sub(r"\s+([.,])", r"\1", text)
    text = re.sub(r"\s+([?!:;])", r" \1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


class Vocabulary:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    def __init__(self, itos: list[str] | None = None) -> None:
        if itos is None:
            itos = [self.PAD, self.BOS, self.EOS, self.UNK]
        if tuple(itos[:4]) != (self.PAD, self.BOS, self.EOS, self.UNK):
            raise ValueError("vocabulary must begin with <pad>, <bos>, <eos>, <unk>")
        self.itos = list(itos)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, sentences: list[list[str]], max_size: int) -> "Vocabulary":
        counts = Counter(token for sentence in sentences for token in sentence)
        vocab = cls()
        for token, _count in counts.most_common():
            if len(vocab) >= max_size:
                break
            if token in vocab.stoi or not token:
                continue
            vocab.stoi[token] = len(vocab.itos)
            vocab.itos.append(token)
        return vocab

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.unk_id
        index = self.stoi
        return [index.get(token, unk) for token in tokens]


def ensure_corpus() -> None:
    """Download the manythings.org French–English zip when fra.txt is absent."""
    if CORPUS.exists():
        return
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATA_URL}", flush=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "attention-demo/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        ZIP_PATH.write_bytes(response.read())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(CORPUS.parent)
    if not CORPUS.exists():
        raise FileNotFoundError(f"extracted archive did not contain {CORPUS.name}")


def read_corpus(path=CORPUS) -> list[tuple[list[str], list[str]]]:
    pairs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            english = tokenize(parts[0])
            french = tokenize(parts[1])
            if english and french:
                pairs.append((english, french))
    return pairs


def normalize_pairs(
    raw_pairs: list[tuple[list[str], list[str]]],
    max_len: int,
    limit: int | None,
    seed: int,
    keep: list[list[str]] | None = None,
) -> tuple[list[tuple[list[str], list[str]]], list[tuple[list[str], list[str]]]]:
    """Keep one short French sentence per English sentence, then hold out a validation slice.

    Alternate Tatoeba translations are common ("Go." is "Va !", "Marche.", "Bouge !", ...).
    The shortest French side is kept so the small model has a single target.
    """
    best: dict[tuple[str, ...], list[str]] = {}
    order: list[tuple[str, ...]] = []
    for english, french in raw_pairs:
        if len(english) > max_len or len(french) > max_len:
            continue
        key = tuple(english)
        previous = best.get(key)
        if previous is None:
            best[key] = list(french)
            order.append(key)
        elif len(french) < len(previous):
            best[key] = list(french)
    pairs = [(list(key), best[key]) for key in order]
    pairs.sort(key=lambda item: (len(item[0]) + len(item[1]), item[0], item[1]))
    if limit is not None:
        pairs = pairs[:limit]
    selected = {tuple(english) for english, _ in pairs}
    for tokens in keep or []:
        key = tuple(tokens)
        if key in best and key not in selected:
            pairs.append((list(key), best[key]))
            selected.add(key)
    random.Random(seed).shuffle(pairs)
    if len(pairs) < 2:
        return pairs, []
    n_val = min(500, max(1, len(pairs) // 20))
    n_val = min(n_val, len(pairs) - 1)
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]
    if keep:
        wanted = {tuple(tokens) for tokens in keep}
        train_pairs.extend(pair for pair in val_pairs if tuple(pair[0]) in wanted)
        val_pairs = [pair for pair in val_pairs if tuple(pair[0]) not in wanted]
    return train_pairs, val_pairs


def unk_rate(sentences: list[list[str]], vocab: Vocabulary) -> float:
    total = 0
    unknown = 0
    for sentence in sentences:
        for token in sentence:
            total += 1
            if token not in vocab.stoi:
                unknown += 1
    return unknown / max(1, total)


def pad_sequences(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(sequence) for sequence in sequences)
    out = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if sequence:
            out[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return out


def collate(
    batch: list[tuple[list[str], list[str]]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = pad_sequences([src_vocab.encode(english) for english, _ in batch], src_vocab.pad_id)
    target = [tgt_vocab.encode(french) for _, french in batch]
    target_in = pad_sequences([[tgt_vocab.bos_id, *tokens] for tokens in target], tgt_vocab.pad_id)
    target_out = pad_sequences([tokens + [tgt_vocab.eos_id] for tokens in target], tgt_vocab.pad_id)
    return source, target_in, target_out


def make_batches(
    pairs: list[tuple[list[str], list[str]]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[tuple[list[str], list[str]]]]:
    ordered = sorted(pairs, key=lambda item: (len(item[0]), len(item[1]), item[0]))
    chunks = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    if shuffle:
        random.Random(seed + epoch).shuffle(chunks)
    return chunks
