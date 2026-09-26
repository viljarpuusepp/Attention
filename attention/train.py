"""Train the encoder–decoder on short Tatoeba sentences.

The optimiser is Adam (beta1 0.9, beta2 0.98, eps 1e-9) with the paper's
warmup schedule. Label smoothing is 0.1.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time

import torch

from attention.config import Config
from attention.data import (
    Vocabulary,
    collate,
    ensure_corpus,
    make_batches,
    normalize_pairs,
    read_corpus,
    tokenize,
    unk_rate,
)
from attention.model import (
    LabelSmoothing,
    build_model,
    causal_padding_mask,
    padding_mask,
    parameter_count,
)
from attention.paths import BEST_CHECKPOINT, DATA_DATE, DATA_LICENSE, DATA_URL, LAST_CHECKPOINT
from attention.schedule import paper_lr
from attention.translate import translate

SAMPLES = (
    "Go.",
    "Hello.",
    "Thank you.",
    "How are you?",
    "I love you.",
    "Good morning.",
    "He is a doctor.",
    "This is a book.",
)


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


class Noam:
    """Set Adam's learning rate from the paper's formula, then take one step."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup: int, factor: float) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        rate = paper_lr(self.step_num, self.d_model, self.warmup, self.factor)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()
        return rate


def parse_args(argv: list[str] | None = None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train a small English-to-French transformer.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--limit", type=int, default=defaults.limit)
    parser.add_argument("--max-len", type=int, default=defaults.max_len)
    parser.add_argument("--max-vocab", type=int, default=defaults.max_vocab)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--layers", type=int, default=defaults.n_layers)
    parser.add_argument("--heads", type=int, default=defaults.n_heads)
    parser.add_argument("--d-ff", type=int, default=defaults.d_ff)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--lr-factor", type=float, default=defaults.lr_factor)
    parser.add_argument("--smoothing", type=float, default=defaults.smoothing)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--beam-size", type=int, default=defaults.beam_size)
    args = parser.parse_args(argv)
    return Config(
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        limit=args.limit,
        max_vocab=args.max_vocab,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup=args.warmup,
        lr_factor=args.lr_factor,
        smoothing=args.smoothing,
        seed=args.seed,
        beam_size=args.beam_size,
    )


def run_epoch(model, batches, criterion, scheduler, src_vocab, tgt_vocab, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    window = []
    learning_rate = 0.0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for index, batch in enumerate(batches, start=1):
            source, target_in, target_out = collate(batch, src_vocab, tgt_vocab)
            source = source.to(device)
            target_in = target_in.to(device)
            target_out = target_out.to(device)
            logits, _ = model(
                source,
                target_in,
                padding_mask(source, src_vocab.pad_id),
                causal_padding_mask(target_in, tgt_vocab.pad_id),
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
            if not math.isfinite(loss.item()):
                raise RuntimeError("loss became non-finite; stop before saving this step")
            token_count = int(target_out.ne(tgt_vocab.pad_id).sum().item())
            if train:
                scheduler.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                learning_rate = scheduler.step()
            prediction = logits.argmax(dim=-1)
            mask = target_out.ne(tgt_vocab.pad_id)
            correct += int((prediction.eq(target_out) & mask).sum().item())
            total_loss += loss.item() * token_count
            total_tokens += token_count
            if train:
                window.append(loss.item())
                if index % 50 == 0:
                    recent = sum(window) / len(window)
                    print(
                        f"  step {scheduler.step_num:5d}  loss {recent:.3f}  lr {learning_rate:.6f}",
                        flush=True,
                    )
                    window.clear()
    return total_loss / max(1, total_tokens), correct / max(1, total_tokens), learning_rate


def save_checkpoint(path, model, config, src_vocab, tgt_vocab, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "src_itos": src_vocab.itos,
            "tgt_itos": tgt_vocab.itos,
            "data_url": DATA_URL,
            "data_date": DATA_DATE,
            "data_license": DATA_LICENSE,
            "saved_at": time.strftime("%Y-%m-%d"),
            **stats,
        },
        path,
    )


def show_samples(model, src_vocab, tgt_vocab, config) -> None:
    for text in SAMPLES:
        try:
            result = translate(
                model,
                src_vocab,
                tgt_vocab,
                text,
                beam_size=config.beam_size,
                max_source=config.max_len,
                max_decode=max(config.max_len * 3, 24),
                alpha=config.length_alpha,
            )
        except ValueError as exc:
            print(f"  {text} -> ({exc})", flush=True)
            continue
        print(f"  {text} -> {result['translation']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    _utf8_stdio()
    config = parse_args(argv)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_corpus()
    print("reading data/fra.txt", flush=True)
    print(f"source {DATA_URL} ({DATA_DATE}, {DATA_LICENSE})", flush=True)
    raw = read_corpus()
    train_pairs, val_pairs = normalize_pairs(
        raw,
        config.max_len,
        config.limit,
        config.seed,
        keep=[tokenize(text) for text in SAMPLES],
    )
    src_vocab = Vocabulary.build([english for english, _ in train_pairs], config.max_vocab)
    tgt_vocab = Vocabulary.build([french for _, french in train_pairs], config.max_vocab)
    print(
        f"{len(raw)} raw pairs -> {len(train_pairs)} train / {len(val_pairs)} val, "
        f"max {config.max_len} tokens",
        flush=True,
    )
    print(
        f"vocab English {len(src_vocab)} / French {len(tgt_vocab)} "
        f"(val unk {unk_rate([e for e, _ in val_pairs], src_vocab):.1%} en, "
        f"{unk_rate([f for _, f in val_pairs], tgt_vocab):.1%} fr)",
        flush=True,
    )

    model = build_model(config, len(src_vocab), len(tgt_vocab)).to(device)
    print(
        f"{parameter_count(model)} parameters on {device}, {torch.get_num_threads()} threads",
        flush=True,
    )
    criterion = LabelSmoothing(len(tgt_vocab), tgt_vocab.pad_id, config.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = Noam(optimizer, config.d_model, config.warmup, config.lr_factor)
    best_val = math.inf
    step = 0

    try:
        for epoch in range(1, config.epochs + 1):
            started = time.perf_counter()
            train_batches = make_batches(train_pairs, config.batch_size, True, config.seed, epoch)
            val_batches = make_batches(val_pairs, config.batch_size, False, config.seed, epoch)
            train_loss, train_acc, learning_rate = run_epoch(
                model, train_batches, criterion, scheduler, src_vocab, tgt_vocab, device, True
            )
            val_loss, val_acc, _ = run_epoch(
                model, val_batches, criterion, scheduler, src_vocab, tgt_vocab, device, False
            )
            elapsed = time.perf_counter() - started
            step = scheduler.step_num
            print(
                f"epoch {epoch}/{config.epochs}  train {train_loss:.3f} acc {train_acc:.3f}  "
                f"val {val_loss:.3f} acc {val_acc:.3f}  lr {learning_rate:.6f}  {elapsed:.0f}s",
                flush=True,
            )
            stats = {
                "epoch": epoch,
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
            }
            save_checkpoint(LAST_CHECKPOINT, model, config, src_vocab, tgt_vocab, stats)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(BEST_CHECKPOINT, model, config, src_vocab, tgt_vocab, stats)
                print(f"  saved {BEST_CHECKPOINT}", flush=True)
            show_samples(model, src_vocab, tgt_vocab, config)
    except KeyboardInterrupt:
        print("interrupted; the last finished epoch is in checkpoints/last.pt", flush=True)
        return

    print(f"best val loss {best_val:.3f} at step {step}", flush=True)
    print("open the demo with: python -m attention.server", flush=True)


if __name__ == "__main__":
    main()
