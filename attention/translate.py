"""Beam search with the paper's beam size of 4 and length penalty of 0.6."""

from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F

from attention.config import Config
from attention.data import Vocabulary, detokenize, tokenize
from attention.model import (
    Transformer,
    build_model,
    causal_padding_mask,
    padding_mask,
    parameter_count,
)
from attention.paths import DEFAULT_CHECKPOINT
from attention.schedule import length_penalty


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def load_checkpoint(
    path=DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> tuple[Transformer, Vocabulary, Vocabulary, Config, dict]:
    if device is None:
        device = torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    config = Config.from_dict(blob["config"])
    src_vocab = Vocabulary(blob["src_itos"])
    tgt_vocab = Vocabulary(blob["tgt_itos"])
    model = build_model(config, len(src_vocab), len(tgt_vocab))
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, config, blob


def _name(vocab: Vocabulary, index: int) -> str:
    return vocab.itos[index]


def _scaling_example(model: Transformer, tokens: list[str], token_ids: list[int]) -> dict:
    """Raw embedding, the √d multiplication, and the Euclidean length of all coordinates."""
    scale = math.sqrt(model.d_model)
    choice = 0
    if "love" in tokens:
        choice = tokens.index("love")
    else:
        for index, token in enumerate(tokens):
            if token.isalpha():
                choice = index
                break
    raw = model.src_embed.weight[token_ids[choice]].detach()
    scaled = raw * scale
    shown = 3
    sum_of_squares = float((scaled * scaled).sum())
    return {
        "token": tokens[choice],
        "scale": round(scale, 3),
        "dimensions": int(model.d_model),
        "raw": [round(float(value), 4) for value in raw[:shown]],
        "scaled": [round(float(value), 3) for value in scaled[:shown]],
        "sum_of_squares": round(sum_of_squares, 3),
        "norm": round(float(scaled.norm()), 3),
    }


def _vector_rows(model: Transformer, token_ids: list[int], table: torch.nn.Embedding) -> list[dict]:
    """Scaled embedding, sinusoid, and their sum. Dropout is off in eval mode."""
    ids = torch.tensor([token_ids], dtype=torch.long, device=table.weight.device)
    scaled = table(ids) * math.sqrt(model.d_model)
    position = model.positional.pe[:, : len(token_ids)]
    summed = scaled + position
    rows = []
    for index in range(len(token_ids)):
        rows.append(
            {
                "embed_norm": round(float(scaled[0, index].norm()), 3),
                "position_norm": round(float(position[0, index].norm()), 3),
                "sum_norm": round(float(summed[0, index].norm()), 3),
                "position_dims": [round(float(value), 3) for value in position[0, index, :4]],
                "sum_dims": [round(float(value), 3) for value in summed[0, index, :4]],
            }
        )
    return rows


def _sinusoids(model: Transformer, tokens: list[str]) -> dict:
    """Curves of the positional encoding, sampled from the same formula as the module."""
    d_model = model.d_model
    count = min(48, int(model.positional.pe.size(1)))
    step = 0.25
    total = (count - 1) * 4 + 1
    positions = [index * step for index in range(total)]
    channels = []
    for sin_dim in (0, 16, 32, 64):
        if sin_dim + 1 >= d_model:
            continue
        exponent = sin_dim / d_model
        omega = 10000 ** (-exponent)
        period = (2 * math.pi) / omega
        channels.append(
            {
                "sin_dim": sin_dim,
                "cos_dim": sin_dim + 1,
                "exponent": round(exponent, 4),
                "period": round(period, 2),
                "sin": [round(math.sin(position * omega), 4) for position in positions],
                "cos": [round(math.cos(position * omega), 4) for position in positions],
            }
        )
    fastest = channels[0]
    slowest = channels[-1]
    marks = ", ".join(f"{token} at position {index}" for index, token in enumerate(tokens))
    return {
        "step": step,
        "channels": channels,
        "marks": [{"position": index, "token": token} for index, token in enumerate(tokens)],
        "summary": (
            f"PE(position, 2i) = sin(position / 10000^(2i/{d_model})), "
            f"and dimension 2i+1 is the cosine. "
            f"Dimension {fastest['sin_dim']} repeats every {fastest['period']} positions; "
            f"dimension {slowest['sin_dim']} repeats every {slowest['period']}. "
            f"Ticks mark {marks}."
        ),
    }


def _mean_attention(weights: torch.Tensor) -> list[list[float]]:
    matrix = weights[0].mean(dim=0)
    return [[round(float(value), 4) for value in row] for row in matrix.tolist()]


def beam_search(
    model: Transformer,
    src: torch.Tensor,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    beam_size: int,
    max_len: int,
    alpha: float,
) -> tuple[list[int], list[dict]]:
    """Return token ids, including <bos> and a trailing <eos>, plus each search step."""
    src_mask = padding_mask(src, src_vocab.pad_id)
    memory = model.encode(src, src_mask)
    beam_size = max(1, min(beam_size, len(tgt_vocab) - 2))
    beams = [([tgt_vocab.bos_id], 0.0)]
    finished: list[tuple[list[int], float]] = []
    recorded: list[dict] = []

    for _ in range(max_len):
        active = []
        for sequence, score in beams:
            if sequence[-1] == tgt_vocab.eos_id:
                finished.append((sequence, score))
            else:
                active.append((sequence, score))
        if not active:
            beams = []
            break

        width = max(len(sequence) for sequence, _ in active)
        target = torch.full(
            (len(active), width),
            tgt_vocab.pad_id,
            dtype=torch.long,
            device=src.device,
        )
        for row, (sequence, _) in enumerate(active):
            target[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=src.device)

        hidden, _ = model.decode(
            target,
            memory.expand(len(active), -1, -1),
            causal_padding_mask(target, tgt_vocab.pad_id),
            src_mask.expand(len(active), -1, -1, -1),
        )
        logits = model.project(hidden)
        last = torch.tensor([len(sequence) - 1 for sequence, _ in active], device=src.device)
        next_log = F.log_softmax(logits[torch.arange(len(active), device=src.device), last], dim=-1)
        next_log[:, tgt_vocab.pad_id] = float("-inf")
        next_log[:, tgt_vocab.bos_id] = float("-inf")
        for row, (sequence, _) in enumerate(active):
            # Every training target has at least one real token, so the first
            # choice is never end-of-sentence. Allowing it makes an uncertain
            # beam prefer the empty string.
            if len(sequence) == 1:
                next_log[row, tgt_vocab.eos_id] = float("-inf")
            if len(sequence) >= 2 and sequence[-1] == sequence[-2]:
                next_log[row, sequence[-1]] = float("-inf")
        show = min(max(beam_size, 5), next_log.size(-1))
        top_log, top_id = next_log.topk(show, dim=-1)

        candidates = []
        for row, (sequence, _score) in enumerate(active):
            alternatives = []
            for col in range(show):
                log_prob = float(top_log[row, col])
                if not math.isfinite(log_prob):
                    continue
                token_id = int(top_id[row, col])
                alternatives.append(
                    {
                        "token": _name(tgt_vocab, token_id),
                        "id": token_id,
                        "log_prob": round(log_prob, 4),
                        "probability": round(math.exp(log_prob), 4),
                    }
                )
            candidates.append(
                {
                    "prefix": [_name(tgt_vocab, index) for index in sequence],
                    "alternatives": alternatives,
                }
            )

        best: dict[tuple[int, ...], float] = {}
        for row, (sequence, score) in enumerate(active):
            for col in range(beam_size):
                extended = tuple(sequence + [int(top_id[row, col])])
                total = score + float(top_log[row, col])
                if not math.isfinite(total):
                    continue
                if extended not in best or total > best[extended]:
                    best[extended] = total
        ranked = sorted(
            best.items(),
            key=lambda item: item[1] / length_penalty(len(item[0]) - 1, alpha),
            reverse=True,
        )
        beams = [(list(sequence), score) for sequence, score in ranked[:beam_size]]
        beam_rows = []
        for sequence, score in beams:
            generated = [_name(tgt_vocab, index) for index in sequence[1:]]
            length = max(1, len(generated))
            penalty = length_penalty(length, alpha)
            beam_rows.append(
                {
                    "tokens": generated,
                    "log_prob": round(score, 4),
                    "length": length,
                    "penalty": round(penalty, 4),
                    "score": round(score / penalty, 4),
                }
            )
        recorded.append({"candidates": candidates, "beam": beam_rows})

    finished.extend(beams)
    best_sequence, _ = max(
        finished,
        key=lambda item: item[1] / length_penalty(len(item[0]) - 1, alpha),
    )
    final_names = [_name(tgt_vocab, index) for index in best_sequence[1:]]
    steps = []
    for index, step in enumerate(recorded[: len(final_names)]):
        wanted = [_name(tgt_vocab, tgt_vocab.bos_id)] + final_names[:index]
        match = next((item for item in step["candidates"] if item["prefix"] == wanted), None)
        alternatives = []
        if match is not None:
            for alternative in match["alternatives"]:
                alternative = dict(alternative)
                alternative["chosen"] = alternative["token"] == final_names[index]
                alternatives.append(alternative)
        beam_rows = []
        for row in step["beam"]:
            row = dict(row)
            row["on_path"] = row["tokens"] == final_names[: len(row["tokens"])]
            beam_rows.append(row)
        chosen = next((item for item in alternatives if item["chosen"]), None)
        on_path = next((item for item in beam_rows if item["on_path"]), None)
        summary = _decode_summary(wanted, chosen, on_path, alpha)
        steps.append(
            {
                "prefix": wanted,
                "alternatives": alternatives,
                "beam": beam_rows,
                "summary": summary,
            }
        )
    return best_sequence, steps


def _decode_summary(prefix: list[str], chosen: dict | None, on_path: dict | None, alpha: float) -> str:
    shown = " ".join(prefix)
    if chosen is None:
        return f"The decoder has read {shown}."
    percent = chosen["probability"] * 100
    text = (
        f"The decoder has read {shown}. "
        f"{chosen['token']} gets probability {chosen['probability']:.4f} "
        f"({percent:.1f}%), from log probability {chosen['log_prob']:.4f}."
    )
    if on_path is not None:
        text += (
            f" Its beam score is {on_path['log_prob']:.4f} / {on_path['penalty']:.4f} "
            f"= {on_path['score']:.4f}, using length penalty ((5+{on_path['length']})/6)^{alpha}."
        )
    return text


def _decoder_start(model: Transformer, tgt_vocab: Vocabulary) -> dict:
    """Target embedding of <bos> plus position 0, the decoder's first input vector."""
    scale = math.sqrt(model.d_model)
    raw = model.tgt_embed.weight[tgt_vocab.bos_id].detach()
    scaled = raw * scale
    position = model.positional.pe[0, 0]
    summed = scaled + position
    return {
        "token": tgt_vocab.BOS,
        "id": tgt_vocab.bos_id,
        "scale": round(scale, 3),
        "dimensions": int(model.d_model),
        "raw": [round(float(value), 4) for value in raw[:3]],
        "scaled": [round(float(value), 3) for value in scaled[:3]],
        "position": [round(float(value), 4) for value in position[:4]],
        "sum": [round(float(value), 4) for value in summed[:4]],
        "sum_norm": round(float(summed.norm()), 3),
    }


def _output_head(model: Transformer, hidden: torch.Tensor, tgt_vocab: Vocabulary) -> dict:
    """Project the <bos> decoder state onto every French token, then apply softmax."""
    vector = hidden[0, 0]
    logits = F.linear(vector, model.tgt_embed.weight)
    log_sum = torch.logsumexp(logits, dim=0)
    probabilities = torch.softmax(logits, dim=0)
    values, indices = torch.topk(logits, k=min(8, int(logits.numel())))
    rows = []
    for logit, index in zip(values.tolist(), indices.tolist()):
        token_id = int(index)
        rows.append(
            {
                "token": tgt_vocab.itos[token_id],
                "id": token_id,
                "logit": round(float(logit), 4),
                "probability": round(float(probabilities[token_id]), 4),
            }
        )
    winner = int(indices[0])
    embedding_row = model.tgt_embed.weight[winner]
    return {
        "input": tgt_vocab.BOS,
        "predicts": rows[0]["token"],
        "dimensions": int(vector.numel()),
        "vocab": int(logits.numel()),
        "h": [round(float(value), 4) for value in vector[:4]],
        "h_norm": round(float(vector.norm()), 3),
        "bias": False,
        "products": [round(float(vector[i] * embedding_row[i]), 4) for i in range(3)],
        "logit": round(float(torch.dot(vector, embedding_row)), 4),
        "sum_exp": round(float(torch.exp(log_sum)), 4),
        "top": rows,
    }


def _full_state(
    model: Transformer,
    src: torch.Tensor,
    source_tokens: list[str],
    tgt_vocab: Vocabulary,
    hypothesis: list[int],
) -> dict:
    decoder_ids = hypothesis[:-1]
    write_ids = hypothesis[1:]
    state = model.inspect(src, torch.tensor([decoder_ids], dtype=torch.long, device=src.device))
    inputs = [tgt_vocab.itos[index] for index in decoder_ids]
    writes = [tgt_vocab.itos[index] for index in write_ids]
    return {
        "word": _rows(state["src_word"]),
        "position": _rows(state["src_position"]),
        "sum": _rows(state["src_sum"]),
        "target_word": _rows(state["tgt_word"]),
        "target_position": _rows(state["tgt_position"]),
        "target_sum": _rows(state["tgt_sum"]),
        "target_inputs": inputs,
        "writes": writes,
        "decoder_start": _decoder_start(model, tgt_vocab),
        "encoder": [_encoder_record(snap, source_tokens) for snap in state["encoder"]],
        "decoder": [_decoder_record(snap, inputs, writes) for snap in state["decoder"]],
        "output_head": _output_head(model, state["decoder"][-1]["after_ffn"], tgt_vocab),
    }


def _rows(tensor: torch.Tensor, digits: int = 3) -> list[list[float]]:
    if tensor.dim() == 3:
        tensor = tensor[0]
    return [[round(float(value), digits) for value in row] for row in tensor.tolist()]


def _heads(tensor: torch.Tensor, digits: int = 4) -> list[list[list[float]]]:
    return [
        [[round(float(value), digits) for value in row] for row in head.tolist()]
        for head in tensor[0]
    ]


def _stream_norms(tensor: torch.Tensor) -> list[float]:
    if tensor.dim() == 3:
        tensor = tensor[0]
    return [round(float(value), 3) for value in tensor.norm(dim=-1).tolist()]


def _relu_counts(tensor: torch.Tensor) -> list[int]:
    if tensor.dim() == 3:
        tensor = tensor[0]
    return [int((row > 0).sum().item()) for row in tensor]


def _sinusoid_matrix(model: Transformer, dims: list[int]) -> dict:
    count = min(48, int(model.positional.pe.size(1)))
    table = model.positional.pe[0, :count]
    listed = ", ".join(str(dim) for dim in dims)
    values = [
        [round(float(table[position, dim]), 4) for position in range(count)]
        for dim in dims
    ]
    return {
        "dims": dims,
        "summary": (
            f"Dimensions {listed}, the same samples as the curves, "
            f"at positions 0–{count - 1}. Even rows are sines and odd rows are cosines."
        ),
        "values": values,
    }


def _encoder_record(snap: dict, labels: list[str]) -> dict:
    active = _relu_counts(snap["ffn_relu"])
    width = int(snap["ffn_relu"].size(-1))
    after_attention = _stream_norms(snap["after_attention"])
    after_ffn = _stream_norms(snap["after_ffn"])
    return {
        "summary": (
            f"At {labels[0]}, {active[0]} of {width} ReLU units are on. "
            f"Residual length {after_attention[0]} after attention and {after_ffn[0]} after the feed-forward block."
        ),
        "self_attention": _heads(snap["self_attention"]),
        "q": _rows(snap["q"]),
        "k": _rows(snap["k"]),
        "v": _rows(snap["v"]),
        "qk": _heads(snap["qk"]),
        "mixed": _rows(snap["mixed"]),
        "attn_out": _rows(snap["attn_out"]),
        "ffn_pre": _rows(snap["ffn_pre"]),
        "after_attention": _rows(snap["after_attention"]),
        "ffn_relu": _rows(snap["ffn_relu"]),
        "ffn_active": active,
        "ffn_width": width,
        "ffn_out": _rows(snap["ffn_out"]),
        "after_ffn": _rows(snap["after_ffn"]),
        "labels": labels,
    }


def _decoder_record(snap: dict, inputs: list[str], writes: list[str]) -> dict:
    active = _relu_counts(snap["ffn_relu"])
    width = int(snap["ffn_relu"].size(-1))
    after_self = _stream_norms(snap["after_self"])
    after_cross = _stream_norms(snap["after_cross"])
    after_ffn = _stream_norms(snap["after_ffn"])
    return {
        "summary": (
            f"While writing {writes[0]}, {active[0]} of {width} ReLU units are on. "
            f"Residual length {after_self[0]} after masked self-attention, "
            f"{after_cross[0]} after cross-attention, and {after_ffn[0]} after the feed-forward block."
        ),
        "self_attention": _heads(snap["self_attention"]),
        "cross_attention": _heads(snap["cross_attention"]),
        "self_labels": inputs,
        "cross_rows": writes,
        "after_self": _rows(snap["after_self"]),
        "after_cross": _rows(snap["after_cross"]),
        "ffn_relu": _rows(snap["ffn_relu"]),
        "ffn_active": active,
        "ffn_width": width,
        "after_ffn": _rows(snap["after_ffn"]),
        "writes": writes,
    }


def _trace(
    model: Transformer,
    src: torch.Tensor,
    source_tokens: list[str],
    source_ids: list[int],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    hypothesis: list[int],
    target_tokens: list[str],
    decode_steps: list[dict],
    alpha: float,
) -> dict:
    tokens = [
        {"text": token, "id": index, "unknown": index == src_vocab.unk_id}
        for token, index in zip(source_tokens, source_ids)
    ]
    vectors = _vector_rows(model, source_ids, model.src_embed)
    for row, token in zip(vectors, tokens):
        row["text"] = token["text"]
        row["id"] = token["id"]
    _memory, weights = model.encode_with_attention(src, padding_mask(src, src_vocab.pad_id))
    encoder_attention = _mean_attention(weights)
    focus = []
    for row_index, token in enumerate(source_tokens):
        column = max(range(len(source_tokens)), key=lambda index: encoder_attention[row_index][index])
        weight = encoder_attention[row_index][column]
        focus.append(f"{token} attends most to {source_tokens[column]} ({weight:.4f})")
    id_list = ", ".join(f"{token['text']} → {token['id']}" for token in tokens)
    scale = math.sqrt(model.d_model)
    sinusoids = _sinusoids(model, source_tokens)
    sample_dims = [
        dim
        for channel in sinusoids["channels"]
        for dim in (channel["sin_dim"], channel["cos_dim"])
    ]
    return {
        "d_model": model.d_model,
        "scale": round(scale, 3),
        "n_heads": model.n_heads,
        "n_layers": len(model.encoder_layers),
        "alpha": alpha,
        "tokens": tokens,
        "token_summary": f"Token ids from the English vocabulary: {id_list}.",
        "vectors": vectors,
        "scaling": _scaling_example(model, source_tokens, source_ids),
        "vector_summary": (
            f"Each id is a {model.d_model}-dimensional vector multiplied by "
            f"√{model.d_model} = {scale:.3f}. A sinusoid of the same width is then added. "
            "The table shows those lengths and the first four coordinates. "
            "Under it, the curves and the grid use the same dimensions."
        ),
        "sample_dims": sample_dims,
        "sinusoids": sinusoids,
        "sinusoid_matrix": _sinusoid_matrix(model, sample_dims),
        "full": _full_state(model, src, source_tokens, tgt_vocab, hypothesis),
        "encoder_attention": encoder_attention,
        "encoder_summary": (
            f"Last of {len(model.encoder_layers)} encoder layers, mean of {model.n_heads} heads. "
            + " ".join(focus)
            + "."
        ),
        "decode": decode_steps,
        "pieces": target_tokens,
        "sentence": detokenize(target_tokens),
    }


def _attention_rows(
    model: Transformer,
    src: torch.Tensor,
    hypothesis: list[int],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
) -> tuple[list[str], list]:
    if len(hypothesis) < 2:
        return [], []
    decoder_in = torch.tensor([hypothesis[:-1]], dtype=torch.long, device=src.device)
    src_mask = padding_mask(src, src_vocab.pad_id)
    memory = model.encode(src, src_mask)
    _hidden, weights = model.decode(
        decoder_in,
        memory,
        causal_padding_mask(decoder_in, tgt_vocab.pad_id),
        src_mask,
    )
    predicted = hypothesis[1:]
    weights = weights[0, :, : len(predicted), : src.size(1)]
    if predicted[-1] == tgt_vocab.eos_id:
        predicted = predicted[:-1]
        weights = weights[:, : len(predicted), :]
    tokens = [tgt_vocab.itos[index] for index in predicted]
    rounded = [
        [[round(float(value), 4) for value in row] for row in head] for head in weights.tolist()
    ]
    return tokens, rounded


def translate(
    model: Transformer,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    text: str,
    beam_size: int = 4,
    max_source: int = 20,
    max_decode: int = 32,
    alpha: float = 0.6,
) -> dict:
    was_training = model.training
    model.eval()
    try:
        source_tokens = tokenize(text)
        if not source_tokens:
            return {
                "translation": "",
                "source_tokens": [],
                "target_tokens": [],
                "attention": [],
                "unknown": [],
                "trace": None,
            }
        if len(source_tokens) > max_source:
            raise ValueError(
                f"That sentence is {len(source_tokens)} tokens. "
                f"This model was trained on sentences of at most {max_source}."
            )
        source_ids = src_vocab.encode(source_tokens)
        unknown = [token for token, index in zip(source_tokens, source_ids) if index == src_vocab.unk_id]
        device = next(model.parameters()).device
        src = torch.tensor([source_ids], dtype=torch.long, device=device)
        decode_limit = min(max_decode, max(4, len(source_tokens) + 8))
        with torch.inference_mode():
            hypothesis, decode_steps = beam_search(
                model, src, src_vocab, tgt_vocab, beam_size, decode_limit, alpha
            )
            target_tokens, attention = _attention_rows(
                model, src, hypothesis, src_vocab, tgt_vocab
            )
            trace = _trace(
                model,
                src,
                source_tokens,
                source_ids,
                src_vocab,
                tgt_vocab,
                hypothesis,
                target_tokens,
                decode_steps,
                alpha,
            )
        return {
            "translation": detokenize(target_tokens),
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "attention": attention,
            "unknown": unknown,
            "trace": trace,
        }
    finally:
        model.train(was_training)


def main(argv: list[str] | None = None) -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description="Translate one English sentence.")
    parser.add_argument("text")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args(argv)
    model, src_vocab, tgt_vocab, config, _blob = load_checkpoint(args.checkpoint)
    result = translate(
        model,
        src_vocab,
        tgt_vocab,
        args.text,
        beam_size=config.beam_size,
        max_source=config.max_len,
        max_decode=max(config.max_len * 3, 24),
        alpha=config.length_alpha,
    )
    print(result["translation"])
    if result["unknown"]:
        print("unknown: " + ", ".join(result["unknown"]))
    print(f"{parameter_count(model)} parameters")


if __name__ == "__main__":
    main()
