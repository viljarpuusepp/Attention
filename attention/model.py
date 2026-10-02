"""Encoder–decoder transformer from Vaswani et al., 2017.

The stack is the paper's: sinusoidal positions, scaled dot-product attention,
multi-head attention, a position-wise ReLU network, and post-norm residuals.
Width and depth are configured by the caller so a CPU can train the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from attention.config import Config


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """True where a key is padding. Shape [batch, 1, 1, length]."""
    return tokens.eq(pad_id).unsqueeze(1).unsqueeze(2)


def causal_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """True where a decoder query must not attend. Shape [batch, 1, length, length]."""
    length = tokens.size(1)
    future = torch.triu(
        torch.ones(length, length, device=tokens.device, dtype=torch.bool),
        diagonal=1,
    )
    return future.view(1, 1, length, length) | padding_mask(tokens, pad_id)


class PositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), and cosine on the odd dimensions."""

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even so each frequency has a sin and a cos")
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", table.unsqueeze(0), persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.size(1) > self.pe.size(1):
            raise ValueError(
                f"sequence length {hidden.size(1)} exceeds the positional table {self.pe.size(1)}"
            )
        hidden = hidden + self.pe[:, : hidden.size(1)]
        return self.dropout(hidden)


class MultiHeadAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, run as several heads."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def decompose(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Same matmuls as forward, keeping Q, K, V, the raw scores, and the mix."""
        batch, query_len, _ = query.shape
        key_len = key.size(1)

        def split(projected: torch.Tensor, length: int) -> torch.Tensor:
            return projected.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        qh = split(q, query_len)
        kh = split(k, key_len)
        vh = split(v, key_len)
        dots = torch.matmul(qh, kh.transpose(-2, -1))
        scores = dots / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        mixed = torch.matmul(weights, vh)
        concat = mixed.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return {
            "q": q,
            "k": k,
            "v": v,
            "qk": dots,
            "weights": weights,
            "mixed": concat,
            "output": self.w_o(concat),
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.decompose(query, key, value, mask)
        return parts["output"], parts["weights"]


class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.relu(self.w_1(hidden)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.self_attn(hidden, hidden, hidden, src_mask)
        hidden = self.norm_attn(hidden + self.dropout(attended))
        hidden = self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))
        return hidden, weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, _ = self.self_attn(hidden, hidden, hidden, tgt_mask)
        hidden = self.norm_self(hidden + self.dropout(attended))
        attended, weights = self.cross_attn(hidden, memory, memory, src_mask)
        hidden = self.norm_cross(hidden + self.dropout(attended))
        hidden = self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))
        return hidden, weights


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_pos: int,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.positional = PositionalEncoding(d_model, max_pos, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.n_heads = n_heads

    def embed(self, tokens: torch.Tensor, table: nn.Embedding) -> torch.Tensor:
        # Section 3.4: scale the embeddings by sqrt(d_model) before adding positions.
        return self.positional(table(tokens) * math.sqrt(self.d_model))

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        hidden, _weights = self.encode_with_attention(src, src_mask)
        return hidden

    def encode_with_attention(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed(src, self.src_embed)
        weights = None
        for layer in self.encoder_layers:
            hidden, weights = layer(hidden, src_mask)
        if weights is None:
            raise RuntimeError("encoder has no layers")
        return hidden, weights

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed(tgt, self.tgt_embed)
        weights = None
        for layer in self.decoder_layers:
            hidden, weights = layer(hidden, memory, tgt_mask, src_mask)
        if weights is None:
            raise RuntimeError("decoder has no layers")
        return hidden, weights

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        # The pre-softmax projection shares its matrix with the target embedding.
        return F.linear(hidden, self.tgt_embed.weight)

    def _encode_layer(self, layer: EncoderLayer, hidden: torch.Tensor, mask: torch.Tensor):
        parts = layer.self_attn.decompose(hidden, hidden, hidden, mask)
        attended = parts["output"]
        after_attention = layer.norm_attn(hidden + layer.dropout(attended))
        pre = layer.ffn.w_1(after_attention)
        activated = F.relu(pre)
        fed = layer.ffn.w_2(activated)
        after_ffn = layer.norm_ffn(after_attention + layer.dropout(fed))
        return {
            "self_attention": parts["weights"],
            "after_attention": after_attention,
            "ffn_relu": activated,
            "after_ffn": after_ffn,
            "q": parts["q"],
            "k": parts["k"],
            "v": parts["v"],
            "qk": parts["qk"],
            "mixed": parts["mixed"],
            "attn_out": attended,
            "ffn_pre": pre,
            "ffn_out": fed,
        }

    def _decode_layer(
        self,
        layer: DecoderLayer,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        attended, self_weights = layer.self_attn(hidden, hidden, hidden, tgt_mask)
        after_self = layer.norm_self(hidden + layer.dropout(attended))
        attended, cross_weights = layer.cross_attn(after_self, memory, memory, src_mask)
        after_cross = layer.norm_cross(after_self + layer.dropout(attended))
        activated = F.relu(layer.ffn.w_1(after_cross))
        fed = layer.ffn.w_2(activated)
        after_ffn = layer.norm_ffn(after_cross + layer.dropout(fed))
        return after_ffn, self_weights, cross_weights, after_self, after_cross, activated, after_ffn

    def inspect(self, src: torch.Tensor, tgt: torch.Tensor) -> dict[str, torch.Tensor | list]:
        """Every tensor on one encoder pass and one decoder pass. Eval mode leaves dropout off."""
        src_mask = padding_mask(src, self.pad_id)
        tgt_mask = causal_padding_mask(tgt, self.pad_id)
        src_word = self.src_embed(src) * math.sqrt(self.d_model)
        src_position = self.positional.pe[:, : src.size(1)]
        hidden = self.positional(src_word)
        src_sum = hidden
        encoder = []
        for layer in self.encoder_layers:
            snap = self._encode_layer(layer, hidden, src_mask)
            hidden = snap["after_ffn"]
            encoder.append(snap)
        memory = hidden
        tgt_word = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_position = self.positional.pe[:, : tgt.size(1)]
        hidden = self.positional(tgt_word)
        tgt_sum = hidden
        decoder = []
        for layer in self.decoder_layers:
            (
                hidden,
                self_weights,
                cross_weights,
                after_self,
                after_cross,
                activated,
                after_ffn,
            ) = self._decode_layer(layer, hidden, memory, tgt_mask, src_mask)
            decoder.append(
                {
                    "self_attention": self_weights,
                    "cross_attention": cross_weights,
                    "after_self": after_self,
                    "after_cross": after_cross,
                    "ffn_relu": activated,
                    "after_ffn": after_ffn,
                }
            )
        return {
            "src_word": src_word,
            "src_position": src_position,
            "src_sum": src_sum,
            "tgt_word": tgt_word,
            "tgt_position": tgt_position,
            "tgt_sum": tgt_sum,
            "encoder": encoder,
            "decoder": decoder,
        }

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode(src, src_mask)
        hidden, weights = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.project(hidden), weights

    def initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()


def build_model(config: Config, src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_pos=config.max_pos,
    )
    model.initialize_parameters()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class LabelSmoothing(nn.Module):
    """Token label smoothing of 0.1, as used in section 5.4 of the paper."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if size <= 2:
            raise ValueError("vocabulary must contain more than the special tokens")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        if self.smoothing == 0.0:
            return F.nll_loss(log_probs, target, ignore_index=self.padding_idx)
        with torch.no_grad():
            true = torch.full_like(log_probs, self.smoothing / (self.size - 2))
            true.scatter_(1, target.unsqueeze(1), self.confidence)
            true[:, self.padding_idx] = 0
            true.masked_fill_(target.eq(self.padding_idx).unsqueeze(1), 0.0)
        loss = F.kl_div(log_probs, true, reduction="none").sum(dim=-1)
        denom = target.ne(self.padding_idx).sum().clamp(min=1)
        return loss.sum() / denom
