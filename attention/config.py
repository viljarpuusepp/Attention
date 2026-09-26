"""Shared hyperparameters for the model, the trainer, and the server."""

from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # Architecture. The 2017 paper uses 512 / 8 heads / 6 layers / 2048.
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_pos: int = 128
    # Data and optimisation.
    max_len: int = 8
    limit: int = 400
    max_vocab: int = 8000
    batch_size: int = 64
    epochs: int = 80
    warmup: int = 60
    lr_factor: float = 0.3
    smoothing: float = 0.1
    seed: int = 7
    beam_size: int = 4
    length_alpha: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Config":
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in valid})
