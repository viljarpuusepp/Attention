"""Learning-rate schedule from Attention Is All You Need, section 5.3.

lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""


def paper_lr(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    """Paper schedule, times `factor`.

    With d_model 512 and warmup 4000 the paper's peak is about 7e-4. A smaller
    model and a shorter warmup would peak several times higher, which collapses
    this network, so training uses a factor below 1.
    """
    step = max(int(step), 1)
    scale = factor * (d_model**-0.5)
    return scale * min(step**-0.5, step * warmup**-1.5)


def length_penalty(length: int, alpha: float = 0.6) -> float:
    """Wu et al. length penalty, with the paper's alpha of 0.6."""
    return ((5.0 + max(length, 1)) / 6.0) ** alpha
