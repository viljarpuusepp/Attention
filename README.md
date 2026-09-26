# Attention

A small English-to-French transformer, following Vaswani et al., *Attention Is All You Need* (NeurIPS 2017). It trains on short sentences and serves a local page that shows the decoder’s cross-attention.

The architecture matches the paper: scaled dot-product attention, multi-head attention, sinusoidal positional encoding, a position-wise ReLU feed-forward network, residual dropout, post-norm layers, tied target embeddings, Adam with the warmup schedule, label smoothing of 0.1, and beam search of size 4 with length penalty 0.6.

The WMT model in the paper is 6 layers, d_model 512, 8 heads, and a 32k subword vocabulary, trained on WMT 2014. This copy is 2 layers, d_model 128, 4 heads, and a word vocabulary, small enough to train on a CPU. It has learned a few hundred of the shortest Tatoeba sentences.

## Data

English–French sentence pairs from the [Tatoeba Project](https://tatoeba.org), compiled by [manythings.org/anki](https://www.manythings.org/anki/) (`fra-eng.zip`, file date 2026-02-13). License: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Attribution: www.manythings.org/anki and tatoeba.org. The third column of `data/fra.txt` is the per-sentence credit.

`python -m attention.train` downloads that zip if `data/fra.txt` is missing. The copy already in `data/` is the file used to train the checkpoint.

Training keeps sentences of at most 8 tokens, keeps the shortest French translation of each English sentence, and uses the shortest 400 of those pairs (387 train, 17 validation after the holdout). The learning-rate schedule is the paper's, multiplied by 0.3 so the peak stays near the paper's on this smaller model. Warmup is 60 steps, then 80 epochs.

The shipped weights are the final epoch (`checkpoints/last.pt`): training token accuracy 99.9%, validation token accuracy 67.8%. `checkpoints/best.pt` is an earlier epoch with a lower validation loss on that tiny holdout. The page is aimed at the short sentences it memorized. A word that never appeared in those 400 lines is shown as unknown.

## Run

From this folder, with the virtual environment that holds PyTorch:

```
.\.venv\Scripts\python.exe -m attention.server
```

Open http://127.0.0.1:8000. Type a short English sentence, or click one of the examples. The grid is the last decoder layer’s cross-attention.

One sentence from the command line:

```
.\.venv\Scripts\python.exe -m attention.translate "Hello."
```

Retrain, or train from scratch after deleting `checkpoints/`:

```
.\.venv\Scripts\python.exe -m attention.train
```

To recreate the environment:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

## Tests

```
.\.venv\Scripts\python.exe -m unittest tests.test_model -v
```

The tests check the positional-encoding formula, the causal and padding masks, the learning-rate peak at the warmup step, and that a tiny model can memorize four hand-written pairs.

## Layout

- `attention/model.py` — the transformer
- `attention/data.py` — tokenizer, download, vocabulary
- `attention/train.py` — training loop
- `attention/translate.py` — beam search
- `attention/server.py` and `attention/static/index.html` — the local page
- `checkpoints/last.pt` — the final epoch, which the page loads
- `checkpoints/best.pt` — the epoch with the lowest validation loss
- `data/fra.txt` — the sentence pairs, with credits
