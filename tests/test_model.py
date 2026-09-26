"""Checks for the masks, the schedule, the data filter, and a tiny memorisation run."""

import math
import unittest

import torch
import torch.nn.functional as F

from attention.config import Config
from attention.data import detokenize, normalize_pairs, tokenize, Vocabulary
from attention.model import (
    LabelSmoothing,
    MultiHeadAttention,
    PositionalEncoding,
    build_model,
    causal_padding_mask,
    padding_mask,
)
from attention.schedule import length_penalty, paper_lr
from attention.translate import translate


class ScheduleTest(unittest.TestCase):
    def test_learning_rate_peaks_at_warmup(self):
        warmup = 400
        d_model = 128
        at_warmup = paper_lr(warmup, d_model, warmup)
        self.assertGreater(at_warmup, paper_lr(warmup - 1, d_model, warmup))
        self.assertGreater(at_warmup, paper_lr(warmup + 1, d_model, warmup))
        self.assertAlmostEqual(at_warmup, (d_model**-0.5) * (warmup**-0.5))

    def test_length_penalty_matches_the_paper_at_length_one(self):
        self.assertAlmostEqual(length_penalty(1, 0.6), 1.0)


class DataTest(unittest.TestCase):
    def test_tokenizer_keeps_french_and_punctuation(self):
        self.assertEqual(tokenize("Go."), ["go", "."])
        self.assertEqual(tokenize("J'ai pigé !"), ["j", "'", "ai", "pigé", "!"])
        self.assertEqual(tokenize("Comment vas-tu\u202f?"), ["comment", "vas", "tu", "?"])
        self.assertEqual(detokenize(["j", "'", "ai", "pigé", "!"]), "J'ai pigé !")
        self.assertEqual(detokenize(["go", "."]), "Go.")
        self.assertEqual(tokenize(detokenize(["i", "'", "m", "happy", "."])), ["i", "'", "m", "happy", "."])

    def test_filter_keeps_the_shortest_french_sentence(self):
        raw = [
            (["go", "."], ["va", "!"]),
            (["go", "."], ["marche", "."]),
            (["hi", "."], ["salut", "."]),
            (["this", "sentence", "is", "far", "too", "long", "now"], ["non"]),
        ]
        train, val = normalize_pairs(raw, max_len=5, limit=10, seed=0)
        paired = {tuple(english): french for english, french in train + val}
        self.assertEqual(set(paired), {("go", "."), ("hi", ".")})
        self.assertEqual(paired[("go", ".")], ["va", "!"])

    def test_keep_forces_demo_sentences_into_training(self):
        raw = [
            (["go", "."], ["va", "!"]),
            (["hi", "."], ["salut", "."]),
            (["a", "bit", "longer", "now"], ["bonjour", "."]),
        ]
        train, val = normalize_pairs(
            raw,
            max_len=6,
            limit=1,
            seed=0,
            keep=[["a", "bit", "longer", "now"]],
        )
        self.assertIn((["a", "bit", "longer", "now"], ["bonjour", "."]), train)
        self.assertTrue(all(english != ["a", "bit", "longer", "now"] for english, _ in val))


class ModelTest(unittest.TestCase):
    def test_positional_encoding_matches_the_formula(self):
        d_model = 16
        encoding = PositionalEncoding(d_model, max_len=30, dropout=0.0)
        encoding.eval()
        values = encoding(torch.zeros(1, 30, d_model))
        position = 5
        index = 2
        angle = position / (10000 ** (2 * index / d_model))
        self.assertAlmostEqual(values[0, position, 2 * index].item(), math.sin(angle), places=5)
        self.assertAlmostEqual(values[0, position, 2 * index + 1].item(), math.cos(angle), places=5)
        self.assertAlmostEqual(values[0, 0, 1].item(), 1.0, places=5)

    def test_causal_mask_blocks_the_future_and_padding(self):
        tokens = torch.tensor([[5, 6, 0]])
        mask = causal_padding_mask(tokens, 0)
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 3))
        self.assertTrue(bool(mask[0, 0, 0, 1]))
        self.assertFalse(bool(mask[0, 0, 1, 0]))
        self.assertTrue(bool(mask[0, 0, 0, 2]))

    def test_attention_cannot_use_a_masked_key(self):
        attention = MultiHeadAttention(4, 1)
        with torch.no_grad():
            for layer in (attention.w_q, attention.w_k, attention.w_v, attention.w_o):
                layer.weight.copy_(torch.eye(4))
                layer.bias.zero_()
        hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [50.0, 0.0, 0.0, 0.0]]])
        mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        mask[:, :, 0, 1] = True
        output, _weights = attention(hidden, hidden, hidden, mask)
        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-4))
        self.assertTrue(torch.allclose(output[0, 1], torch.tensor([50.0, 0.0, 0.0, 0.0]), atol=1e-3))

    def test_padding_does_not_change_real_positions(self):
        torch.manual_seed(0)
        config = Config(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0, max_pos=32)
        model = build_model(config, 20, 20)
        model.eval()
        source = torch.tensor([[5, 6, 7]])
        padded = torch.tensor([[5, 6, 7, 0, 0]])
        target = torch.tensor([[1, 5, 6]])
        short, _ = model(source, target, padding_mask(source, 0), causal_padding_mask(target, 0))
        long, _ = model(padded, target, padding_mask(padded, 0), causal_padding_mask(target, 0))
        self.assertTrue(torch.allclose(short, long, atol=1e-5))
        self.assertEqual(tuple(short.shape), (1, 3, 20))

    def test_output_projection_uses_the_target_embedding(self):
        config = Config(d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0)
        model = build_model(config, 20, 18)
        hidden = torch.randn(2, 3, config.d_model)
        self.assertTrue(torch.allclose(model.project(hidden), F.linear(hidden, model.tgt_embed.weight)))

    def test_label_smoothing_is_finite(self):
        criterion = LabelSmoothing(10, padding_idx=0, smoothing=0.1)
        logits = torch.randn(4, 10, requires_grad=True)
        target = torch.tensor([1, 2, 0, 3])
        loss = criterion(logits, target)
        loss.backward()
        self.assertTrue(math.isfinite(loss.item()))
        self.assertIsNotNone(logits.grad)

    def test_model_memorizes_four_pairs(self):
        torch.manual_seed(0)
        pairs = [
            (["i", "am", "happy", "."], ["je", "suis", "content", "."]),
            (["i", "am", "sad", "."], ["je", "suis", "triste", "."]),
            (["you", "are", "happy", "."], ["tu", "es", "content", "."]),
            (["you", "are", "sad", "."], ["tu", "es", "triste", "."]),
        ]
        src_vocab = Vocabulary.build([english for english, _ in pairs], 50)
        tgt_vocab = Vocabulary.build([french for _, french in pairs], 50)
        config = Config(d_model=64, n_heads=4, n_layers=1, d_ff=128, dropout=0.0, max_pos=32)
        model = build_model(config, len(src_vocab), len(tgt_vocab))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = LabelSmoothing(len(tgt_vocab), tgt_vocab.pad_id, smoothing=0.0)
        from attention.data import collate

        source, target_in, target_out = collate(pairs, src_vocab, tgt_vocab)
        model.train()
        last = None
        for _ in range(250):
            logits, _ = model(
                source,
                target_in,
                padding_mask(source, src_vocab.pad_id),
                causal_padding_mask(target_in, tgt_vocab.pad_id),
            )
            last = criterion(logits.reshape(-1, logits.size(-1)), target_out.reshape(-1))
            optimizer.zero_grad()
            last.backward()
            optimizer.step()
        self.assertLess(last.item(), 0.05)
        for english, french in pairs:
            result = translate(
                model,
                src_vocab,
                tgt_vocab,
                detokenize(english),
                beam_size=4,
                max_source=12,
                max_decode=12,
            )
            self.assertEqual(result["target_tokens"], french)
            self.assertEqual(result["source_tokens"], english)
            self.assertEqual(len(result["attention"]), config.n_heads)
            self.assertEqual(len(result["attention"][0]), len(french))
            self.assertEqual(len(result["attention"][0][0]), len(english))
            trace = result["trace"]
            self.assertEqual([row["text"] for row in trace["tokens"]], english)
            self.assertEqual(trace["pieces"], french)
            self.assertEqual(len(trace["encoder_attention"]), len(english))
            self.assertEqual(len(trace["vectors"][0]["sum_dims"]), 4)
            scaling = trace["scaling"]
            self.assertEqual(scaling["token"], english[0])
            self.assertEqual(scaling["dimensions"], config.d_model)
            self.assertAlmostEqual(math.sqrt(scaling["sum_of_squares"]), scaling["norm"], places=2)
            for raw, scaled in zip(scaling["raw"], scaling["scaled"]):
                self.assertAlmostEqual(raw * scaling["scale"], scaled, places=2)
            waves = trace["sinusoids"]
            fastest = waves["channels"][0]
            self.assertEqual(fastest["sin_dim"], 0)
            self.assertEqual(fastest["cos_dim"], 1)
            self.assertAlmostEqual(fastest["period"], 2 * math.pi, places=2)
            self.assertAlmostEqual(fastest["sin"][0], 0.0, places=3)
            self.assertAlmostEqual(fastest["cos"][0], 1.0, places=3)
            sample = int(round(4 / waves["step"]))
            self.assertAlmostEqual(fastest["sin"][sample], math.sin(4), places=3)
            self.assertEqual(waves["marks"][0]["token"], english[0])
            self.assertEqual(waves["marks"][0]["position"], 0)
            matrix = trace["sinusoid_matrix"]["values"]
            self.assertEqual(trace["sinusoid_matrix"]["dims"], trace["sample_dims"])
            self.assertEqual(len(matrix), len(trace["sample_dims"]))
            self.assertEqual(trace["sample_dims"][:2], [0, 1])
            self.assertAlmostEqual(matrix[0][0], 0.0, places=3)
            self.assertAlmostEqual(matrix[1][0], 1.0, places=3)
            full = trace["full"]
            self.assertEqual(len(full["word"][0]), config.d_model)
            self.assertEqual(len(full["position"][0]), config.d_model)
            self.assertEqual(len(full["sum"][0]), config.d_model)
            self.assertEqual(len(full["encoder"]), config.n_layers)
            self.assertEqual(len(full["decoder"]), config.n_layers)
            self.assertEqual(len(full["encoder"][0]["self_attention"]), config.n_heads)
            self.assertEqual(len(full["encoder"][0]["ffn_relu"][0]), config.d_ff)
            self.assertEqual(len(full["encoder"][0]["after_attention"][0]), config.d_model)
            self.assertEqual(len(full["encoder"][0]["after_ffn"][0]), config.d_model)
            self.assertEqual(len(full["decoder"][0]["self_attention"]), config.n_heads)
            self.assertEqual(len(full["decoder"][0]["cross_attention"]), config.n_heads)
            self.assertEqual(len(full["decoder"][0]["ffn_relu"][0]), config.d_ff)
            self.assertGreaterEqual(min(full["encoder"][0]["ffn_relu"][0]), 0.0)
            self.assertEqual(len(full["decoder"][0]["after_self"][0]), config.d_model)
            self.assertEqual(len(full["decoder"][0]["after_cross"][0]), config.d_model)
            written = []
            for step in trace["decode"]:
                chosen = [item["token"] for item in step["alternatives"] if item["chosen"]]
                self.assertEqual(len(chosen), 1)
                self.assertGreater(step["alternatives"][0]["probability"], 0)
                if chosen[0] != "<eos>":
                    written.append(chosen[0])
            self.assertEqual(written, french)


if __name__ == "__main__":
    unittest.main()
