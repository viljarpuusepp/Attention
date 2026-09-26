"""Local page for typing an English sentence and reading the attention map."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from attention.model import parameter_count
from attention.paths import DEFAULT_CHECKPOINT
from attention.translate import load_checkpoint, translate

STATIC = Path(__file__).resolve().parent / "static" / "index.html"


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


class State:
    model = None
    src_vocab = None
    tgt_vocab = None
    config = None
    meta: dict = {}
    error = ""
    lock = threading.Lock()


def load_state(checkpoint: Path) -> None:
    if not checkpoint.exists():
        State.error = f"No checkpoint at {checkpoint.name}. Run python -m attention.train."
        return
    model, src_vocab, tgt_vocab, config, blob = load_checkpoint(
        checkpoint, torch.device("cpu")
    )
    State.model = model
    State.src_vocab = src_vocab
    State.tgt_vocab = tgt_vocab
    State.config = config
    State.meta = blob
    State.error = ""


class Handler(BaseHTTPRequestHandler):
    # One request per connection. Threading this server on Windows was closing
    # sockets before a response was written.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path == "/api/info":
            self._send_json(200, info_payload())
            return
        if path != "/":
            self._send_json(404, {"error": "Not found."})
            return
        if not STATIC.exists():
            self._send_json(500, {"error": "The page file is missing."})
            return
        body = STATIC.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/translate":
            self._send_json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > 20_000:
            self._send_json(400, {"error": "Send a short JSON body."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "That was not JSON."})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON object expected."})
            return
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            self._send_json(400, {"error": "Type an English sentence."})
            return
        if State.model is None or State.config is None:
            self._send_json(503, {"error": State.error or "The model is not loaded."})
            return
        try:
            with State.lock:
                result = translate(
                    State.model,
                    State.src_vocab,
                    State.tgt_vocab,
                    text.strip(),
                    beam_size=State.config.beam_size,
                    max_source=State.config.max_len,
                    max_decode=max(State.config.max_len * 3, 24),
                    alpha=State.config.length_alpha,
                )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            traceback.print_exc()
            self._send_json(500, {"error": "Translation failed."})
            return
        self._send_json(200, result)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def info_payload() -> dict:
    if State.model is None or State.config is None:
        return {"ready": False, "error": State.error or "The model is not loaded."}
    config = State.config
    meta = State.meta
    return {
        "ready": True,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "params": parameter_count(State.model),
        "val_loss": meta.get("val_loss"),
        "val_acc": meta.get("val_acc"),
        "train_pairs": meta.get("train_pairs"),
        "epoch": meta.get("epoch"),
        "step": meta.get("step"),
        "data_date": meta.get("data_date"),
        "data_license": meta.get("data_license"),
    }


def main(argv: list[str] | None = None) -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description="Serve the translation demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    try:
        load_state(args.checkpoint)
    except Exception as exc:
        State.error = str(exc)
        print(f"checkpoint did not load: {exc}", flush=True)
    if State.model is None:
        print(State.error, flush=True)
    else:
        meta = State.meta
        loss = meta.get("val_loss")
        acc = meta.get("val_acc")
        detail = ""
        if isinstance(loss, float) and isinstance(acc, float):
            detail = f"epoch {meta.get('epoch')} val loss {loss:.3f} acc {acc:.3f}"
        print(f"loaded {args.checkpoint.name} {detail}".rstrip(), flush=True)
    try:
        class DemoServer(HTTPServer):
            allow_reuse_address = True

        server = DemoServer((args.host, args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"could not listen on {args.host}:{args.port}: {exc}") from exc
    print(f"open http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
