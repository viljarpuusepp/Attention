"""Project paths. All of them hang off the repository root, not the shell cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS = DATA_DIR / "fra.txt"
ZIP_PATH = DATA_DIR / "fra-eng.zip"
CHECKPOINT_DIR = ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"
# The final epoch, not the best validation snapshot. On a few hundred short
# sentences the validation slice is too small to pick a better model.
DEFAULT_CHECKPOINT = LAST_CHECKPOINT

DATA_URL = "https://www.manythings.org/anki/fra-eng.zip"
DATA_DATE = "2026-02-13"
DATA_LICENSE = "CC BY 2.0"
