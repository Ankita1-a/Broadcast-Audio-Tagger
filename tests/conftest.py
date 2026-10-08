# Lets tests import code from src/ (e.g. `from serve.timeline import tag`).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))