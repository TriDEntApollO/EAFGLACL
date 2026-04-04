from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_prep.normalization import clean_text, normalize_label


def test_label_normalization() -> None:
    assert normalize_label("non-rumour") == "non-rumor"
    assert normalize_label("REAL") == "true"
    assert normalize_label("fake") == "false"


def test_clean_text() -> None:
    raw = "hello URL <b>world</b> https://example.com"
    cleaned = clean_text(raw)
    assert cleaned == "hello world"
