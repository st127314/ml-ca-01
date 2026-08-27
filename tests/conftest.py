import sys
from pathlib import Path

# pages import "model_utils" directly, the same way they do when app.py runs from
# inside app/code, so the tests need that directory on the path too
APP_CODE_DIR = Path(__file__).resolve().parents[1] / "app" / "code"
sys.path.insert(0, str(APP_CODE_DIR))
