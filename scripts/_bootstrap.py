"""Allow source-checkout scripts before an editable installation."""
import os
import sys
from pathlib import Path


def run(command: str) -> None:
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    # Repeated gate evaluation launches python -m children; inherit this checkout.
    entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if source not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([source, *filter(None, entries)])
    from mi_eeg.cli import main
    main([command, *sys.argv[1:]])
