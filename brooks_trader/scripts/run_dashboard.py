"""Start the Streamlit Dashboard with a stable Arrow memory allocator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_APP = PROJECT_ROOT / "dashboard" / "app.py"
SOURCE_ROOT = PROJECT_ROOT / "src"


def main() -> None:
    """Configure Arrow before importing Streamlit, then run the Dashboard."""
    os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(SOURCE_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    sys.path.insert(0, str(SOURCE_ROOT))

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(DASHBOARD_APP), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
