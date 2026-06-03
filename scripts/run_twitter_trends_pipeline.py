"""Run the Twitter/X trend-count data pipeline step."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_twitter_trends import main


if __name__ == "__main__":
    main()
