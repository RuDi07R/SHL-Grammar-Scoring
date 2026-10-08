from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline_common import OUTPUTS, build_linguistic_v2


def main() -> None:
    transcripts = pd.read_csv(OUTPUTS / "train_transcripts.csv")
    result = build_linguistic_v2(transcripts)
    path = OUTPUTS / "linguistic_features_v2.csv"
    result.to_csv(path, index=False)
    print(f"Saved {path} with shape {result.shape}")


if __name__ == "__main__":
    main()