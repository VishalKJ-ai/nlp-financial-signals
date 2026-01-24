"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config(project_root: Path) -> dict:
    """Load and return the pipeline configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_speeches_df() -> pd.DataFrame:
    """Create a small synthetic speeches DataFrame for testing.

    Returns:
        DataFrame with 20 sample central bank speech records.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-15", periods=20, freq="3W")
    sources = ["boe", "fed", "ecb"]
    speakers = {
        "boe": ["Andrew Bailey", "Ben Broadbent", "Silvana Tenreyro"],
        "fed": ["Jerome Powell", "Lael Brainard", "Christopher Waller"],
        "ecb": ["Christine Lagarde", "Philip Lane", "Isabel Schnabel"],
    }

    records = []
    for i, date in enumerate(dates):
        source = sources[i % 3]
        speaker = speakers[source][i % 3]
        records.append({
            "date": date,
            "title": f"Speech on monetary policy outlook {i + 1}",
            "speaker": speaker,
            "source": source,
            "doc_type": "speech",
            "text": (
                f"The current economic outlook presents significant challenges. "
                f"Inflation remains above our target, and we must carefully "
                f"consider the appropriate monetary policy stance. Recent data "
                f"suggests that inflationary pressures are {'easing' if i % 2 == 0 else 'persistent'}. "
                f"We remain committed to price stability while supporting "
                f"sustainable economic growth. The labour market continues to "
                f"show resilience, with unemployment at historically low levels. "
                f"Financial conditions have {'tightened' if i % 3 == 0 else 'remained broadly stable'}. "
                f"Global risks include geopolitical tensions and supply chain "
                f"disruptions. We will continue to monitor incoming data closely "
                f"and adjust our policy stance as appropriate. The path of "
                f"interest rates will depend on the evolving economic outlook."
            ),
            "url": f"https://example.com/speech-{i + 1}",
        })

    return pd.DataFrame(records)


@pytest.fixture
def sample_text() -> str:
    """Return a sample central bank speech text for preprocessing tests."""
    return (
        "Speech by the Governor\n\n"
        "Good morning. Thank you for the invitation to speak today.\n\n"
        "The UK economy has shown remarkable resilience in the face of "
        "significant global headwinds. GDP growth has moderated but remains "
        "positive, while inflation has begun its descent from peak levels.\n\n"
        "Our monetary policy decisions [1] are guided by careful analysis of "
        "incoming data. We assess a wide range of indicators including labour "
        "market conditions, wage growth, and inflation expectations.\n\n"
        "Looking ahead, we expect inflation to continue falling gradually "
        "towards our 2% target, though the pace of disinflation may be uneven. "
        "https://www.bankofengland.co.uk/report\n\n"
        "All rights reserved.\n"
        "For further information, please contact the press office.\n"
        "Page 1 of 3\n"
    )
