"""Project-wide filesystem paths.

Every path is resolved relative to the project root (not the current working
directory), so the scripts work no matter where they are launched from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.keras"
