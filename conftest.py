"""Ensure the project root is importable when running pytest.

Tests do `import db`, `import recommender`, etc. Running `python -m pytest`
puts the current directory on sys.path automatically, but a bare `pytest`
(like CI uses) does not — so we add the repo root here explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
