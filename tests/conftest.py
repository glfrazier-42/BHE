"""
Pytest configuration for the black hole explosion simulation tests.

This file ensures the src module is importable from tests.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
