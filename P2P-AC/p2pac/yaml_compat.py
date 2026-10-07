"""``import yaml`` that also works where PyYAML is not installed (falls back to third_party/pyyaml, pure Python)."""

from __future__ import annotations

import os
import sys

try:
    import yaml  # noqa: F401
except ImportError:  # e.g. the system python3 of an HPC login node
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "pyyaml"))
    import yaml  # noqa: F401,E402


def safe_load(stream):
    return yaml.safe_load(stream)
