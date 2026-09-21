"""Example: attach PreViC to LLaVA-OneVision.

Importing this module installs the PreViC compression hook into the base VLM.
It is a thin wrapper around ``previc.preload`` (see that file for the hook point
`HOOK_MODULE`/`HOOK_ATTR` and the configuration environment variables).

Usage as an lmms-eval preload:

    PREVIC_KEEP_RATIO=0.3 python -m lmms_eval --model llava_onevision ... \
        --model_args "...,preload=integration/llava_onevision_patch.py"

Equivalent to ``import previc.preload``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import previc.preload  # noqa: F401,E402  installs the PreViC hook on import
