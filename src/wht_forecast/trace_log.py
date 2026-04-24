"""
Colored, structured trace lines for pipeline debugging (stdout).

Respects https://no-color.org/ via NO_COLOR. Set FORCE_COLOR=1 to emit ANSI
sequences even when stdout is not a TTY (e.g. some CI logs).
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    return sys.stdout.isatty()


class _S:
    RST = "\033[0m"
    BLD = "\033[1m"
    DIM = "\033[2m"
    YLW = "\033[33m"
    BLU = "\033[34m"
    MAG = "\033[35m"
    CYN = "\033[36m"


def _tag_for_pipeline(pipeline: str, color: bool) -> str:
    p = pipeline.upper()
    if not color:
        return f"[{p}]"
    if p == "WHT":
        return f"{_S.BLD}{_S.CYN}[WHT]{_S.RST}"
    if p == "EXP":
        return f"{_S.BLD}{_S.MAG}[EXP]{_S.RST}"
    return f"{_S.BLD}{_S.BLU}[{p}]{_S.RST}"


def log_trace(
    enabled: bool,
    *,
    pipeline: str,
    title: str,
    detail: str,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None:
    """
    Print one trace line: tag, optional Stage i/n, title, detail.

    Parameters
    ----------
    enabled
        When False, no output.
    pipeline
        Short pipeline id, e.g. "WHT", "EXP".
    title
        Short headline (professional, English).
    detail
        Compact facts (numbers, shapes); kept on the same line.
    step, total_steps
        If both set, renders "Stage step/total_steps" in highlighted form.
    """
    if not enabled:
        return
    use_color = _color_enabled()
    tag = _tag_for_pipeline(pipeline, use_color)
    if step is not None and total_steps is not None:
        if use_color:
            stage = f"{_S.BLD}{_S.YLW}Stage {step}/{total_steps}{_S.RST}"
        else:
            stage = f"Stage {step}/{total_steps}"
        stage_part = f"{stage} "
    else:
        stage_part = ""
    if use_color:
        head = f"{_S.BLD}{title}{_S.RST}"
        body = f"{_S.DIM}{detail}{_S.RST}"
    else:
        head = title
        body = detail
    print(f"{tag} {stage_part}{head} — {body}", flush=True)
