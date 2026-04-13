#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN Environment Evaluation Script

Compatibility facade for refactored close-loop eval modules.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from thinkvln.eval.close_eval_cli import build_parser, eval, evaluate
from thinkvln.eval.close_eval_dist import (
    all_reduce_scalar_dict,
    get_rank,
    get_world_size,
    init_dist_mode,
)
from thinkvln.eval.close_eval_models import build_nav_model, load_thinkvln_actor_model
from thinkvln.eval.close_eval_runner import VLNEvaluator
from thinkvln.eval.close_eval_utils import (
    _normalize_subtask_idx,
    build_episode_key,
    build_subtask_spans,
    compute_step_budget,
    distance_based_progress,
    extract_scene_id,
    load_summary_full,
    normalize_action,
    parse_plan_steps,
    summarize_subtask_aggregation,
    timeline_progress,
)

__all__ = [
    "extract_scene_id",
    "build_episode_key",
    "_normalize_subtask_idx",
    "parse_plan_steps",
    "load_summary_full",
    "build_subtask_spans",
    "timeline_progress",
    "distance_based_progress",
    "compute_step_budget",
    "summarize_subtask_aggregation",
    "all_reduce_scalar_dict",
    "normalize_action",
    "load_thinkvln_actor_model",
    "build_nav_model",
    "VLNEvaluator",
    "init_dist_mode",
    "get_rank",
    "get_world_size",
    "build_parser",
    "eval",
    "evaluate",
]


if __name__ == "__main__":
    eval()
