import argparse
import json
import os
import logging
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from thinkvln.eval.close_eval_dist import all_reduce_scalar_dict, init_dist_mode
from thinkvln.eval.close_eval_models import build_nav_model
from thinkvln.eval.close_eval_runner import VLNEvaluator
from thinkvln.eval.close_eval_utils import (
    load_summary_full,
    summarize_oracle_aggregation,
    summarize_subtask_aggregation,
)
from thinkvln.models.navigation_model import NavigationModel


logger = logging.getLogger(__name__)


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=0, type=int, dest="local_rank", help="node rank")
    parser.add_argument("--model_path", type=str, default="", help="Path to model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="thinkvln",
        choices=["thinkvln", "streamvln", "thinkvln_actor"],
        help="Model type",
    )
    parser.add_argument(
        "--ladder_mode",
        type=str,
        default="legacy",
        choices=["legacy", "subtask", "oracle_switch", "both"],
        help="Closed-loop ladder evaluation mode",
    )
    parser.add_argument(
        "--summary_full_path",
        type=str,
        default=None,
        help="Path to summary_full.jsonl (required for ladder modes)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path when --model_path is a LoRA adapter",
    )
    parser.add_argument(
        "--subgoal_success_distance",
        type=float,
        default=0.5,
        help="Distance threshold (meters) for subtask success",
    )
    parser.add_argument(
        "--subtask_step_budget_factor",
        type=float,
        default=2.0,
        help="Subtask step budget = ceil(gt_subtask_steps * factor)",
    )
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Episode sampling rate in (0, 1]. Use <1.0 for quick partial evaluation.",
    )
    parser.add_argument("--output_path", type=str, default="./results/env_eval")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--model_max_length", type=int, default=4096)

    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_history", type=int, default=8)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level for close-loop evaluation",
    )
    parser.add_argument(
        "--debug_log_interval",
        type=int,
        default=25,
        help="Log rollout progress every N steps in subtask mode (<=0 disables interval logs)",
    )
    return parser


def eval():
    parser = build_parser()
    args = parser.parse_args()
    if not (0.0 < args.sample_rate <= 1.0):
        raise ValueError(f"--sample_rate must be in (0, 1], got {args.sample_rate}")
    _setup_logging(args.log_level)
    logger.info("Eval args parsed: ladder_mode=%s model_type=%s model_path=%s", args.ladder_mode, args.model_type, args.model_path)
    logger.info(
        "Habitat-Sim log env: MAGNUM_LOG=%s HABITAT_SIM_LOG=%s GLOG_minloglevel=%s",
        os.environ.get("MAGNUM_LOG"),
        os.environ.get("HABITAT_SIM_LOG"),
        os.environ.get("GLOG_minloglevel"),
    )

    rank, world_size, gpu = init_dist_mode()
    device = f"cuda:{gpu}" if world_size > 1 else args.device
    args.device = device
    logger.info("Distributed initialized: rank=%d world_size=%d gpu=%d device=%s", rank, world_size, gpu, device)

    if args.ladder_mode != "legacy":
        if args.model_type != "thinkvln_actor":
            raise ValueError("Ladder modes currently support model_type=thinkvln_actor only.")
        if not args.summary_full_path:
            raise ValueError("--summary_full_path is required when ladder_mode is not legacy.")
        logger.info("Loading summary_full from %s", args.summary_full_path)
        summary_full = load_summary_full(args.summary_full_path)
        logger.info("summary_full loaded: %d episode entries", len(summary_full))
    else:
        summary_full = None

    logger.info("Building navigation model...")
    nav_model = build_nav_model(args, str(device), rank, world_size)
    logger.info("Navigation model ready: %s", type(nav_model).__name__)

    os.makedirs(args.output_path, exist_ok=True)
    logger.info("Output path: %s", args.output_path)
    evaluate(nav_model, args, rank, world_size, gpu, summary_full=summary_full)


def evaluate(
    nav_model: NavigationModel,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    gpu: int,
    summary_full: Optional[Dict[str, Dict[str, Any]]] = None,
):
    nav_model.eval()
    sample_rate = float(getattr(args, "sample_rate", 1.0))
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(f"--sample_rate must be in (0, 1], got {sample_rate}")
    args.sample_rate = sample_rate
    logger.info(
        "Start evaluate(): ladder_mode=%s split=%s sample_rate=%.4f",
        args.ladder_mode,
        args.eval_split,
        sample_rate,
    )
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        nav_model=nav_model,
        epoch=0,
        args=args,
    )

    if args.ladder_mode == "legacy":
        logger.info("Running legacy closed-loop evaluation...")
        sucs, spls, oss, ones, ep_num = evaluator.eval_action(rank)

        if world_size > 1:
            ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
            dist.all_gather(ep_num_all, ep_num)
            ep_counts = [int(x.item()) for x in ep_num_all]

            sucs_all = [torch.zeros(ep_counts[i], dtype=sucs.dtype, device=sucs.device) for i in range(world_size)]
            spls_all = [torch.zeros(ep_counts[i], dtype=spls.dtype, device=spls.device) for i in range(world_size)]
            oss_all = [torch.zeros(ep_counts[i], dtype=oss.dtype, device=oss.device) for i in range(world_size)]
            ones_all = [torch.zeros(ep_counts[i], dtype=ones.dtype, device=ones.device) for i in range(world_size)]

            dist.barrier()
            dist.all_gather(sucs_all, sucs)
            dist.all_gather(spls_all, spls)
            dist.all_gather(oss_all, oss)
            dist.all_gather(ones_all, ones)
            dist.barrier()

            sucs_all = torch.cat(sucs_all, dim=0)
            spls_all = torch.cat(spls_all, dim=0)
            oss_all = torch.cat(oss_all, dim=0)
            ones_all = torch.cat(ones_all, dim=0)
        else:
            sucs_all = sucs
            spls_all = spls
            oss_all = oss
            ones_all = ones

        if rank == 0:
            result_all = {
                "sucs_all": (sum(sucs_all) / len(sucs_all)).item() if len(sucs_all) > 0 else 0.0,
                "spls_all": (sum(spls_all) / len(spls_all)).item() if len(spls_all) > 0 else 0.0,
                "oss_all": (sum(oss_all) / len(oss_all)).item() if len(oss_all) > 0 else 0.0,
                "ones_all": (sum(ones_all) / len(ones_all)).item() if len(ones_all) > 0 else 0.0,
                "length": len(sucs_all),
            }
            with open(os.path.join(args.output_path, "result.json"), "a", encoding="utf-8") as f:
                f.write(json.dumps(result_all) + "\n")
            logger.info("Legacy aggregate written: %s", result_all)

        if world_size > 1:
            dist.destroy_process_group()
        return

    if summary_full is None:
        raise ValueError("summary_full must be provided for ladder modes.")

    ladder_summary: Dict[str, Dict[str, Any]] = {}

    if args.ladder_mode in ("subtask", "both"):
        logger.info("Running subtask closed-loop evaluation...")
        local_subtask_stats = evaluator.eval_subtask_closed_loop(rank, summary_full)
        global_subtask_stats = all_reduce_scalar_dict(local_subtask_stats, evaluator.device)
        if rank == 0:
            subtask_metrics = summarize_subtask_aggregation(global_subtask_stats)
            ladder_summary["subtask_closed_loop"] = {
                **subtask_metrics,
                "total_subtasks": int(global_subtask_stats.get("subtasks_total", 0.0)),
                "successful_subtasks": int(global_subtask_stats.get("subtasks_success", 0.0)),
                "progress_samples": int(global_subtask_stats.get("progress_count", 0.0)),
                "episodes_total": int(global_subtask_stats.get("episodes_total", 0.0)),
                "episodes_evaluated": int(global_subtask_stats.get("episodes_evaluated", 0.0)),
                "episodes_missing_meta": int(global_subtask_stats.get("episodes_missing_meta", 0.0)),
                "episodes_malformed": int(global_subtask_stats.get("episodes_malformed", 0.0)),
            }
            logger.info("Subtask aggregate: %s", ladder_summary["subtask_closed_loop"])

    if args.ladder_mode in ("oracle_switch", "both"):
        logger.info("Running oracle-switch closed-loop evaluation...")
        local_oracle_stats = evaluator.eval_oracle_switch_closed_loop(rank, summary_full)
        global_oracle_stats = all_reduce_scalar_dict(local_oracle_stats, evaluator.device)
        if rank == 0:
            oracle_metrics = summarize_oracle_aggregation(global_oracle_stats)
            ladder_summary["oracle_switch_closed_loop"] = {
                **oracle_metrics,
                "episodes_total": int(global_oracle_stats.get("episodes_total", 0.0)),
                "episodes_evaluated": int(global_oracle_stats.get("episodes_evaluated", 0.0)),
                "episodes_missing_meta": int(global_oracle_stats.get("episodes_missing_meta", 0.0)),
                "episodes_malformed": int(global_oracle_stats.get("episodes_malformed", 0.0)),
            }
            logger.info("Oracle-switch aggregate: %s", ladder_summary["oracle_switch_closed_loop"])

    if rank == 0:
        ladder_summary_path = os.path.join(args.output_path, "ladder_summary.json")
        with open(ladder_summary_path, "w", encoding="utf-8") as f:
            json.dump(ladder_summary, f, indent=2)
        logger.info("Ladder summary written: %s", ladder_summary_path)

    if world_size > 1:
        dist.destroy_process_group()


__all__ = ["build_parser", "eval", "evaluate"]
