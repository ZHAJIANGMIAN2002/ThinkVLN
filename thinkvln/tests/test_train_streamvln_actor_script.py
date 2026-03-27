import importlib.util
import json
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "train_streamvln_actor.py"
    spec = importlib.util.spec_from_file_location("train_streamvln_actor_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_streamvln_train_argv_from_yaml_sections():
    module = _load_script_module()
    config = {
        "model": {
            "model_name_or_path": "/models/streamvln",
            "model_type": "streamvln_actor",
            "progress_loss_weight": 1.0,
            "done_loss_weight": 1.0,
            "lora_enable": True,
            "lora_r": 64,
        },
        "data": {
            "summary_data_path": "/data/train.jsonl",
            "num_history": 8,
            "num_future_steps": 4,
            "done_threshold": 0.85,
        },
        "training": {
            "output_dir": "/outputs/actor",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "bf16": True,
            "deepspeed": "config/zero2.json",
        },
        "logging": {
            "report_to": ["wandb"],
            "run_name": "streamvln-actor-test",
            "logging_dir": "/logs/actor",
        },
    }

    argv = module.build_streamvln_train_argv(config)

    assert "--model_name_or_path" in argv
    assert "/models/streamvln" in argv
    assert "--summary_data_path" in argv
    assert "/data/train.jsonl" in argv
    assert "--num_history" in argv
    assert "8" in argv
    assert "--num_future_steps" in argv
    assert "4" in argv
    assert "--lora_enable" in argv
    assert "True" in argv
    assert "--deepspeed" in argv
    assert "config/zero2.json" in argv
    assert "--report_to" in argv
    assert "wandb" in argv
    assert "--run_name" in argv
    assert "streamvln-actor-test" in argv


def test_materialize_subset_jsonl_from_max_samples(tmp_path):
    module = _load_script_module()
    src_path = tmp_path / "train.jsonl"
    rows = [{"id": idx} for idx in range(5)]
    src_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    resolved_path, cleanup_path = module.materialize_summary_data_path(
        {
            "summary_data_path": str(src_path),
            "max_samples": 2,
        }
    )

    assert cleanup_path is not None
    assert resolved_path != str(src_path)
    subset_lines = Path(resolved_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(subset_lines) == 2


def test_build_runtime_env_includes_wandb_settings():
    module = _load_script_module()
    config = {
        "logging": {
            "wandb": {
                "project": "thinkvln",
                "entity": "lab",
                "mode": "offline",
                "tags": ["streamvln_actor", "debug"],
            }
        },
        "runtime": {
            "env": {
                "TOKENIZERS_PARALLELISM": "false",
            }
        },
    }

    env = module.build_runtime_env(config)

    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["WANDB_PROJECT"] == "thinkvln"
    assert env["WANDB_ENTITY"] == "lab"
    assert env["WANDB_MODE"] == "offline"
    assert env["WANDB_TAGS"] == "streamvln_actor,debug"
