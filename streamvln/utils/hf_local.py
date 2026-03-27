import os
from typing import Dict, Sequence, Tuple


def prepare_local_only_pretrained_kwargs(model_name_or_path: str) -> Tuple[Dict[str, str], Dict[str, bool]]:
    path = str(model_name_or_path or "").strip()
    if not path or not os.path.isdir(path):
        return {}, {}
    env_updates = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return env_updates, {"local_files_only": True}


def apply_local_only_env(env_updates: Dict[str, str]) -> None:
    for key, value in env_updates.items():
        os.environ[str(key)] = str(value)


def bootstrap_local_only_env_from_argv(argv: Sequence[str]) -> Dict[str, str]:
    args = list(argv or [])
    for idx, token in enumerate(args):
        if token != "--model_name_or_path":
            continue
        if idx + 1 >= len(args):
            break
        env_updates, _ = prepare_local_only_pretrained_kwargs(args[idx + 1])
        apply_local_only_env(env_updates)
        return env_updates
    return {}
