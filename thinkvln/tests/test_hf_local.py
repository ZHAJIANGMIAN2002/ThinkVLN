import os

from streamvln.utils.hf_local import (
    bootstrap_local_only_env_from_argv,
    prepare_local_only_pretrained_kwargs,
)


def test_prepare_local_only_pretrained_kwargs_enables_offline_for_local_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    env_updates, kwargs = prepare_local_only_pretrained_kwargs(str(model_dir))

    assert kwargs["local_files_only"] is True
    assert env_updates["HF_HUB_OFFLINE"] == "1"
    assert env_updates["TRANSFORMERS_OFFLINE"] == "1"


def test_prepare_local_only_pretrained_kwargs_noops_for_remote_repo():
    env_updates, kwargs = prepare_local_only_pretrained_kwargs("meta-llama/Meta-Llama-3-8B-Instruct")

    assert env_updates == {}
    assert kwargs == {}


def test_bootstrap_local_only_env_from_argv_sets_offline(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    env_updates = bootstrap_local_only_env_from_argv(
        ["train.py", "--model_name_or_path", str(model_dir)]
    )

    assert env_updates["HF_HUB_OFFLINE"] == "1"
    assert env_updates["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
