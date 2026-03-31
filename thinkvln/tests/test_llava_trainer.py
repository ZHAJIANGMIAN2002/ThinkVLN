from types import SimpleNamespace


def test_llava_trainer_uses_accelerator_config_when_legacy_batch_attrs_are_missing(monkeypatch):
    import llava.train.llava_trainer as trainer_mod

    captured = {}

    class _DummyAccelerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.gather_for_metrics = object()
            self.state = SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=None)

    class _DummyGradPlugin:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    monkeypatch.setattr(trainer_mod, "Accelerator", _DummyAccelerator)
    monkeypatch.setattr(trainer_mod, "GradientAccumulationPlugin", _DummyGradPlugin)
    monkeypatch.setattr(trainer_mod, "rank0_print", lambda *args, **kwargs: None)

    trainer = object.__new__(trainer_mod.LLaVATrainer)
    trainer.model = SimpleNamespace(tp_size=None)
    trainer.args = SimpleNamespace(
        gradient_accumulation_steps=8,
        accelerator_config=trainer_mod.AcceleratorConfig(),
        deepspeed_plugin=None,
        fsdp_config={},
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        parallelism_config=None,
        data_seed=None,
    )
    trainer.propagate_args_to_deepspeed = lambda: None

    trainer.create_accelerator_and_postprocess()

    assert "dataloader_config" in captured
    assert "dispatch_batches" not in captured
    assert captured["dataloader_config"].dispatch_batches is None
    assert trainer.is_tp_enabled is False


def test_llava_trainer_worker_init_fn_matches_torch_dataloader_signature(monkeypatch):
    import llava.train.llava_trainer as trainer_mod

    captured = {}

    class _DummyDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = dict(kwargs)

    trainer = object.__new__(trainer_mod.LLaVATrainer)
    trainer.train_dataset = [1, 2, 3]
    trainer.data_collator = lambda batch: batch
    trainer.accelerator = SimpleNamespace(prepare=lambda dataloader: dataloader)
    trainer.args = SimpleNamespace(
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
    )
    trainer._train_batch_size = 1
    trainer._get_train_sampler = lambda: None
    trainer._get_collator_with_removed_columns = lambda data_collator, description: data_collator

    monkeypatch.setattr(trainer_mod, "DataLoader", _DummyDataLoader)

    trainer.get_train_dataloader()

    worker_init_fn = captured["kwargs"]["worker_init_fn"]
    worker_init_fn(0)


def test_llava_trainer_save_checkpoint_uses_transformers_compat_signature(monkeypatch):
    import llava.train.llava_trainer as trainer_mod
    from transformers import Trainer

    captured = {}

    def _fake_parent_save_checkpoint(self, model, trial):
        captured["self"] = self
        captured["model"] = model
        captured["trial"] = trial

    monkeypatch.setattr(Trainer, "_save_checkpoint", _fake_parent_save_checkpoint)

    trainer = object.__new__(trainer_mod.LLaVATrainer)
    trainer.args = SimpleNamespace(
        tune_mm_mlp_adapter=False,
        mm_tunable_parts=None,
    )

    model = object()
    trial = object()
    metrics = {"loss": 1.0}

    trainer._save_checkpoint(model, trial, metrics)

    assert captured["self"] is trainer
    assert captured["model"] is model
    assert captured["trial"] is trial
