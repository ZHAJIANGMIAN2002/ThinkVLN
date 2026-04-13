from types import SimpleNamespace


def test_load_tokenizer_uses_fallback_for_unknown_model_family(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    calls = {}

    class _DummyTokenizer:
        pad_token = None
        unk_token = "<unk>"
        model_max_length = 4096
        padding_side = "right"

    def _fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = dict(kwargs)
        return _DummyTokenizer()

    monkeypatch.setattr(
        train_mod.transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=_fake_from_pretrained),
    )

    tokenizer = train_mod.load_tokenizer(
        SimpleNamespace(model_name_or_path=str(tmp_path / "custom-streamvln-model")),
        SimpleNamespace(cache_dir="/tmp/cache", model_max_length=2048),
        {"local_files_only": True},
    )

    assert isinstance(tokenizer, _DummyTokenizer)
    assert calls["path"] == str(tmp_path / "custom-streamvln-model")
    assert calls["kwargs"]["cache_dir"] == "/tmp/cache"
    assert calls["kwargs"]["model_max_length"] == 2048
    assert calls["kwargs"]["padding_side"] == "right"
    assert "use_fast" not in calls["kwargs"]


def test_get_model_loads_config_when_actor_overrides_exist(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    calls = {}

    class _DummyConfig:
        def __init__(self):
            self.num_hidden_layers = 4
            self.sliding_window = 128
            self.max_window_layers = 2

    def _fake_config_from_pretrained(path, **kwargs):
        calls["config_path"] = path
        calls["config_kwargs"] = dict(kwargs)
        return _DummyConfig()

    class _DummyModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = dict(kwargs)
            assert kwargs["config"].layer_types == [
                "full_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
            ]
            return SimpleNamespace(config=kwargs.get("config"))

    monkeypatch.setattr(
        train_mod,
        "AutoConfig",
        SimpleNamespace(from_pretrained=_fake_config_from_pretrained),
    )
    monkeypatch.setattr(train_mod, "StreamVLNForCausalLM", _DummyModel)

    model_args = SimpleNamespace(
        model_name_or_path=str(tmp_path),
        rope_scaling_factor=None,
        rope_scaling_type=None,
        mm_spatial_pool_stride=None,
        mm_spatial_pool_out_channels=None,
        mm_spatial_pool_mode=None,
        mm_resampler_type=None,
        use_pos_skipping=False,
        pos_skipping_range=4096,
        mm_spatial_pool_size=None,
        progress_loss_weight=1.0,
        done_loss_weight=1.0,
        mm_tunable_parts=None,
        mm_newline_position="grid",
        mm_patch_merge_type="flat",
    )
    training_args = SimpleNamespace(
        attn_implementation="flash_attention_2",
        cache_dir=None,
        bf16=False,
    )
    data_args = SimpleNamespace(
        num_future_steps=4,
        num_history=8,
    )

    model = train_mod.get_model(model_args, training_args, data_args, {})

    assert calls["config_path"] == str(tmp_path)
    assert calls["config_kwargs"]["local_files_only"] is True
    assert isinstance(model.config, _DummyConfig)


def test_find_all_linear_names_ignores_numeric_suffix_modules(monkeypatch, tmp_path):
    import sys
    import torch

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "self_attn": torch.nn.ModuleDict(
                                {
                                    "q_proj": torch.nn.Linear(4, 4),
                                    "k_proj": torch.nn.Linear(4, 4),
                                }
                            )
                        }
                    )
                ]
            )
            self.progress_head = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            )

    names = train_mod.find_all_linear_names(_DummyModel())

    assert "q_proj" in names
    assert "k_proj" in names
    assert "0" not in names
    assert "2" not in names


def test_smart_tokenizer_and_embedding_resize_does_not_shrink_padded_vocab(monkeypatch, tmp_path):
    import sys
    import torch

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    class _DummyTokenizer:
        def __len__(self):
            return 151647

        def add_special_tokens(self, special_tokens_dict):
            return len(special_tokens_dict)

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_embeddings = torch.nn.Embedding(152064, 8)
            self.output_embeddings = torch.nn.Linear(8, 152064, bias=False)
            self.resize_calls = []

        def resize_token_embeddings(self, new_size):
            self.resize_calls.append(int(new_size))

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

    model = _DummyModel()
    tokenizer = _DummyTokenizer()

    train_mod.smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, model)

    assert model.resize_calls == []


def test_make_supervised_data_module_splits_streamvln_actor_dataset(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    class _DummyDataset:
        def __init__(self, *args, **kwargs):
            self.lengths = [1] * 10

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, idx):
            return {"idx": idx}

    monkeypatch.setattr(train_mod, "StreamVLNActorDataset", _DummyDataset)

    module = train_mod.make_supervised_data_module(
        tokenizer=object(),
        vision_tower=None,
        data_args=SimpleNamespace(val_split_ratio=0.2, multi_task_training=False),
        model_args=SimpleNamespace(model_type="streamvln_actor"),
        training_args=SimpleNamespace(seed=123),
    )

    assert len(module["train_dataset"]) == 8
    assert len(module["eval_dataset"]) == 2


def test_make_supervised_data_module_marks_actor_subset_for_sequential_sampler(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    class _DummyDataset:
        def __init__(self, *args, **kwargs):
            self.samples = [
                {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": idx}
                for idx in range(10)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {"idx": idx}

    monkeypatch.setattr(train_mod, "StreamVLNActorDataset", _DummyDataset)

    module = train_mod.make_supervised_data_module(
        tokenizer=object(),
        vision_tower=None,
        data_args=SimpleNamespace(
            val_split_ratio=0.2,
            multi_task_training=False,
            use_sequential_subtask_sampler=True,
        ),
        model_args=SimpleNamespace(model_type="streamvln_actor"),
        training_args=SimpleNamespace(seed=123),
    )

    assert getattr(module["train_dataset"], "use_sequential_subtask_sampler", False) is True


def test_prepare_streamvln_actor_special_tokens_adds_next_without_shrinking_vocab(monkeypatch, tmp_path):
    import sys
    import torch

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    class _DummyTokenizer:
        def __init__(self):
            self._size = 151646
            self.added = []

        def __len__(self):
            return self._size

        def add_tokens(self, tokens, special_tokens=True):
            added = 0
            for token in tokens:
                if token not in self.added:
                    self.added.append(token)
                    self._size += 1
                    added += 1
            return added

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_embeddings = torch.nn.Embedding(152064, 8)
            self.resize_calls = []

        def get_input_embeddings(self):
            return self.input_embeddings

        def resize_token_embeddings(self, new_size):
            self.resize_calls.append(int(new_size))

    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    train_mod.prepare_streamvln_actor_special_tokens(model, tokenizer)

    assert "<next>" in tokenizer.added
    assert model.resize_calls == []


def test_get_actor_lora_modules_to_save_uses_gru_head_when_enabled(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    assert train_mod.get_actor_lora_modules_to_save(SimpleNamespace(use_gru_progress=True)) == ["gru_progress_head"]
    assert train_mod.get_actor_lora_modules_to_save(SimpleNamespace(use_gru_progress=False)) == [
        "progress_head",
        "done_head",
    ]


def test_get_peft_state_maybe_zero_3_keeps_modules_to_save(monkeypatch, tmp_path):
    import sys
    import torch

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    named_params = [
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", torch.nn.Parameter(torch.ones(1))),
        ("base_model.model.progress_head.modules_to_save.default.0.weight", torch.nn.Parameter(torch.full((1,), 2.0))),
        ("base_model.model.progress_head.modules_to_save.default.0.bias", torch.nn.Parameter(torch.full((1,), 3.0))),
    ]

    state = train_mod.get_peft_state_maybe_zero_3(named_params, bias="none")

    assert "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight" in state
    assert "base_model.model.progress_head.modules_to_save.default.0.weight" in state
    assert "base_model.model.progress_head.modules_to_save.default.0.bias" in state


def test_get_peft_state_non_lora_maybe_zero_3_excludes_modules_to_save(monkeypatch, tmp_path):
    import sys
    import torch

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    regular = torch.nn.Parameter(torch.ones(1))
    regular.requires_grad_(True)
    modules_to_save = torch.nn.Parameter(torch.ones(1))
    modules_to_save.requires_grad_(True)
    named_params = [
        ("base_model.model.mm_projector.weight", regular),
        ("base_model.model.progress_head.modules_to_save.default.0.weight", modules_to_save),
    ]

    state = train_mod.get_peft_state_non_lora_maybe_zero_3(named_params)

    assert "base_model.model.mm_projector.weight" in state
    assert "base_model.model.progress_head.modules_to_save.default.0.weight" not in state


def test_streamvln_actor_trainer_uses_sequential_sampler_when_enabled(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["streamvln_train.py", "--model_name_or_path", str(tmp_path)],
    )

    import streamvln.streamvln_train as train_mod

    trainer = object.__new__(train_mod.StreamVLNActorTrainer)
    trainer.args = SimpleNamespace(world_size=1, process_index=0, seed=123, data_seed=None)
    trainer.train_dataset = SimpleNamespace(
        use_sequential_subtask_sampler=True,
        samples=[
            {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 1},
            {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 0},
            {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 2},
        ],
    )

    sampler = train_mod.StreamVLNActorTrainer._get_train_sampler(trainer)

    assert isinstance(sampler, train_mod.SequentialSubtaskSampler)
    sampled = list(iter(sampler))
    assert sampled in ([1, 0, 2], [2, 1, 0])


def test_streamvln_actor_trainer_logs_aux_losses():
    import torch

    import streamvln.streamvln_train as train_mod

    trainer = object.__new__(train_mod.StreamVLNActorTrainer)
    trainer.args = SimpleNamespace(logging_steps=1)
    trainer.state = SimpleNamespace(global_step=0)
    logged = {}
    trainer.log = logged.update

    class _DummyModel:
        def __call__(self, **inputs):
            return {
                "loss": torch.tensor(3.0),
                "progress_loss": torch.tensor(0.25),
                "done_loss": torch.tensor(0.75),
            }

    loss = train_mod.StreamVLNActorTrainer.compute_loss(trainer, _DummyModel(), inputs={}, return_outputs=False)

    assert float(loss.item()) == 3.0
    assert logged["train/progress_loss"] == 0.25
    assert logged["train/done_loss"] == 0.75


def test_streamvln_actor_trainer_threads_gru_hidden_across_matching_sequence_keys():
    import torch

    import streamvln.streamvln_train as train_mod

    trainer = object.__new__(train_mod.StreamVLNActorTrainer)
    trainer.args = SimpleNamespace(logging_steps=100)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.log = lambda metrics: None
    trainer._reset_recurrent_gru_state()

    seen_hidden = []

    class _DummyModel:
        use_gru_progress = True

        def __call__(self, **inputs):
            seen_hidden.append(inputs.get("gru_hidden"))
            hidden = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            return {
                "loss": torch.tensor(3.0),
                "gru_hidden_out": hidden,
            }

    first_loss = train_mod.StreamVLNActorTrainer.compute_loss(
        trainer,
        _DummyModel(),
        inputs={"recurrent_sequence_keys": ["ep1::1/2"]},
        return_outputs=False,
    )
    second_loss = train_mod.StreamVLNActorTrainer.compute_loss(
        trainer,
        _DummyModel(),
        inputs={"recurrent_sequence_keys": ["ep1::1/2"]},
        return_outputs=False,
    )

    assert float(first_loss.item()) == 3.0
    assert float(second_loss.item()) == 3.0
    assert seen_hidden[0] is None
    assert torch.equal(seen_hidden[1], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert trainer._gru_recurrent_sequence_key == "ep1::1/2"
    assert torch.equal(trainer._gru_recurrent_hidden, torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_streamvln_actor_trainer_resets_gru_hidden_when_sequence_key_changes():
    import torch

    import streamvln.streamvln_train as train_mod

    trainer = object.__new__(train_mod.StreamVLNActorTrainer)
    trainer.args = SimpleNamespace(logging_steps=100)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.log = lambda metrics: None
    trainer._reset_recurrent_gru_state()

    seen_hidden = []

    class _DummyModel:
        use_gru_progress = True

        def __call__(self, **inputs):
            seen_hidden.append(inputs.get("gru_hidden"))
            return {
                "loss": torch.tensor(2.0),
                "gru_hidden_out": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
            }

    train_mod.StreamVLNActorTrainer.compute_loss(
        trainer,
        _DummyModel(),
        inputs={"recurrent_sequence_keys": ["ep1::1/2"]},
        return_outputs=False,
    )
    train_mod.StreamVLNActorTrainer.compute_loss(
        trainer,
        _DummyModel(),
        inputs={"recurrent_sequence_keys": ["ep1::2/2"]},
        return_outputs=False,
    )

    assert seen_hidden[0] is None
    assert seen_hidden[1] is None
    assert trainer._gru_recurrent_sequence_key == "ep1::2/2"
