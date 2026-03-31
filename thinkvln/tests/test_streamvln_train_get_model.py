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
