import pytest
import torch
from transformers import Qwen2Config


def test_progress_loss_helper_uses_l1_on_sigmoid_predictions():
    import streamvln.model.stream_video_vln as model_mod

    progress_preds = torch.tensor([0.1, 0.7], dtype=torch.float32)
    progress_labels = torch.tensor([0.4, 0.1], dtype=torch.float32)

    progress_loss = model_mod.compute_masked_progress_loss(progress_preds, progress_labels)

    assert progress_loss is not None
    assert torch.allclose(progress_loss, torch.tensor(0.45, dtype=torch.float32))


def test_aux_position_resolver_uses_prompt_boundary_for_training_and_prompt_only_inference():
    import streamvln.model.stream_video_vln as model_mod

    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    train_labels = torch.tensor([[-100, -100, 7, 8]], dtype=torch.long)
    prompt_only_labels = torch.tensor([[-100, -100, -100, -100]], dtype=torch.long)

    train_positions = model_mod.StreamVLNForCausalLM._resolve_aux_positions(
        attention_mask=attention_mask,
        labels=train_labels,
    )
    prompt_only_positions = model_mod.StreamVLNForCausalLM._resolve_aux_positions(
        attention_mask=attention_mask,
        labels=prompt_only_labels,
    )

    assert train_positions.tolist() == [1]
    assert prompt_only_positions.tolist() == [3]


def test_aux_pooling_uses_prompt_boundary_when_labels_present():
    import streamvln.model.stream_video_vln as model_mod

    hidden_states = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 7, 8]], dtype=torch.long)

    pooled = model_mod.StreamVLNForCausalLM._pool_aux_hidden(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 20.0]))


def test_aux_pooling_uses_prompt_boundary_for_prompt_only_sequences():
    import streamvln.model.stream_video_vln as model_mod

    hidden_states = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, -100]], dtype=torch.long)

    pooled = model_mod.StreamVLNForCausalLM._pool_aux_hidden(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0], torch.tensor([4.0, 40.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for bf16 backward coverage")
def test_progress_loss_helper_keeps_bfloat16_backward_stable():
    import streamvln.model.stream_video_vln as model_mod

    progress_logits = torch.randn(4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    progress_preds = torch.sigmoid(progress_logits)
    progress_labels = torch.rand(4, device="cuda", dtype=torch.float32)

    progress_loss = model_mod.compute_masked_progress_loss(progress_preds, progress_labels)

    assert progress_loss is not None
    assert progress_loss.dtype == torch.bfloat16
    progress_loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GRU bf16 coverage")
def test_gru_progress_head_handles_bfloat16_module_without_nan():
    import streamvln.model.stream_video_vln as model_mod

    head = model_mod.GRUProgressHead(input_size=8, hidden_size=4).cuda().to(torch.bfloat16)
    pooled_hidden = torch.randn(2, 8, device="cuda", dtype=torch.float32)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        progress_preds, done_logits, new_hidden = head(pooled_hidden)

    assert torch.isfinite(progress_preds).all()
    assert torch.isfinite(done_logits).all()
    assert torch.isfinite(new_hidden).all()


def test_gru_progress_head_reset_recurrent_parameters_restores_finite_state():
    import streamvln.model.stream_video_vln as model_mod

    head = model_mod.GRUProgressHead(input_size=8, hidden_size=4)
    with torch.no_grad():
        head.gru.bias_ih.fill_(float("inf"))
        head.gru.bias_hh.fill_(float("nan"))

    head.reset_recurrent_parameters()

    for param in head.gru.parameters():
        assert torch.isfinite(param).all()


def test_streamvln_model_initializes_gru_recurrent_parameters(monkeypatch):
    import streamvln.model.stream_video_vln as model_mod
    import llava.model.llava_arch as llava_arch

    class _DummyVisionTower(torch.nn.Identity):
        def __init__(self):
            super().__init__()
            self.config = Qwen2Config(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
            )

    monkeypatch.setattr(llava_arch, "build_vision_tower", lambda *args, **kwargs: _DummyVisionTower())
    monkeypatch.setattr(llava_arch, "build_vision_projector", lambda *args, **kwargs: torch.nn.Identity())

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    config.mm_vision_tower = "dummy"
    config.use_gru_progress = True
    config.progress_num_bins = 0
    config.num_history = 6
    config.num_future_steps = 4

    model = model_mod.StreamVLNForCausalLM(config)

    for param in model.gru_progress_head.gru.parameters():
        assert torch.isfinite(param).all()


def test_ensure_gru_recurrent_parameters_repairs_non_finite_gru_state():
    import streamvln.model.stream_video_vln as model_mod

    holder = type("Holder", (), {})()
    holder.use_gru_progress = True
    holder.gru_progress_head = model_mod.GRUProgressHead(input_size=8, hidden_size=4)
    with torch.no_grad():
        holder.gru_progress_head.gru.weight_ih.zero_()
        holder.gru_progress_head.gru.weight_hh.zero_()
        holder.gru_progress_head.gru.bias_ih.fill_(float("inf"))
        holder.gru_progress_head.gru.bias_hh.fill_(float("nan"))

    model_mod.StreamVLNForCausalLM.ensure_gru_recurrent_parameters(holder)

    for param in holder.gru_progress_head.gru.parameters():
        assert torch.isfinite(param).all()
    assert torch.count_nonzero(holder.gru_progress_head.gru.weight_ih).item() > 0
