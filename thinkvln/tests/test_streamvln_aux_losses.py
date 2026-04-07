import pytest
import torch


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
