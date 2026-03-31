import pytest
import torch


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
