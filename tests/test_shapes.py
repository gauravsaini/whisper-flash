import torch
from transformers import WhisperForConditionalGeneration
from whisper_flash.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from whisper_flash.generate import whisper_dflash_generate

def test_shapes():
    # Instantiate draft model
    config = WhisperDFlashConfig(d_draft=256, num_layers=2)
    draft_model = WhisperDFlashDraftModel(config)
    print(f"Draft model instantiated. Params: {draft_model.num_parameters:,}")

    # Dummy inputs
    batch, B, d_target, ctx_len = 2, 8, 1280, 5
    num_taps = len(config.target_layer_ids)
    
    noise_embedding = torch.randn(batch, B, d_target)
    target_hidden = torch.randn(batch, ctx_len, num_taps * d_target)
    audio_summary = torch.randn(batch, 1, d_target)
    position_ids = torch.arange(B).unsqueeze(0).expand(batch, -1)

    # Forward pass
    output = draft_model(
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        audio_summary=audio_summary,
        position_ids=position_ids
    )
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch, B, d_target), "Unexpected output shape"
    print("Shape test passed!")

if __name__ == '__main__':
    test_shapes()
