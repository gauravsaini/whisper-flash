import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper_flash.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from whisper_flash.generate import whisper_dflash_generate
from whisper_flash.evaluate import baseline_generate

def test_correctness():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    model_name = "openai/whisper-tiny"  # Use tiny for fast local testing
    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    target = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    
    # Instantiate draft model
    config = WhisperDFlashConfig(
        d_target=target.config.d_model,
        num_target_layers=target.config.decoder_layers,
        vocab_size=target.config.vocab_size,
        max_target_positions=target.config.max_target_positions,
        block_size=1,  # block_size=1 means no speculation, just target model
    )
    draft_model = WhisperDFlashDraftModel(config).to(device).eval()

    # Create dummy audio
    import numpy as np
    audio = np.random.randn(16000)  # 1 second of noise
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generate baseline
    print("Running baseline...")
    bl_result = baseline_generate(target, inputs, max_new_tokens=20)
    bl_tokens = bl_result["output_ids"][0, 1:].cpu().tolist()
    
    # Generate dflash with block_size=1
    print("Running dflash generate (block_size=1)...")
    df_result = whisper_dflash_generate(
        draft_model, target, inputs, max_new_tokens=20, block_size=1, return_stats=True
    )
    df_tokens = df_result.output_ids[0, df_result.num_prompt_tokens:].cpu().tolist()
    
    print(f"Baseline tokens: {bl_tokens}")
    print(f"DFlash tokens:   {df_tokens}")
    
    assert bl_tokens == df_tokens, "Tokens do not match!"
    print("Correctness test passed! block_size=1 matches standard decoding.")

if __name__ == '__main__':
    test_correctness()
