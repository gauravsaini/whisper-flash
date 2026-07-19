"""P27: Encoder Cross-Attention Downsampling

Goal: Reduce cross-attention memory bandwidth by 2x or 4x without dropping context
mass (which broke P24). We average-pool the encoder frames before passing them to 
the decoder. Since 20ms frames are highly stationary, a 40ms or 80ms pooled frame
might retain all necessary semantic info while shrinking the K/V matrices massively.
"""

import time
import json
from pathlib import Path
import mlx.core as mx
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
MAX_TOKENS = 50

def decode(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    kv_cache = None
    
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache
        )
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache) 
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total

def downsample_encoder(enc, factor=2):
    """Average pooling along the time dimension (axis 1)"""
    B, T, D = enc.shape
    pad_len = (factor - (T % factor)) % factor
    if pad_len > 0:
        pad_frames = mx.repeat(enc[:, -1:, :], pad_len, axis=1)
        enc = mx.concatenate([enc, pad_frames], axis=1)
        
    B, new_T, D = enc.shape
    enc_reshaped = enc.reshape(B, new_T // factor, factor, D)
    return enc_reshaped.mean(axis=2)

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    whisper_model = load_target_model(model_name, dtype=mx.float32)
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=80)
    mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
    mel = mx.array(mel)[None]
    
    enc = encoder_forward(whisper_model, mel)
    
    print("\nRunning Standard Decoding (1500 frames)...")
    std_ids, std_time = decode(whisper_model, enc)
    std_tps = len(std_ids) / std_time
    
    print("\nRunning Downsampled Decoding (Factor=2, 750 frames)...")
    enc_2x = downsample_encoder(enc, factor=2)
    ds2_ids, ds2_time = decode(whisper_model, enc_2x)
    ds2_tps = len(ds2_ids) / ds2_time
    
    print("\nRunning Downsampled Decoding (Factor=4, 375 frames)...")
    enc_4x = downsample_encoder(enc, factor=4)
    ds4_ids, ds4_time = decode(whisper_model, enc_4x)
    ds4_tps = len(ds4_ids) / ds4_time
    
    tokenizer = get_tokenizer(multilingual=False)
    std_text = tokenizer.decode([t for t in std_ids if t < tokenizer.eot])
    ds2_text = tokenizer.decode([t for t in ds2_ids if t < tokenizer.eot])
    ds4_text = tokenizer.decode([t for t in ds4_ids if t < tokenizer.eot])
    
    print(f"\n--- Performance ---")
    print(f"Standard TPS: {std_tps:.1f}")
    print(f"Factor=2 TPS: {ds2_tps:.1f} (Speedup: {ds2_tps/std_tps:.2f}x)")
    print(f"Factor=4 TPS: {ds4_tps:.1f} (Speedup: {ds4_tps/std_tps:.2f}x)")
    
    print(f"\n--- Output ---")
    print(f"Standard Text: {std_text}")
    print(f"Factor=2 Text: {ds2_text}")
    print(f"Factor=4 Text: {ds4_text}")
    
    out_path = Path("results/p27_encoder_downsample.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P27: Encoder Downsampling",
            "std_tps": std_tps,
            "ds2_tps": ds2_tps,
            "ds4_tps": ds4_tps,
            "std_text": std_text,
            "ds2_text": ds2_text,
            "ds4_text": ds4_text
        }, f, indent=2)

if __name__ == "__main__":
    main()
