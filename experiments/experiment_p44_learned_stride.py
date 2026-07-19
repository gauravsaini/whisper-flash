"""P44: Learned Stride Conv vs Avg-Pool

This experiment tests if we can replace simple average-pooling downsampling
with a learned 1D Convolution downsampler. We train the downsampler
parameters while keeping the rest of the Whisper model frozen, optimizing
cross-entropy loss under teacher forcing on the LibriSpeech dummy dataset.

We test if learned downsampling allows us to push the sequence reduction factor
beyond 4x (e.g. 8x or 16x) losslessly.
"""
import time
import json
import sys
import os
import tempfile
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_whisper
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer
import pyarrow as pa
from jiwer import wer as _wer

DUMMY_ARROW = "/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"
LONG_WAV = "/tmp/flashbench/long.wav"

# --- Conv Downsampler Module ---
class ConvDownsampler(nn.Module):
    def __init__(self, d_model, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=stride,
            stride=stride,
            bias=True
        )
        # Initialize to average pooling
        w = np.zeros((d_model, stride, d_model), dtype=np.float32)
        for o in range(d_model):
            w[o, :, o] = 1.0 / stride
        self.conv.weight = mx.array(w)
        self.conv.bias = mx.zeros((d_model,))
        
    def __call__(self, x):
        return self.conv(x)

# --- Learned Encoder Wrapper ---
class LearnedEncoderWrapper(nn.Module):
    def __init__(self, enc, downsampler):
        super().__init__()
        object.__setattr__(self, "_enc", enc)
        object.__setattr__(self, "_downsampler", downsampler)
        
    def __call__(self, x):
        o = self._enc(x)
        return self._downsampler(o)
        
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_enc"), name)

# --- Average Pool Wrapper (Baseline) ---
class AvgPoolWrapper(nn.Module):
    def __init__(self, enc, stride):
        super().__init__()
        object.__setattr__(self, "_enc", enc)
        object.__setattr__(self, "_stride", stride)
        
    def __call__(self, x):
        o = self._enc(x)
        if self._stride > 1:
            B, T, D = o.shape
            Tt = (T // self._stride) * self._stride
            o = o[:, :Tt, :]
            o = mx.mean(o.reshape(B, Tt // self._stride, self._stride, D), axis=2)
        return o
        
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_enc"), name)

# --- Helper functions ---
def load_dummy_rows():
    with pa.memory_map(DUMMY_ARROW) as src:
        try:
            t = pa.ipc.open_file(src).read_all()
        except Exception:
            t = pa.ipc.open_stream(src).read_all()
    return t.to_pylist()

def prepare_data(model, tokenizer, rows):
    from mlx_whisper.audio import log_mel_spectrogram
    data_list = []
    
    # SOT prefix tokens for English transcription
    sot_prefix = [tokenizer.sot, tokenizer.transcribe, tokenizer.no_timestamps]
    eot_id = tokenizer.eot
    
    print("Pre-extracting encoder features and tokens...")
    for idx, r in enumerate(rows):
        audio_bytes = r["audio"]["bytes"]
        text = r["text"].strip().lower()
        
        # 1. Mel spectrogram
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            import soundfile as sf
            audio_data, sr = sf.read(path)
            mel = log_mel_spectrogram(audio_data.astype(np.float32), n_mels=80)
            mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
            mel = mx.array(mel)[None]
        finally:
            os.unlink(path)
            
        # 2. Run encoder
        enc_out = model.encoder(mel)
        mx.eval(enc_out)
        
        # 3. Target tokens
        toks = sot_prefix + tokenizer.encode(" " + text) + [eot_id]
        toks_arr = mx.array(toks)[None] # (1, seq_len)
        
        data_list.append((enc_out, toks_arr, text))
        if (idx + 1) % 10 == 0 or idx + 1 == len(rows):
            print(f"  Processed {idx + 1}/{len(rows)}")
            
    return data_list

# --- Custom decoder forward pass ---
def custom_decoder_forward(model, tokens, audio_features):
    decoder = model.decoder
    x = (
        decoder.token_embedding(tokens)
        + decoder.positional_embedding[:tokens.shape[-1]]
    )
    for block in decoder.blocks:
        x, _, _ = block(x, audio_features, mask=decoder._mask)
    x = decoder.ln(x)
    logits = decoder.token_embedding.as_linear(x)
    return logits

# --- Loss Function ---
def loss_fn(model, downsampler, enc_out, target_tokens):
    down_out = downsampler(enc_out)
    # Teacher forcing: feed tokens[:-1], predict tokens[1:]
    logits = custom_decoder_forward(model, target_tokens[:, :-1], down_out)
    loss = mx.mean(nn.losses.cross_entropy(logits, target_tokens[:, 1:]))
    return loss

def train_downsampler(model, downsampler, train_data, epochs=15, lr=1e-4):
    optimizer = optim.Adam(learning_rate=lr)
    
    # Compile loss & grad function
    loss_and_grad = nn.value_and_grad(downsampler, lambda m_down, enc, tar: loss_fn(model, m_down, enc, tar))
    
    print(f"Training learned downsampler (lr={lr}, epochs={epochs})...")
    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_losses = []
        # Shuffle training data
        np.random.shuffle(train_data)
        
        for enc_out, target_tokens, _ in train_data:
            loss, grads = loss_and_grad(downsampler, enc_out, target_tokens)
            optimizer.update(downsampler, grads)
            mx.eval(downsampler.parameters(), loss)
            epoch_losses.append(loss.item())
            
        dt = time.perf_counter() - t0
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {np.mean(epoch_losses):.5f} | Time: {dt:.3f}s")

# --- Evaluation helper ---
_GLOBAL_MODEL = None
def evaluate_wer(repo, encoder_wrapper, val_rows):
    global _GLOBAL_MODEL
    model = load_model(repo, mx.float16)
    model.encoder = encoder_wrapper(model.encoder)
    mx.eval(model.parameters())
    _GLOBAL_MODEL = model
    
    real_load = mlx_whisper.load_models.load_model
    mlx_whisper.load_models.load_model = lambda *a, **k: _GLOBAL_MODEL
    
    hyps = []
    refs = [r["text"].strip().lower() for r in val_rows]
    
    try:
        for r in val_rows:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(r["audio"]["bytes"])
                path = f.name
            try:
                out = mlx_whisper.transcribe(path, fp16=True, verbose=False, condition_on_previous_text=False)
                hyps.append(out["text"].strip().lower())
            finally:
                os.unlink(path)
    finally:
        mlx_whisper.load_models.load_model = real_load
        del model
        _GLOBAL_MODEL = None
        mx.clear_cache()
        
    w = float(np.mean([_wer(refs[i], hyps[i]) for i in range(len(val_rows))]))
    return w

def run_long_timing(repo, encoder_wrapper):
    global _GLOBAL_MODEL
    model = load_model(repo, mx.float16)
    model.encoder = encoder_wrapper(model.encoder)
    mx.eval(model.parameters())
    _GLOBAL_MODEL = model
    
    real_load = mlx_whisper.load_models.load_model
    mlx_whisper.load_models.load_model = lambda *a, **k: _GLOBAL_MODEL
    
    try:
        t0 = time.perf_counter()
        mlx_whisper.transcribe(LONG_WAV, fp16=True, verbose=False, condition_on_previous_text=False)
        dt = time.perf_counter() - t0
    finally:
        mlx_whisper.load_models.load_model = real_load
        del model
        mx.clear_cache()
    return dt

def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-tiny-mlx"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    print(f"Target model: {repo}")
    
    # 1. Load tokenizer and baseline model for pre-extraction
    tokenizer = get_tokenizer(multilingual=False)
    base_model = load_model(repo, mx.float16)
    
    # Load dataset
    rows = load_dummy_rows()
    print(f"Total rows in dataset: {len(rows)}")
    
    # Pre-extract encoder outputs and tokens
    data_list = prepare_data(base_model, tokenizer, rows)
    
    # Split into train/validation
    # We use 50 samples for training, 23 for validation
    train_data = data_list[:50]
    val_rows = rows[50:]
    print(f"Split: {len(train_data)} train samples, {len(val_rows)} validation samples")
    
    # Clean up base model to free memory
    del base_model
    mx.clear_cache()
    
    # Check LONG_WAV existence
    if not os.path.exists(LONG_WAV):
        print(f"LONG_WAV not found at {LONG_WAV}. Constructing temporary long clip...")
        os.makedirs(os.path.dirname(LONG_WAV), exist_ok=True)
        import soundfile as sf
        long_audio_data = []
        for r in rows[:25]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(r["audio"]["bytes"])
                path = f.name
            data, sr = sf.read(path)
            long_audio_data.append(data)
            os.unlink(path)
        concatenated = np.concatenate(long_audio_data)
        sf.write(LONG_WAV, concatenated, sr)
        print(f"Created concatenated long clip: {len(concatenated)/sr:.2f}s")
        
    # We sweep stride factors: 4, 8, 16
    strides = [4, 8, 16]
    results = {}
    
    # Get FP16 baseline (stride=1) timing and WER
    print("\n--- Running FP16 Baseline (Stride 1) ---")
    base_wer = evaluate_wer(repo, lambda enc: enc, val_rows)
    base_time = run_long_timing(repo, lambda enc: enc)
    print(f"Baseline FP16 S1: WER={base_wer:.4f} | Time={base_time:.3f}s")
    
    results["s1_baseline"] = {
        "wer": round(base_wer, 4),
        "long_time_s": round(base_time, 3),
        "speedup": 1.0,
        "lossless": True
    }
    
    d_model = 384 if "tiny" in repo else 1280
    
    for s in strides:
        print(f"\n==================================================")
        print(f"Evaluating Stride Factor: {s}x")
        print(f"==================================================")
        
        # 1. Avg-Pool Baseline
        avg_wrapper = lambda enc: AvgPoolWrapper(enc, s)
        avg_wer = evaluate_wer(repo, avg_wrapper, val_rows)
        avg_time = run_long_timing(repo, avg_wrapper)
        print(f"Avg-Pool: WER={avg_wer:.4f} | Time={avg_time:.3f}s | Speedup={base_time/avg_time:.3f}x")
        
        # 2. Learned Downsampler
        downsampler = ConvDownsampler(d_model=d_model, stride=s)
        mx.eval(downsampler.parameters())
        
        # Train downsampler
        train_downsampler(model=load_model(repo, mx.float16), downsampler=downsampler, train_data=train_data, epochs=epochs, lr=1e-4)
        
        # Evaluate Learned Downsampler
        learned_wrapper = lambda enc: LearnedEncoderWrapper(enc, downsampler)
        learned_wer = evaluate_wer(repo, learned_wrapper, val_rows)
        learned_time = run_long_timing(repo, learned_wrapper)
        print(f"Learned:  WER={learned_wer:.4f} | Time={learned_time:.3f}s | Speedup={base_time/learned_time:.3f}x")
        
        results[f"s{s}_avgpool"] = {
            "wer": round(avg_wer, 4),
            "long_time_s": round(avg_time, 3),
            "speedup": round(base_time/avg_time, 3),
            "lossless": bool(avg_wer <= base_wer + 0.01)
        }
        results[f"s{s}_learned"] = {
            "wer": round(learned_wer, 4),
            "long_time_s": round(learned_time, 3),
            "speedup": round(base_time/learned_time, 3),
            "lossless": bool(learned_wer <= base_wer + 0.01)
        }
        
    out = {
        "experiment": "P44: Learned Stride Conv vs Avg-Pool",
        "model": repo,
        "n_train_samples": len(train_data),
        "n_val_samples": len(val_rows),
        "results": results
    }
    
    print("\n" + json.dumps(out, indent=2))
    
    # Save results to file
    out_path = "/Users/ektasaini/Desktop/whisper-flash/results/p44_learned_stride.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
