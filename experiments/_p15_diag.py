import time, numpy as np, soundfile as sf
import mlx.core as mx
from datasets import load_dataset as hf_load
from whisper_flash_mlx.production import ProductionConfig, GreedyDecoder

MODEL = "mlx-community/whisper-large-v3-turbo"
ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sr=16000; silence=np.zeros(int(0.25*sr), dtype=np.float32); target=int(30*sr)
chunks=[]; cur=0; i=0
while cur<target and i<len(ds):
    a=ds[i]["audio"]; arr=np.array(a["array"],dtype=np.float32)
    if arr.ndim==2: arr=arr.mean(axis=1)
    chunks.append(arr); chunks.append(silence); cur+=len(arr)+len(silence); i+=1
audio=np.concatenate(chunks)[:target]
sf.write("experiments/p15_audio/diag_long.wav", audio, sr)

for name, cfg in [
    ("BASELINE noKV", ProductionConfig(model_path=MODEL, quantize=False, encoder_stride=1, kv_compress=False, use_kv_cache=False)),
    ("KV",            ProductionConfig(model_path=MODEL, quantize=False, encoder_stride=1, kv_compress=False, use_kv_cache=True)),
    ("FULL lossless", ProductionConfig(model_path=MODEL, quantize=True,  encoder_stride=2, kv_compress=False, use_kv_cache=True)),
    ("FULL literal",  ProductionConfig(model_path=MODEL, quantize=True,  encoder_stride=2, kv_compress=True,  use_kv_cache=True)),
]:
    dec = GreedyDecoder(cfg)
    r = dec.decode("experiments/p15_audio/diag_long.wav")
    print(f"{name}: steps={r.n_decoder_steps} time={r.wall_time_s:.3f}s text={r.text[:70]!r}")
