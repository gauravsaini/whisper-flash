"""P40: Fused lossless stack (stride-4 + Q8) on the NATIVE mlx_whisper path.

The synthesis (P37/P38) showed the fair lossless speedup on the native path is
~1.1-1.4x. The two validated LOSSLESS levers are:
  - encoder stride downsample (lossless to stride-4 on large-v3-turbo, P38)
  - Q8 quantization (lossless on both tiny and large, P30/production.py)

This experiment fuses them at the AGGRESSIVE end (stride-4 + Q8) and measures
the combined lossless speedup vs fp16 baseline on:
  - WER over the FULL 73-sample dummy validation set (not just 8)
  - wall-clock on a 481s long clip

It also sweeps stride x quant to find the best lossless point.
"""
import time, json, sys, os, io, tempfile
import numpy as np
import soundfile as sf
import mlx.core as mx, mlx.nn as nn, mlx_whisper
from mlx_whisper.load_models import load_model
import pyarrow as pa
from jiwer import wer as _wer

DUMMY_ARROW="/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"
LONG="/tmp/flashbench/long.wav"

class EncStride(nn.Module):
    def __init__(s,enc,st):
        super().__init__(); object.__setattr__(s,"_enc",enc); object.__setattr__(s,"_st",st)
    def __call__(s,x):
        o=s._enc(x)
        if s._st>1:
            B,T,D=o.shape; Tt=(T//s._st)*s._st; o=o[:,:Tt,:]
            o=mx.mean(o.reshape(B,Tt//s._st,s._st,D),axis=2)
        return o
    def __getattr__(s,n): return getattr(object.__getattribute__(s,"_enc"),n)

def quant(m,b):
    m.apply_to_modules(lambda nm,mod: nn.QuantizedLinear.from_linear(mod,64 if mod.weight.shape[-1]%64==0 else 32,b,"affine") if isinstance(mod,nn.Linear) else mod)

G=None
def transcribe_audio(audio, repo, stride, q8):
    global G
    m=load_model(repo,mx.float16)
    if stride>1: m.encoder=EncStride(m.encoder,stride)
    if q8: quant(m,8)
    mx.eval(m.parameters()); G=m
    real=mlx_whisper.load_models.load_model
    mlx_whisper.load_models.load_model=lambda *a,**k: G
    try:
        out=mlx_whisper.transcribe(audio,fp16=True,verbose=False,condition_on_previous_text=False)
    finally:
        mlx_whisper.load_models.load_model=real; del m; mx.clear_cache()
    return out["text"]

def load_dummy(n=None):
    with pa.memory_map(DUMMY_ARROW) as src:
        try: t=pa.ipc.open_file(src).read_all()
        except: t=pa.ipc.open_stream(src).read_all()
    rows=t.to_pylist() if n is None else t.to_pylist()[:n]
    return rows

def main():
    repo=sys.argv[1] if len(sys.argv)>1 else "mlx-community/whisper-large-v3-turbo"
    n_wer=int(sys.argv[2]) if len(sys.argv)>2 else 73
    rows=load_dummy(n_wer)
    refs=[r["text"].strip().lower() for r in rows]
    sweep=[(1,False),(2,False),(4,False),(1,True),(2,True),(4,True)]
    wer_res={}; base=None; base_long=None
    for s,q in sweep:
        # load model ONCE per config
        m=load_model(repo,mx.float16)
        if s>1: m.encoder=EncStride(m.encoder,s)
        if q: quant(m,8)
        mx.eval(m.parameters()); 
        global G; G=m
        real=mlx_whisper.load_models.load_model
        mlx_whisper.load_models.load_model=lambda *a,**k: G
        try:
            hyps=[]
            for r in rows:
                with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
                    f.write(r["audio"]["bytes"]); p=f.name
                try:
                    h=mlx_whisper.transcribe(p,fp16=True,verbose=False,condition_on_previous_text=False)["text"].strip().lower()
                finally:
                    os.unlink(p)
                hyps.append(h)
        finally:
            mlx_whisper.load_models.load_model=real
        w=float(np.mean([_wer(refs[i],hyps[i]) for i in range(len(rows))]))
        long_t = 0.0
        if os.environ.get("DO_LONG") == "1":
            mlx_whisper.load_models.load_model=lambda *a,**k: G
            lt=time.perf_counter(); mlx_whisper.transcribe(LONG,fp16=True,verbose=False,condition_on_previous_text=False); long_t=time.perf_counter()-lt
            mlx_whisper.load_models.load_model=real
        del m; mx.clear_cache()
        if base is None: base=w; base_long=long_t if long_t>0 else 1.0
        sp = round(base_long/long_t,3) if long_t>0 else None
        wer_res[f"s{s}_{'q8' if q else 'fp16'}"]=dict(wer=round(w,4), lossless=bool(w<=base+0.01), long_time_s=round(long_t,3) if long_t>0 else None, speedup=sp)
        print(f"s{s} {'q8' if q else 'fp16'}: wer={w:.4f}" + (f" long={long_t:.2f}s speedup={sp}x" if long_t>0 else ""), flush=True)
    out=dict(experiment="P40 fused lossless stack (native path)", model=repo, n_wer=n_wer, results=wer_res)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
