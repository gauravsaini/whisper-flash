"""P17: Continuous Word-Emission via Contrastive Search

Goal: Test if Whisper's hidden states can directly emit full words instead of
subword tokens by doing a cosine similarity search against pre-computed word embeddings.

If successful, we could decode words in a single step, bypassing multiple autoregressive
subword steps for long words.

Methodology:
1. Extract standard token-by-token hidden states.
2. Compute embeddings for target words by averaging their constituent token embeddings.
3. Check if the hidden state before a word is more similar to the full word embedding
   or just the first token embedding.
4. Attempt to decode using word-level cosine similarity.
"""

import time
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)

EOS_ID, SOT_ID = 50257, 50258

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading {model_name}...")
    model = load_target_model(model_name, dtype=mx.float16)
    tokenizer = get_tokenizer(multilingual=False)
    
    # Get token embedding matrix (Vocab, D)
    W_emb = model.decoder.token_embedding.weight
    
    # 1. Get dummy audio
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=80)
    if mel.shape[0] < 3000:
        mel = np.pad(mel, [(0, 3000 - mel.shape[0]), (0, 0)])
    else:
        mel = mel[:3000, :]
    mel = mx.array(mel)[None]  # (1, 3000, 80)
    
    enc = encoder_forward(model, mel)
    
    # 2. Extract hidden states and greedy tokens
    print("\nExtracting baseline sequence...")
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    output_ids = [SOT_ID]
    hidden_states = []
    
    kv_cache = None
    for _ in range(50):
        logits, kv_cache, h = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache,
            collect_hidden_states=True)
            
        last_h = h[-1]  # (1, 1, D)
        hidden_states.append(last_h)
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
    # Remove SOT and EOS for analysis
    target_tokens = output_ids[1:-1] if output_ids[-1] == EOS_ID else output_ids[1:]
    gt_text = tokenizer.decode(target_tokens)
    print(f"Target text: '{gt_text}'")
    
    # 3. Group tokens into words and compute word embeddings
    # Whisper tokenizer typically prefixes words with a space.
    words = []
    current_word_tokens = []
    current_word_str = ""
    
    for i, tok in enumerate(target_tokens):
        s = tokenizer.decode([tok])
        if s.startswith(" ") and len(current_word_tokens) > 0:
            words.append({
                "text": current_word_str,
                "tokens": current_word_tokens,
                "start_idx": i - len(current_word_tokens)
            })
            current_word_tokens = []
            current_word_str = ""
            
        current_word_tokens.append(tok)
        current_word_str += s
        
    if len(current_word_tokens) > 0:
        words.append({
            "text": current_word_str,
            "tokens": current_word_tokens,
            "start_idx": len(target_tokens) - len(current_word_tokens)
        })
        
    print(f"Extracted {len(words)} words.")
    
    # Compute embeddings
    for w in words:
        toks = mx.array(w["tokens"])
        # Mean of constituent token embeddings
        embs = W_emb[toks]
        word_emb = mx.mean(embs, axis=0)
        w["embedding"] = word_emb
        w["first_tok_emb"] = embs[0]
        
    # 4. Compare hidden state to word embedding vs first token embedding
    # We look at the hidden state EXACTLY before the word starts
    
    results_list = []
    wins_word = 0
    wins_tok = 0
    
    for w in words:
        idx = w["start_idx"]
        z_t = hidden_states[idx][0, 0]  # shape (D,)
        
        # Cosine similarity
        def cos_sim(a, b):
            return mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))
            
        sim_word = cos_sim(z_t, w["embedding"]).item()
        sim_tok = cos_sim(z_t, w["first_tok_emb"]).item()
        
        results_list.append({
            "word": w["text"],
            "num_tokens": len(w["tokens"]),
            "sim_word": sim_word,
            "sim_first_tok": sim_tok
        })
        
        if len(w["tokens"]) > 1:
            if sim_word > sim_tok:
                wins_word += 1
            else:
                wins_tok += 1
                
    for r in results_list:
        print(f"Word: {r['word']:<15} | len={r['num_tokens']} | sim_word={r['sim_word']:.3f} | sim_tok={r['sim_first_tok']:.3f}")
        
    print("\n--- Multi-Token Word Summary ---")
    print(f"Word Embedding won: {wins_word}")
    print(f"First Token Embedding won: {wins_tok}")
    
    out_path = Path("results/p17_word_emission.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P17: Continuous Word Emission",
            "words": results_list,
            "wins_word": wins_word,
            "wins_tok": wins_tok
        }, f, indent=2)

if __name__ == "__main__":
    main()
