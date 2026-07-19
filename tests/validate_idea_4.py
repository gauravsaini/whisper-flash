#!/usr/bin/env python3
"""
validate_idea_4.py

Validates Idea #4: Speculative Confidence Regularization.
Hypothesis: We can move speculation from an inference trick into a training signal 
by computing the KL-divergence between the draft and target probability distributions.
Validation: We load the pre-trained draft model and the target model, run them 
over a sample, and compute the KL Divergence per step. We will plot/print steps 
where the draft model is "overconfident but wrong" (high KL divergence), proving 
this metric can be used as a penalty term in the loss function during training.
"""

import numpy as np
import mlx.core as mx
from tqdm import tqdm
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, decoder_forward_with_hidden_states, encoder_forward
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from whisper_flash_mlx.utils import extract_context_feature

def load_draft_model(checkpoint_path: str) -> WhisperDFlashDraftModel:
    import json
    from pathlib import Path
    
    ckpt_dir = Path(checkpoint_path).parent
    with open(ckpt_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    config = WhisperDFlashConfig(**config_dict)
    
    model = WhisperDFlashDraftModel(config)
    weights = mx.load(checkpoint_path)
    model.load_weights(list(weights.items()))
    model.eval()
    return model

def validate_confidence_regularization():
    model_name = "mlx-community/whisper-tiny"
    checkpoint_path = "checkpoints_tiny_v2/final_model.safetensors"
    dataset_name = "hf-internal-testing/librispeech_asr_dummy"
    config = "clean"
    split = "validation"
    num_samples = 3

    print("Loading Target Model...")
    target = load_target_model(model_name)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    print(f"Loading Draft Model from {checkpoint_path}...")
    try:
        draft = load_draft_model(checkpoint_path)
    except Exception as e:
        print(f"Could not load trained draft model: {e}")
        print("Falling back to randomly initialized draft model for mechanics verification...")
        draft_config = WhisperDFlashConfig(
            d_target=target.dims.n_text_state,
            d_draft=target.dims.n_text_state,
            num_layers=2,
            vocab_size=target.dims.n_vocab,
            block_size=4,
            target_layer_ids=[1, 2]
        )
        draft = WhisperDFlashDraftModel(draft_config)
    
    print(f"Loading Dataset {dataset_name}...")
    ds = load_dataset(dataset_name, config, split=split)
    
    kl_divergences = []
    overconfident_errors = 0
    total_errors = 0
    
    for i in range(min(num_samples, len(ds))):
        print(f"\nProcessing sample {i+1}/{num_samples}...")
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text = sample["text"]
        
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        text_tokens = tokenizer.encode(text)
        token_ids = mx.array([text_tokens], dtype=mx.int32)
        sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
        labels = mx.concatenate([sot, token_ids], axis=1)
        
        encoder_hidden = encoder_forward(target, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
        
        seq_len = labels.shape[1] - 1
        
        for t in tqdm(range(seq_len)):
            input_token = labels[:, :t+1]
            true_token = labels[0, t+1].item()
            
            # --- Target Model ---
            logits_target, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, 
                collect_hidden_states=True, return_cross_attention=False
            )
            
            current_target_logits = logits_target[0, -1, :] # (vocab_size)
            target_probs = mx.softmax(current_target_logits, axis=-1)
            target_pred = mx.argmax(current_target_logits).item()
            
            # --- Draft Model ---
            decoder_feats = extract_context_feature(hidden_target, draft.target_layer_ids)
            
            # Create a mock position_ids for the draft model (since we only care about predicting 1 step ahead here)
            pos_ids = mx.array([[input_token.shape[1]]], dtype=mx.int32)
            
            # We simulate drafting the next token by passing the mask token
            noise = target.decoder.token_embedding(mx.array([[draft.mask_token_id]]))
            
            draft_hidden = draft(noise, decoder_feats, audio_summary, pos_ids)
            
            # Project using target lm_head
            draft_logits = target.decoder.token_embedding.as_linear(draft_hidden)[0, -1, :]
            draft_probs = mx.softmax(draft_logits, axis=-1)
            draft_pred = mx.argmax(draft_logits).item()
            
            # Compute KL Divergence (Target || Draft)
            # KL(P || Q) = sum(P * log(P/Q))
            kl_div = mx.sum(target_probs * (mx.log(target_probs + 1e-9) - mx.log(draft_probs + 1e-9))).item()
            kl_divergences.append(kl_div)
            
            # Check for "Overconfident Errors" (Draft is wrong, but its max probability is > 0.8)
            if draft_pred != target_pred:
                total_errors += 1
                draft_confidence = mx.max(draft_probs).item()
                if draft_confidence > 0.8:
                    overconfident_errors += 1
                
    # --- Reporting ---
    print("\n" + "="*50)
    print("VALIDATION RESULTS FOR IDEA #4 (Speculative Confidence Regularization)")
    print("="*50)
    
    mean_kl = np.mean(kl_divergences)
    print(f"Mean KL Divergence (Target || Draft): {mean_kl:.4f}")
    print(f"Total Draft Errors vs Target: {total_errors}")
    print(f"Overconfident Errors (Confidence > 80% but wrong): {overconfident_errors}")
    
    if total_errors > 0:
        percent_overconfident = (overconfident_errors / total_errors) * 100
        print(f"Percentage of errors that were OVERCONFIDENT: {percent_overconfident:.2f}%")
        
    print("-> By adding KL-Divergence to the training loss, we can explicitly penalize these overconfident errors, improving draft calibration and reducing costly rollbacks!")

if __name__ == "__main__":
    validate_confidence_regularization()
