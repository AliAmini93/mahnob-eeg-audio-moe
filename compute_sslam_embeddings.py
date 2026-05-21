# compute_sslam_embeddings.py

import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

BASE_DIR = os.environ.get("MAHNOB_BASE_DIR", "/path/to/HCI_Tagging_Database")

AUDIO_10SEC_DIR = os.path.join(BASE_DIR, "Continuous_Audio_32k", "10sec")
AUDIO_X_PATH = os.path.join(AUDIO_10SEC_DIR, "Cont_Audio_10sec_X_32k.npy")
EMB_PATH = os.path.join(AUDIO_10SEC_DIR, "audio_emb_sslam_10sec.npy")

SSLAM_MODEL_ID = "ta012/SSLAM_pretrain"

BATCH_SIZE = 16
TARGET_SR = 16000
ORIG_SR = 32000
TARGET_LENGTH = 1024
NUM_MEL_BINS = 128
PRINT_EVERY_SEC = 20.0

print("torch.cuda.is_available():", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print(f"Loading SSLAM model from: {SSLAM_MODEL_ID}")
model = AutoModel.from_pretrained(SSLAM_MODEL_ID, trust_remote_code=True)
model.to(device)
model.eval()

HAS_EXTRACT = hasattr(model, "extract_features")
print("Model has extract_features():", HAS_EXTRACT)
print("Model hidden_size (config) =", getattr(model.config, "hidden_size", None))

print("Opening audio memmap:", AUDIO_X_PATH)
X_audio = np.load(AUDIO_X_PATH, mmap_mode="r")
print("  X_audio shape:", X_audio.shape)

N, C, T = X_audio.shape
if C != 1:
    raise RuntimeError(f"Expected mono audio (C=1), got C={C}")
print(f"  Each clip length: {T} samples @ {ORIG_SR} Hz ~= {T / ORIG_SR:.3f} s")


def wave_batch_to_fbank(batch_wave_np: np.ndarray,
                        orig_sr: int = ORIG_SR,
                        target_sr: int = TARGET_SR,
                        num_mel_bins: int = NUM_MEL_BINS,
                        target_length: int = TARGET_LENGTH,
                        device: torch.device = device) -> torch.Tensor:
    B, _ = batch_wave_np.shape
    feats = []
    for i in range(B):
        w = torch.from_numpy(batch_wave_np[i]).float().to(device)
        if orig_sr != target_sr:
            w = torchaudio.functional.resample(w, orig_freq=orig_sr, new_freq=target_sr)
        w = w - w.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            w.unsqueeze(0),
            htk_compat=True,
            sample_frequency=target_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        n_frames = fbank.size(0)
        if n_frames < target_length:
            fbank = F.pad(fbank, (0, 0, 0, target_length - n_frames))
        elif n_frames > target_length:
            fbank = fbank[:target_length, :]
        feats.append(fbank)
    return torch.stack(feats, dim=0).unsqueeze(1)


def sslam_embed_batch(batch_wave_np: np.ndarray) -> np.ndarray:
    mel = wave_batch_to_fbank(batch_wave_np, device=device)
    with torch.no_grad():
        mel = mel.to(device)
        if HAS_EXTRACT:
            feats = model.extract_features(mel)
        else:
            feats = model(mel).last_hidden_state
        emb = feats.mean(dim=1)
    return emb.cpu().numpy()


print("Running a small warm-up batch to determine embedding dimension...")
with torch.no_grad():
    first_wave = np.asarray(X_audio[0:1, 0, :], dtype="float32")
    first_emb = sslam_embed_batch(first_wave)
    D_ssl = first_emb.shape[1]

print(f"SSLAM embedding dimension inferred as D_ssl={D_ssl}")
print("Creating embedding memmap:", EMB_PATH)
emb_mm = np.lib.format.open_memmap(EMB_PATH, mode="w+", dtype="float32", shape=(N, D_ssl))

num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
print("Starting batched SSLAM embedding extraction...")
start_time = time.time()
last_print_time = start_time

with torch.no_grad():
    for b_idx in tqdm(range(num_batches), desc="Extracting SSLAM embeddings"):
        b_start = b_idx * BATCH_SIZE
        b_end = min((b_idx + 1) * BATCH_SIZE, N)
        batch_wave = np.asarray(X_audio[b_start:b_end, 0, :], dtype="float32")
        emb_np = sslam_embed_batch(batch_wave)
        emb_mm[b_start:b_end, :] = emb_np

        now = time.time()
        if now - last_print_time >= PRINT_EVERY_SEC or b_idx == num_batches - 1:
            processed = b_end
            elapsed = now - start_time
            cps = processed / max(elapsed, 1e-6)
            eta_sec = (N - processed) / max(cps, 1e-6)
            print(f"Processed {processed}/{N} clips | {cps:.1f} clips/s | elapsed {elapsed/60:.1f} min | ETA {eta_sec/60:.1f} min")
            last_print_time = now

emb_mm.flush()
print("Done extracting SSLAM embeddings.")
print("Embeddings saved to:", EMB_PATH)
