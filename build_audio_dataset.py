"""
Build the MAHNOB-HCI continuous audio dataset used by the EEG+Audio paper.

This script creates 10-second audio-window files expected by
compute_sslam_embeddings.py.

Window convention
-----------------
This version uses same-window supervision:
    window k = [k * 0.25 s, k * 0.25 s + W)
    target   = valence[k]
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm.auto import tqdm

BASE_DIR = Path(os.environ.get("MAHNOB_BASE_DIR", "/path/to/HCI_Tagging_Database"))
MEDIA_DIR = BASE_DIR / "MediaFiles"
LABEL_FILE = BASE_DIR / "lable_continous_Mahnob.mat"
TRIAL_TO_MEDIA_PATH = BASE_DIR / "TRIAL_TO_MEDIA.npy"

OUTPUT_DIR = BASE_DIR / "Continuous_Audio_32k" / "10sec"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 10
LABEL_FREQUENCY = 4.0
HOP_SEC = 1.0 / LABEL_FREQUENCY
TARGET_SR_AUDIO = 32000
AUDIO_DTYPE = "float32"

X_OUT = OUTPUT_DIR / "Cont_Audio_10sec_X_32k.npy"
Y_OUT = OUTPUT_DIR / "Cont_Audio_10sec_y_32k.npy"
META_OUT = OUTPUT_DIR / "metadata_audio_10sec.csv"
DESC_OUT = OUTPUT_DIR / "dataset_description.json"


def load_trial_to_media_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"TRIAL_TO_MEDIA file not found: {path}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def parse_trial_key(key: str):
    p, t = key.split("-T")
    return int(p[1:]), int(t)


def load_continuous_labels(mat_path: Path):
    if not mat_path.exists():
        raise FileNotFoundError(f"Continuous label MAT file not found: {mat_path}")
    mat = sio.loadmat(mat_path)
    trials_included = mat["trials_included"]
    raw_labels = np.squeeze(mat["labels"])

    labels = {}
    for i in range(trials_included.shape[0]):
        subj_id = int(trials_included[i, 0])
        trial_id = int(trials_included[i, 1])
        arr = np.asarray(raw_labels[i])
        if arr.ndim == 2:
            arr = arr.T
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[:, 0]
        labels[(subj_id, trial_id)] = arr.astype("float32").reshape(-1)
    return labels


_audio_cache = {}


def audio_for_media(media_file: str):
    if media_file in _audio_cache:
        return _audio_cache[media_file]
    path = MEDIA_DIR / media_file
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    clip = VideoFileClip(str(path))
    try:
        if clip.audio is None:
            raise RuntimeError(f"No audio track in {path}")
        audio = clip.audio.to_soundarray(fps=TARGET_SR_AUDIO)
    finally:
        clip.close()

    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1).astype(AUDIO_DTYPE)
    _audio_cache[media_file] = audio
    return audio


def count_windows(n_samples: int, n_labels: int) -> int:
    win = int(round(WINDOW_SEC * TARGET_SR_AUDIO))
    hop = int(round(HOP_SEC * TARGET_SR_AUDIO))
    max_by_audio = (n_samples - win) // hop + 1 if n_samples >= win else 0
    return max(0, min(n_labels, max_by_audio))


def main():
    trial_to_media_raw = load_trial_to_media_mapping(TRIAL_TO_MEDIA_PATH)
    pair_to_media = {}
    for k, v in trial_to_media_raw.items():
        if isinstance(k, str) and k.startswith("P") and "-T" in k:
            pair_to_media[parse_trial_key(k)] = str(v)

    labels = load_continuous_labels(LABEL_FILE)
    sessions = sorted(set(labels.keys()) & set(pair_to_media.keys()))
    if not sessions:
        raise RuntimeError("No overlap between label sessions and TRIAL_TO_MEDIA mapping.")

    total = 0
    per_session_counts = {}
    for subj_id, trial_id in tqdm(sessions, desc="Counting windows"):
        media_file = pair_to_media[(subj_id, trial_id)]
        audio = audio_for_media(media_file)
        n = count_windows(len(audio), len(labels[(subj_id, trial_id)]))
        per_session_counts[(subj_id, trial_id)] = n
        total += n

    win_samples = int(round(WINDOW_SEC * TARGET_SR_AUDIO))
    hop_samples = int(round(HOP_SEC * TARGET_SR_AUDIO))
    print(f"Total windows: {total:,}")

    X = np.lib.format.open_memmap(X_OUT, mode="w+", dtype=AUDIO_DTYPE, shape=(total, 1, win_samples))
    y = np.lib.format.open_memmap(Y_OUT, mode="w+", dtype="float32", shape=(total,))

    rows = []
    row = 0
    for subj_id, trial_id in tqdm(sessions, desc="Writing windows"):
        media_file = pair_to_media[(subj_id, trial_id)]
        audio = audio_for_media(media_file)
        lab = labels[(subj_id, trial_id)]
        n = per_session_counts[(subj_id, trial_id)]
        session_id = f"P{subj_id}-T{trial_id}"
        for label_idx in range(n):
            start = label_idx * hop_samples
            end = start + win_samples
            X[row, 0, :] = audio[start:end]
            y[row] = lab[label_idx]
            rows.append({
                "row_idx": row,
                "session_id": session_id,
                "subject_id": subj_id,
                "trial_id": trial_id,
                "media_file": media_file,
                "label_idx": label_idx,
                "time_sec_label": label_idx * HOP_SEC,
                "start_sample": start,
                "end_sample": end,
                "fs": TARGET_SR_AUDIO,
                "window_sec": WINDOW_SEC,
                "supervision_window_convention": "same_window",
            })
            row += 1

    X.flush(); y.flush()
    pd.DataFrame(rows).to_csv(META_OUT, index=False)
    desc = {
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "sample_rate_hz": TARGET_SR_AUDIO,
        "n_windows": int(total),
        "n_sessions": int(len(sessions)),
        "x_file": str(X_OUT),
        "y_file": str(Y_OUT),
        "metadata_file": str(META_OUT),
        "target_definition": "per-session continuous valence label at label_idx",
        "window_convention": "same_window: [label_idx*hop, label_idx*hop + window_sec)",
    }
    DESC_OUT.write_text(json.dumps(desc, indent=2), encoding="utf-8")
    print(f"Saved: {X_OUT}\nSaved: {Y_OUT}\nSaved: {META_OUT}\nSaved: {DESC_OUT}")


if __name__ == "__main__":
    main()
