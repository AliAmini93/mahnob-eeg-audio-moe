# ============================================================
# Multimodal REGRESSION with Strategy-1 MoE — EEG + Audio SSLAM
# CONTENT-HELD-OUT 5-FOLD CV (outer+inner) + TRAIN-STAT EEG NORMALIZATION
# Key evaluation fix:
#   We split folds at the STIMULUS level (video/clip identity), so that
#   no stimulus audio can appear in both train and test. This prevents
#   cross-subject stimulus/audio overlap.
#
# What this script does:
#  1) Align EEG and Audio windows via metadata keys (session_id, subject_id, trial_id, label_idx)
#  2) Infer a robust stimulus_id column (shared across subjects) from merged metadata
#  3) Build audio_window_id = (stimulus_id + time-index) (exclude subject/session)
#  4) OUTER 5-fold split is done on stimulus_id (stratified by stimulus-mean target when feasible)
#  5) INNER split (within trv) is also done on stimulus_id (stratified when feasible)
#  6) EEG normalization uses TRAIN-only stats per fold
#  7) Train/validate/test with standard unweighted CCC loss
#
# Guarantees (programmatically asserted):
#  • stimulus_id(train) ∩ stimulus_id(test) = ∅
#  • stimulus_id(train) ∩ stimulus_id(val)  = ∅
#  • audio_window_id(train) ∩ audio_window_id(test) = ∅  (if audio_window_id includes stimulus_id)
# ============================================================

import os, time, random, warnings, subprocess, gc
import numpy as np
import pandas as pd
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Optional: psutil for RAM monitoring (if installed)
try:
    import psutil
except ImportError:
    psutil = None


# -------------------------
# Repro/Device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------
# High-level config
# -------------------------
BASE_DIR = os.environ.get("MAHNOB_BASE_DIR", r"/path/to/HCI_Tagging_Database")

AUDIO_ROOT = os.path.join(BASE_DIR, "Continuous_Audio_32k")
EEG_ROOT   = os.path.join(BASE_DIR, "Continuous EEG Time-series dataset")

WINDOW_SEC     = 10
AUDIO_FOLDER   = f"{WINDOW_SEC}sec"
EEG_ICA_STATUS = "WithICA"
EEG_FOLDER     = f"Cont_{EEG_ICA_STATUS}_{WINDOW_SEC}sec"

AUDIO_DATA_DIR = os.path.join(AUDIO_ROOT, AUDIO_FOLDER)
EEG_DATA_DIR   = os.path.join(EEG_ROOT, EEG_FOLDER)

print("Audio data dir:", AUDIO_DATA_DIR)
print("EEG data dir  :", EEG_DATA_DIR)

# -------- Targets --------
# Choices: "eeg" or "audio"
TARGET_EEG_KIND   = "eeg"
TARGET_AUDIO_KIND = "eeg"
TARGET_FINAL_KIND = "eeg"

# Expert supervision
USE_EXPERT_LOSS = False
LAMBDA_EEG      = 1.0
LAMBDA_AUDIO    = 1.0

# -------- Training hyperparams --------
BATCH_SIZE = 64
LR         = 3e-4
WD         = 1e-4
EPOCHS     = 1000
PATIENCE   = 20
USE_AMP    = False
CLIP_GRAD  = 1.0

# EEG normalization toggles
NORM_SUBJECT   = True
NORM_CHANNEL   = True
LEAK_SAFE      = True          # per-fold train-stats normalization (leak-safe)
NORM_KIND      = "bn"          # {'bn','gn'}
USE_TEMP_POOLS = False

# ---- Model config ----
EEG_MODEL_SIZE     = "base"       # {"lite","base"}
AUDIO_ADAPTER_TYPE = "linear"     # {"linear","bottleneck"}

# Audio embedding dim (SSLAM: 768)
EMB_DIM_AUDIO = 768
D_AUDIO_OUT   = 256

# MoE head
EXPERT_HIDDEN_DIM = 128
GATE_HIDDEN_DIM   = 64

# CV config
OUTER_FOLDS = 5
INNER_FOLDS = 5
MAX_BINS    = 5

# ---- Stimulus split config ----
# If you KNOW the correct stimulus key in your metadata, set it here.
# Examples: "video_id", "clip_id", "trial_id", or a dataset-specific column.
FORCE_STIMULUS_KEY = "media_file"  # content-held-out must split by actual stimulus/audio file


RESULTS_DIR = os.path.join(
    BASE_DIR,
    f"results_multimodal_eeg_audio_ssl_moe_strategy1_CONTENT_HELD_OUT_5F_final_{TARGET_FINAL_KIND}_CCC"
)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESOURCE_LOG_PATH = os.path.join(RESULTS_DIR, "resource_usage.log")


# ------------------------------------------------
# Simple resource monitoring utility
# ------------------------------------------------
def log_resource_usage(tag=""):
    parts = []
    if psutil is not None:
        proc = psutil.Process(os.getpid())
        mem_gb = proc.memory_info().rss / (1024 ** 3)
        parts.append(f"RAM={mem_gb:.2f}GB")

    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved  = torch.cuda.memory_reserved()  / (1024 ** 3)
            parts.append(f"GPU_alloc={allocated:.2f}GB")
            parts.append(f"GPU_resv={reserved:.2f}GB")
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            smi = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
            line = smi.stdout.strip().splitlines()[0]
            used_mb, total_mb = [float(x.strip()) for x in line.split(",")]
            parts.append(f"nvsmi={used_mb/1024:.2f}/{total_mb/1024:.2f}GB")
        except Exception:
            pass

    msg = f"[RES] {tag} | " + " | ".join(parts) if parts else f"[RES] {tag} | (no info)"
    print(msg)
    try:
        with open(RESOURCE_LOG_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ------------------------------------------------
# Load Audio SSLAM embeddings, labels, metadata
# ------------------------------------------------
AUDIO_EMB_PATH    = os.path.join(AUDIO_DATA_DIR, f"audio_emb_sslam_{WINDOW_SEC}sec.npy")
AUDIO_Y_PATH = os.path.join(AUDIO_DATA_DIR, f"Cont_Audio_{WINDOW_SEC}sec_y_32k.npy")
AUDIO_META_PATH   = os.path.join(AUDIO_DATA_DIR, f"metadata_audio_{WINDOW_SEC}sec.csv")

print("\nLoading Audio SSLAM embeddings...")
X_audio = np.load(AUDIO_EMB_PATH).astype("float32")
print("  X_audio shape:", X_audio.shape)

print("Loading per-window audio targets...")
y_audio = np.load(AUDIO_Y_PATH).astype("float32")
if y_audio.ndim == 2:
    y_audio = y_audio.squeeze(-1)
print("  y_audio shape:", y_audio.shape)

print("Loading Audio metadata...")
meta_audio = pd.read_csv(AUDIO_META_PATH)
print("  audio meta columns:", meta_audio.columns.tolist())


# ------------------------------------------------
# Load EEG dataset
# ------------------------------------------------
print("\nLoading EEG dataset...")
X_eeg    = np.load(os.path.join(EEG_DATA_DIR, "X_input.npy")).astype("float32")
y_eeg    = np.load(os.path.join(EEG_DATA_DIR, "y_target.npy")).astype("float32")
meta_eeg = pd.read_csv(os.path.join(EEG_DATA_DIR, "metadata.csv"))

if y_eeg.ndim == 2:
    y_eeg = y_eeg.squeeze(-1)

Ne, C, S = X_eeg.shape
print("  X_eeg shape:", X_eeg.shape, "y_eeg shape:", y_eeg.shape)
print("  eeg meta columns:", meta_eeg.columns.tolist())
assert C == 32, f"Expected 32 EEG channels, got {C}"

if "subject_id" in meta_eeg.columns:
    subjects_eeg_all = meta_eeg["subject_id"].astype(str).values
else:
    subjects_eeg_all = np.array(["0"] * len(X_eeg), dtype=str)


# ------------------------------------------------
# Align EEG windows with Audio windows via metadata
# ------------------------------------------------
print("\nAligning EEG and Audio windows...")

meta_audio = meta_audio.reset_index().rename(columns={"index": "a_idx"})
meta_eeg   = meta_eeg.reset_index().rename(columns={"index": "e_idx"})

candidate_keys = ["session_id", "subject_id", "trial_id", "label_idx"]
keys = [k for k in candidate_keys if k in meta_audio.columns and k in meta_eeg.columns]
if len(keys) == 0:
    raise RuntimeError(
        f"No common alignment keys found between EEG and Audio metadata. "
        f"Tried {candidate_keys}, found none."
    )
print("  Using alignment keys:", keys)

merged = meta_eeg.merge(
    meta_audio,
    on=keys,
    how="inner",
    suffixes=("_eeg", "_audio"),
)

if len(merged) == 0:
    raise RuntimeError("No aligned EEG-Audio windows found. Check your metadata and keys.")

idx_eeg_aligned   = merged["e_idx"].values
idx_audio_aligned = merged["a_idx"].values
print(f"  Found {len(idx_eeg_aligned)} aligned windows.")

# Build aligned arrays
X_eeg_all    = X_eeg[idx_eeg_aligned].astype("float32")
X_audio_all  = X_audio[idx_audio_aligned].astype("float32")
y_eeg_all    = y_eeg[idx_eeg_aligned].astype("float32")
y_audio_all  = y_audio[idx_audio_aligned].astype("float32")
subjects_all = subjects_eeg_all[idx_eeg_aligned]

N = len(y_eeg_all)
print(f"\nAligned multimodal dataset: N={N}, EEG={X_eeg_all.shape}, Audio={X_audio_all.shape}")
print(f"TARGET_EEG_KIND   = {TARGET_EEG_KIND}")
print(f"TARGET_AUDIO_KIND = {TARGET_AUDIO_KIND}")
print(f"TARGET_FINAL_KIND = {TARGET_FINAL_KIND}")
log_resource_usage("after_alignment")


# ------------------------------------------------
# Helpers for merged column picking and stimulus inference
# ------------------------------------------------
def _pick_col(df, base):
    """Return base, or base_audio/base_eeg if present (merge can suffix)."""
    if base in df.columns:
        return base
    if f"{base}_audio" in df.columns:
        return f"{base}_audio"
    if f"{base}_eeg" in df.columns:
        return f"{base}_eeg"
    return None


def infer_stimulus_column(meta_df: pd.DataFrame, force_key=None, min_multisubject_groups=1):
    """
    Infer a column that represents stimulus identity shared across subjects.

    Selection logic:
      1) If force_key is provided, use it (or its _audio/_eeg variant) if present.
      2) Try preferred keys in order: stimulus_id, video_id, clip_id, trial_id
      3) If subject_id exists, score candidates by how many stimulus values are seen by >1 subject.
         Pick the best-scoring candidate.
      4) If no candidate passes multi-subject sanity check, raise with a clear error.
    """
    subj_col = _pick_col(meta_df, "subject_id")

    # --- explicit override ---
    if force_key is not None:
        forced = _pick_col(meta_df, force_key)
        if forced is None:
            raise RuntimeError(
                f"FORCE_STIMULUS_KEY='{force_key}' not found in merged metadata columns. "
                f"Available columns: {meta_df.columns.tolist()}"
            )
        print(f"[STIMULUS] FORCE_STIMULUS_KEY used: {force_key} -> column '{forced}'")
        return forced

    # --- preferred candidates ---
    preferred_bases = ["media_file", "stimulus_id", "video_id", "clip_id", "trial_id"]
    cand_cols = []
    for b in preferred_bases:
        c = _pick_col(meta_df, b)
        if c is not None and c not in cand_cols:
            cand_cols.append(c)

    # If no subject column, we can't do multi-subject verification. Use first available.
    if subj_col is None:
        if len(cand_cols) == 0:
            raise RuntimeError(
                "No stimulus candidate columns found (stimulus_id/video_id/clip_id/trial_id) "
                "and subject_id is missing, so we cannot infer stimulus_id."
            )
        print(f"[STIMULUS] subject_id missing; using '{cand_cols[0]}' as stimulus key (best effort).")
        return cand_cols[0]

    if len(cand_cols) == 0:
        raise RuntimeError(
            "No stimulus candidate columns found (stimulus_id/video_id/clip_id/trial_id). "
            "Add a stimulus identifier to your metadata or set FORCE_STIMULUS_KEY."
        )

    # Score candidates: number of stimulus values with >1 subject
    scores = []
    for c in cand_cols:
        n_unique = meta_df[c].astype(str).nunique()
        n_multi = (meta_df.groupby(c)[subj_col].nunique() > 1).sum()
        scores.append((c, n_multi, n_unique))

    print("[STIMULUS] Candidate stimulus keys (col | multi-subject-groups | unique-values):")
    for c, n_multi, n_unique in scores:
        print(f"  - {c:20s} | {n_multi:6d} | {n_unique:6d}")

    # Pick the candidate with max n_multi; break ties by fewer unique values (often indicates stimulus-level)
    scores_sorted = sorted(scores, key=lambda x: (x[1], -x[2]), reverse=True)
    best_col, best_multi, best_unique = scores_sorted[0]

    if best_multi < min_multisubject_groups:
        raise RuntimeError(
            f"Could not find a valid stimulus identifier shared across subjects. "
            f"Best candidate '{best_col}' has multi-subject-groups={best_multi}. "
            "This likely means your trial_id/video_id is subject-specific. "
            "You MUST provide a true stimulus key (e.g., video_id/clip_id/filename-derived ID) "
            "and set FORCE_STIMULUS_KEY accordingly."
        )

    print(f"[STIMULUS] Selected stimulus key column: '{best_col}'")
    return best_col


# ------------------------------------------------
# Build STIMULUS_ID (the key for leak-proof splitting)
# ------------------------------------------------
STIM_COL = infer_stimulus_column(merged, force_key=FORCE_STIMULUS_KEY, min_multisubject_groups=1)
merged["stimulus_id"] = merged[STIM_COL].astype(str)
stimulus_id_all = merged["stimulus_id"].values

n_stim_total = len(np.unique(stimulus_id_all))
print(f"[STIMULUS] unique stimuli = {n_stim_total}")
if n_stim_total < OUTER_FOLDS:
    raise RuntimeError(
        f"Need at least {OUTER_FOLDS} unique stimuli for {OUTER_FOLDS}-fold CV; got {n_stim_total}."
    )


# ------------------------------------------------
# Build AUDIO_WINDOW_ID (stimulus+time; exclude subject/session)
# Detects duplicates across subjects WITHIN TRAIN.
# ------------------------------------------------
AUDIO_ID_KEYS = ["stimulus_id"]

kk = _pick_col(merged, "label_idx")
if kk is not None:
    AUDIO_ID_KEYS.append(kk)

for k in ["start_sample", "end_sample"]:
    kk = _pick_col(merged, k)
    if kk is not None:
        AUDIO_ID_KEYS.append(kk)

# Explicitly ensure subject/session not in the keys
AUDIO_ID_KEYS = [k for k in AUDIO_ID_KEYS if "subject_id" not in k and "session" not in k]

if len(AUDIO_ID_KEYS) < 2:
    raise RuntimeError(f"Not enough columns to define audio_window_id. Got keys: {AUDIO_ID_KEYS}")

merged["audio_window_id"] = merged[AUDIO_ID_KEYS].astype(str).agg("_".join, axis=1)
audio_window_id_all = merged["audio_window_id"].values

print(f"[AUDIO DUP CHECK] unique audio_window_id = {len(np.unique(audio_window_id_all))} / windows = {len(audio_window_id_all)}")
print(f"[AUDIO DUP CHECK] audio_window_id keys = {AUDIO_ID_KEYS}")


# ------------------------------------------------
# (Optional) Trial-group ids for reporting (subject × stimulus × session)
# Not used for splitting (splitting is stimulus-disjoint).
# ------------------------------------------------
def build_trial_group_ids(meta_df: pd.DataFrame):
    keys = []
    # stimulus_id is guaranteed to exist now
    keys.append("stimulus_id")

    # include subject to represent "trial instance per subject"
    subj_col = _pick_col(meta_df, "subject_id")
    if subj_col is not None:
        keys.append(subj_col)

    # include session context if available (optional)
    sess_col = _pick_col(meta_df, "session_id") or _pick_col(meta_df, "session")
    if sess_col is not None:
        keys.append(sess_col)

    gid = meta_df[keys].astype(str).agg("_".join, axis=1).values
    return gid, keys

group_ids_all, GROUP_KEYS = build_trial_group_ids(merged)
print(f"[TRIAL-GROUP] Using group keys (reporting only): {GROUP_KEYS}")
print(f"[TRIAL-GROUP] Unique groups: {len(np.unique(group_ids_all))}")


# ------------------------------------------------
# Label stream selection
# ------------------------------------------------
def select_label_array(kind: str):
    if kind == "eeg":
        return y_eeg_all
    elif kind in ("audio", "audio_y", "audio_label"):
        return y_audio_all
    else:
        raise ValueError(f"Unknown label kind: {kind}. Use 'eeg' or 'audio'.")


# ------------------------------------------------
# Robust binning for stratified CV (stimulus-level)
# ------------------------------------------------
def make_group_bins(y_group: np.ndarray, n_splits: int, max_bins: int):
    """
    Try qcut/cut with bins from max_bins down to 2.
    Feasible if every bin has at least n_splits samples.
    """
    y_group = np.asarray(y_group, dtype=float)
    for q in range(min(max_bins, len(y_group)), 1, -1):
        try:
            bins = pd.qcut(y_group, q=q, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.cut(y_group, bins=q, labels=False)

        bins = np.asarray(bins, dtype=int)
        counts = np.bincount(bins) if bins.size > 0 else np.array([0])
        if counts.min() >= n_splits:
            return bins, int(q)
    return None, None


# ------------------------------------------------
# Build OUTER splits at STIMULUS-level (stratified on stimulus-mean FINAL target)
# ------------------------------------------------
y_final_all = select_label_array(TARGET_FINAL_KIND)

sdf = pd.DataFrame({"stimulus_id": stimulus_id_all, "y": y_final_all})
stim_mean = sdf.groupby("stimulus_id")["y"].mean()
stim_unique = stim_mean.index.to_numpy()
y_stim = stim_mean.to_numpy()

outer_bins, outer_q = make_group_bins(y_stim, n_splits=OUTER_FOLDS, max_bins=MAX_BINS)

if outer_bins is not None:
    print(f"[SPLIT] OUTER: StratifiedKFold on {len(stim_unique)} stimuli (bins={outer_q})")
    outer_splitter = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_iter = outer_splitter.split(stim_unique, outer_bins)
else:
    print(f"[SPLIT] OUTER: Fallback to KFold on {len(stim_unique)} stimuli")
    outer_splitter = KFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_iter = outer_splitter.split(stim_unique)

outer_splits = []
for s_tr, s_te in outer_iter:
    tr_stim = set(stim_unique[s_tr])
    te_stim = set(stim_unique[s_te])

    trv_idx = np.where(np.isin(stimulus_id_all, list(tr_stim)))[0]
    te_idx  = np.where(np.isin(stimulus_id_all, list(te_stim)))[0]

    # HARD guarantees
    assert set(stimulus_id_all[trv_idx]).isdisjoint(set(stimulus_id_all[te_idx])), "Stimulus overlap in OUTER split!"
    # Audio-window disjointness (strong sanity check; must hold if audio_window_id includes stimulus_id)
    assert set(audio_window_id_all[trv_idx]).isdisjoint(set(audio_window_id_all[te_idx])), "Audio-window overlap in OUTER split!"

    outer_splits.append((trv_idx, te_idx))

print(f"[CONTENT-HELD-OUT CV] unique stimuli={len(stim_unique)} | windows={len(stimulus_id_all)}")

print("\n" + "=" * 110)
print(
    f"EEG+Audio CONTENT-HELD-OUT Config | WIN={WINDOW_SEC}s | EEG={EEG_MODEL_SIZE} | AudioAdapter={AUDIO_ADAPTER_TYPE} | "
    f"OUTER={OUTER_FOLDS} INNER={INNER_FOLDS} | "
    f"BINS={outer_q if outer_bins is not None else 'KFold'} | "
    f"NORM_SUBJ={NORM_SUBJECT} NORM_CH={NORM_CHANNEL} | LEAK_SAFE={LEAK_SAFE} | AMP={USE_AMP} | "
    f"FINAL_TARGET={TARGET_FINAL_KIND} | TRAIN=CCC"
)
print("=" * 110 + "\n")


# ------------------------------------------------
# Loss & metrics
# ------------------------------------------------
class CCCLoss(nn.Module):
    """1 - Concordance Correlation Coefficient (unweighted)."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        x = x.squeeze()
        y = y.squeeze()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        var_x = torch.sum(vx ** 2)
        var_y = torch.sum(vy ** 2)
        rho = cov / (torch.sqrt(var_x * var_y) + self.eps)

        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        x_std = torch.sqrt(var_x / (x.numel() - 1 + self.eps))
        y_std = torch.sqrt(var_y / (y.numel() - 1 + self.eps))

        ccc = (2 * rho * x_std * y_std) / (
            x_std ** 2 + y_std ** 2 + (x_mean - y_mean) ** 2 + self.eps
        )
        return 1.0 - ccc



def safe_pearsonr(y_true, y_pred):
    if np.allclose(y_true, y_true.mean()) or np.allclose(y_pred, y_pred.mean()):
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(r) else float(r)


def concordance_correlation_coefficient(y_true, y_pred):
    cor = safe_pearsonr(y_true, y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.sqrt(var_true)
    sd_pred = np.sqrt(var_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float(numerator / (denominator + 1e-8))


def metrics_from_preds_reg(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc  = safe_pearsonr(y_true, y_pred)
    ccc  = concordance_correlation_coefficient(y_true, y_pred)
    return {"rmse": rmse, "pcc": pcc, "ccc": ccc}


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(train)


def mstr(m):
    return f"rmse={m['rmse']:.4f} pcc={m['pcc']:.4f} ccc={m['ccc']:.4f}"


# ------------------------------------------------
# Normalization helpers (EEG only) — in-place
# ------------------------------------------------
def apply_norm_subject_global_scalar(X_all, subs_all, eps=1e-7):
    for sid in np.unique(subs_all):
        idx = np.where(subs_all == sid)[0]
        mu = X_all[idx].mean(axis=(0,1,2), keepdims=True)
        sd = X_all[idx].std (axis=(0,1,2), keepdims=True) + eps
        X_all[idx] = (X_all[idx] - mu) / sd
    return X_all


def apply_norm_subject_trainstats_scalar(X_all, subs_all, train_idx, eps=1e-7):
    train_mask = np.zeros(len(X_all), dtype=bool)
    train_mask[train_idx] = True

    mu_global = X_all[train_idx].mean(axis=(0,1,2), keepdims=True)
    sd_global = X_all[train_idx].std (axis=(0,1,2), keepdims=True) + eps

    for sid in np.unique(subs_all):
        idx_all = np.where(subs_all == sid)[0]
        idx_tr  = idx_all[train_mask[idx_all]]
        if len(idx_tr) == 0:
            mu, sd = mu_global, sd_global
        else:
            mu = X_all[idx_tr].mean(axis=(0,1,2), keepdims=True)
            sd = X_all[idx_tr].std (axis=(0,1,2), keepdims=True) + eps
        X_all[idx_all] = (X_all[idx_all] - mu) / sd
    return X_all


def apply_norm_channel_global(X_all, eps=1e-7):
    mu = X_all.mean(axis=(0,2), keepdims=True)
    sd = X_all.std (axis=(0,2), keepdims=True) + eps
    X_all[:] = (X_all - mu) / sd
    return X_all


def apply_norm_channel_trainstats(X_all, train_idx, eps=1e-7):
    mu = X_all[train_idx].mean(axis=(0,2), keepdims=True)
    sd = X_all[train_idx].std (axis=(0,2), keepdims=True) + eps
    X_all[:] = (X_all - mu) / sd
    return X_all


# Precompute EEG normalization when LEAK_SAFE = False
if not LEAK_SAFE:
    print("\nApplying global EEG normalization (subject/channel)...")
    X_eeg_norm_all = X_eeg_all.copy()
    if NORM_SUBJECT:
        X_eeg_norm_all = apply_norm_subject_global_scalar(X_eeg_norm_all, subjects_all)
    if NORM_CHANNEL:
        X_eeg_norm_all = apply_norm_channel_global(X_eeg_norm_all)
    log_resource_usage("after_global_eeg_norm")
else:
    X_eeg_norm_all = None
    X_eeg_backup = X_eeg_all.copy()  # restore each fold


# ------------------------------------------------
# EEG encoder (Stage-A)
# ------------------------------------------------
def _gn(num_channels, groups):
    groups = min(groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)

def _norm1d(in_channels):
    return nn.BatchNorm1d(in_channels) if NORM_KIND == "bn" else _gn(in_channels, groups=32)

class DepthwiseTemporalStem(nn.Module):
    def __init__(self, C, F_per, k0=15, s0=1, dropout=0.05):
        super().__init__()
        p0 = k0 // 2
        self.dw = nn.Conv1d(C, C, kernel_size=k0, stride=s0, padding=p0, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C * F_per, kernel_size=1, bias=False)
        self.norm = _norm1d(C * F_per)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = F.elu(x)
        x = self.drop(x)
        return x

class TCNBlockDW(nn.Module):
    def __init__(self, C, F_per, dilation, dropout=0.1, k=3):
        super().__init__()
        in_ch = C * F_per
        pad = dilation * (k // 2)
        self.norm1 = _norm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, dilation=dilation, groups=C, bias=False)
        self.norm2 = _norm1d(in_ch)
        self.mix   = nn.Conv1d(in_ch, in_ch, kernel_size=1, groups=C, bias=False)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = F.elu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.mix(x)
        x = self.drop(x)
        return x + res

class TCNStackDW(nn.Module):
    def __init__(self, C, F_per, dilations, dropout=0.1, k=3):
        super().__init__()
        self.blocks = nn.ModuleList([TCNBlockDW(C, F_per, d, dropout=dropout, k=k) for d in dilations])
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class TemporalAttnPool1D(nn.Module):
    def __init__(self, C, F_per):
        super().__init__()
        self.C, self.F = C, F_per
        self.scorer = nn.Conv1d(C * F_per, C, kernel_size=1, groups=C, bias=True)
    def forward(self, x):
        B, CF, T = x.shape
        C, Fp = self.C, self.F
        scores = self.scorer(x)
        alpha  = torch.softmax(scores, dim=-1)
        x4 = x.view(B, C, Fp, T)
        h = (x4 * alpha.unsqueeze(2)).sum(dim=-1)
        return h

class ChannelGAT(nn.Module):
    def __init__(self, C, F_in, D_gat=64, nheads=4, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(F_in, D_gat)
        self.attn    = nn.MultiheadAttention(D_gat, nheads, dropout=dropout, batch_first=False)
        self.ln1     = nn.LayerNorm(D_gat)
        self.ffn     = nn.Sequential(
            nn.Linear(D_gat, 2 * D_gat), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(2 * D_gat, D_gat), nn.Dropout(dropout)
        )
        self.ln2     = nn.LayerNorm(D_gat)
    def forward(self, h):
        x = self.proj_in(h)
        x_t = x.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t, need_weights=False)
        x_t = self.ln1(x_t + attn_out)
        x_ffn = self.ffn(x_t)
        x_t = self.ln2(x_t + x_ffn)
        return x_t.transpose(0, 1)

class ChannelAttnPool(nn.Module):
    def __init__(self, D_gat, dropout=0.1):
        super().__init__()
        self.score = nn.Linear(D_gat, 1)
        self.drop  = nn.Dropout(dropout)
    def forward(self, h):
        w = self.score(h).squeeze(-1)
        alpha = torch.softmax(w, dim=-1).unsqueeze(-1)
        z = (h * alpha).sum(dim=1)
        return self.drop(z)

class EEGEncoderStageA(nn.Module):
    PRESETS = {
        "base": {10: dict(k0=41, s0=4, F_per=16, dilations=[1,2,4,8,16,32,64,128], D_gat=64, nheads=4)},
        "lite": {10: dict(k0=41, s0=4, F_per=8,  dilations=[1,2,4,8,16,32,64],     D_gat=32, nheads=2)},
    }
    def __init__(self, C=32, window_sec=10, dropout=0.1, D_embed=256, model_size="base"):
        super().__init__()
        if model_size not in self.PRESETS:
            raise ValueError(f"Unknown EEG model_size: {model_size}")
        key = window_sec if window_sec in self.PRESETS[model_size] else 10
        P = self.PRESETS[model_size][key]

        self.C, self.F_per = C, P["F_per"]
        self.stem  = DepthwiseTemporalStem(C=C, F_per=self.F_per, k0=P["k0"], s0=P["s0"], dropout=0.05)
        self.pool1 = nn.AvgPool1d(4) if USE_TEMP_POOLS else nn.Identity()
        self.tcn   = TCNStackDW(C=C, F_per=self.F_per, dilations=P["dilations"], dropout=dropout, k=3)
        self.pool2 = nn.AvgPool1d(8) if USE_TEMP_POOLS else nn.Identity()
        self.tpool = TemporalAttnPool1D(C=C, F_per=self.F_per)
        self.gat   = ChannelGAT(C=C, F_in=self.F_per, D_gat=P["D_gat"], nheads=P["nheads"], dropout=dropout)
        self.cpool = ChannelAttnPool(D_gat=P["D_gat"], dropout=dropout)
        self.proj  = nn.Sequential(nn.LayerNorm(P["D_gat"]), nn.Linear(P["D_gat"], D_embed))

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(x)
        x = self.tcn(x)
        x = self.pool2(x)
        h = self.tpool(x)
        h = self.gat(h)
        z = self.cpool(h)
        return self.proj(z)


# ------------------------------------------------
# Audio adapter: 768 -> 256
# ------------------------------------------------
class AudioAdapter(nn.Module):
    def __init__(self, D_in=EMB_DIM_AUDIO, D_out=D_AUDIO_OUT, p_drop=0.3, adapter_type="linear"):
        super().__init__()
        if adapter_type == "linear":
            self.net = nn.Sequential(nn.LayerNorm(D_in), nn.Linear(D_in, D_out))
        elif adapter_type == "bottleneck":
            self.net = nn.Sequential(
                nn.LayerNorm(D_in),
                nn.Linear(D_in, 32),
                nn.ELU(),
                nn.Dropout(p_drop),
                nn.Linear(32, D_out),
            )
        else:
            raise ValueError("AudioAdapter: adapter_type must be 'linear' or 'bottleneck'.")
    def forward(self, x):
        return self.net(x)


# ------------------------------------------------
# Strategy-1 MoE head + Fusion model
# ------------------------------------------------
class MoEHeadStrategy1(nn.Module):
    def __init__(self, D_eeg, D_audio, expert_hidden=128, gate_hidden=64, p_drop=0.3):
        super().__init__()
        self.eeg_expert = nn.Sequential(
            nn.LayerNorm(D_eeg),
            nn.Linear(D_eeg, expert_hidden),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Linear(expert_hidden, 1),
        )
        self.aud_expert = nn.Sequential(
            nn.LayerNorm(D_audio),
            nn.Linear(D_audio, expert_hidden),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Linear(expert_hidden, 1),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(D_eeg + D_audio),
            nn.Linear(D_eeg + D_audio, gate_hidden),
            nn.ELU(),
            nn.Linear(gate_hidden, 2),
        )

    def forward(self, z_eeg, z_audio):
        y_eeg = self.eeg_expert(z_eeg)
        y_aud = self.aud_expert(z_audio)
        z_f = torch.cat([z_eeg, z_audio], dim=-1)
        g = torch.softmax(self.gate(z_f), dim=-1)
        y_hat = g[:, 0:1] * y_eeg + g[:, 1:2] * y_aud
        return y_hat, g, y_eeg, y_aud


class MoEFusionModelStrategy1(nn.Module):
    def __init__(self, window_sec=10, p_drop=0.3,
                 eeg_model_size=EEG_MODEL_SIZE, audio_adapter_type=AUDIO_ADAPTER_TYPE,
                 expert_hidden=EXPERT_HIDDEN_DIM, gate_hidden=GATE_HIDDEN_DIM):
        super().__init__()
        self.D_eeg   = 256 if eeg_model_size == "base" else 128
        self.D_audio = D_AUDIO_OUT

        self.eeg_enc = EEGEncoderStageA(
            C=32, window_sec=window_sec, dropout=0.1, D_embed=self.D_eeg, model_size=eeg_model_size
        )
        self.audio_enc = AudioAdapter(
            D_in=EMB_DIM_AUDIO, D_out=self.D_audio, p_drop=p_drop, adapter_type=audio_adapter_type
        )
        self.moe = MoEHeadStrategy1(
            D_eeg=self.D_eeg, D_audio=self.D_audio,
            expert_hidden=expert_hidden, gate_hidden=gate_hidden, p_drop=p_drop
        )

    def forward(self, x_eeg, x_audio):
        z_eeg = self.eeg_enc(x_eeg)
        z_aud = self.audio_enc(x_audio)
        return self.moe(z_eeg, z_aud)


# ------------------------------------------------
# Dataset (returns weights)
# ------------------------------------------------
class EEGAudioDS(Dataset):
    def __init__(self, X_eeg, X_audio, y_eeg, y_audio, subs, idx):
        self.X_eeg   = X_eeg
        self.X_audio = X_audio
        self.y_eeg   = y_eeg
        self.y_audio = y_audio
        self.idx     = np.asarray(idx, dtype=int)
        self.subs    = subs[self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gi = int(self.idx[i])
        return (
            torch.from_numpy(self.X_eeg[gi]),
            torch.from_numpy(self.X_audio[gi]),
            torch.tensor(self.y_eeg[gi], dtype=torch.float32),
            torch.tensor(self.y_audio[gi], dtype=torch.float32),
            self.subs[i],
        )


def select_batch_target(kind: str, y_eeg_b, y_audio_b):
    if kind == "eeg":
        return y_eeg_b
    elif kind in ("audio", "audio_y", "audio_label"):
        return y_audio_b
    else:
        raise ValueError(f"Unknown label kind: {kind}.")


# ------------------------------------------------
# Train / Eval (standard CCC loss)
# ------------------------------------------------
def train_epoch(model, loader, opt, scaler, loss_fn_train, clip_grad=None):
    model.train()
    loss_sum, n_samples = 0.0, 0
    use_amp = (device.type == "cuda" and USE_AMP)

    for xb_eeg, xb_audio, yb_eeg, yb_audio, _ in loader:
        xb_eeg   = xb_eeg.to(device, non_blocking=True)
        xb_audio = xb_audio.to(device, non_blocking=True)
        yb_eeg   = yb_eeg.to(device, non_blocking=True)
        yb_audio = yb_audio.to(device, non_blocking=True)

        yb_final = select_batch_target(TARGET_FINAL_KIND, yb_eeg, yb_audio)
        yb_eeg_t = select_batch_target(TARGET_EEG_KIND,   yb_eeg, yb_audio)
        yb_aud_t = select_batch_target(TARGET_AUDIO_KIND, yb_eeg, yb_audio)

        opt.zero_grad(set_to_none=True)

        ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            preds, gate_probs, y_eeg_exp, y_aud_exp = model(xb_eeg, xb_audio)
            preds    = preds.squeeze()
            yb_final = yb_final.squeeze()

            total_loss = loss_fn_train(preds, yb_final)

            if USE_EXPERT_LOSS:
                if LAMBDA_EEG > 0.0:
                    total_loss = total_loss + LAMBDA_EEG * loss_fn_train(
                        y_eeg_exp.squeeze(), yb_eeg_t.squeeze()
                    )
                if LAMBDA_AUDIO > 0.0:
                    total_loss = total_loss + LAMBDA_AUDIO * loss_fn_train(
                        y_aud_exp.squeeze(), yb_aud_t.squeeze()
                    )

        if use_amp:
            scaler.scale(total_loss).backward()
            if clip_grad is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            total_loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        loss_sum  += float(total_loss.item()) * xb_eeg.size(0)
        n_samples += xb_eeg.size(0)

    return loss_sum / max(1, n_samples)


def evaluate(model, loader, loss_fn_eval):
    model.eval()
    n_samples, loss_sum = 0, 0.0
    all_t, all_p = [], []
    use_amp = (device.type == "cuda" and USE_AMP)

    with torch.no_grad():
        ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            for xb_eeg, xb_audio, yb_eeg, yb_audio, _ in loader:
                xb_eeg   = xb_eeg.to(device, non_blocking=True)
                xb_audio = xb_audio.to(device, non_blocking=True)
                yb_eeg   = yb_eeg.to(device, non_blocking=True)
                yb_audio = yb_audio.to(device, non_blocking=True)

                yb_final = select_batch_target(TARGET_FINAL_KIND, yb_eeg, yb_audio)

                preds, _, _, _ = model(xb_eeg, xb_audio)
                preds    = preds.squeeze()
                yb_final = yb_final.squeeze()

                loss = loss_fn_eval(preds, yb_final)

                all_t.append(yb_final.detach().cpu().numpy())
                all_p.append(preds.detach().float().cpu().numpy())

                loss_sum  += float(loss.item()) * xb_eeg.size(0)
                n_samples += xb_eeg.size(0)

    yt = np.concatenate(all_t)
    yp = np.concatenate(all_p)
    return {"loss": loss_sum / max(1, n_samples), "sample": metrics_from_preds_reg(yt, yp)}


# ------------------------------------------------
# Train with CONTENT-HELD-OUT OUTER + INNER splits
# ------------------------------------------------
all_results_rows = []
per_fold_rows = []

log_resource_usage("before_stimulus_disjoint_cv")

for fold_idx, (trv_idx, te_idx) in enumerate(outer_splits, 1):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # INNER split at stimulus-level within trv (NOT at window-level)
    trv_stim_unique = np.unique(stimulus_id_all[trv_idx])
    y_trv_stim = stim_mean.loc[trv_stim_unique].to_numpy()

    inner_bins, inner_q = make_group_bins(y_trv_stim, n_splits=INNER_FOLDS, max_bins=MAX_BINS)
    if inner_bins is not None:
        inner_splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold_idx)
        s_tr2, s_va2 = next(inner_splitter.split(trv_stim_unique, inner_bins))
    else:
        inner_splitter = KFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold_idx)
        s_tr2, s_va2 = next(inner_splitter.split(trv_stim_unique))

    train_stim = set(trv_stim_unique[s_tr2])
    val_stim   = set(trv_stim_unique[s_va2])
    test_stim  = set(np.unique(stimulus_id_all[te_idx]))

    tr_idx = trv_idx[np.isin(stimulus_id_all[trv_idx], list(train_stim))]
    va_idx = trv_idx[np.isin(stimulus_id_all[trv_idx], list(val_stim))]

    # Stimulus-disjoint checks (HARD guarantees)
    assert set(stimulus_id_all[tr_idx]).isdisjoint(set(stimulus_id_all[va_idx])), "Stimulus overlap train/val!"
    assert set(stimulus_id_all[tr_idx]).isdisjoint(set(stimulus_id_all[te_idx])), "Stimulus overlap train/test!"
    assert set(stimulus_id_all[va_idx]).isdisjoint(set(stimulus_id_all[te_idx])), "Stimulus overlap val/test!"

    # Audio-window disjoint checks (strong sanity checks)
    assert set(audio_window_id_all[tr_idx]).isdisjoint(set(audio_window_id_all[va_idx])), "Audio-window overlap train/val!"
    assert set(audio_window_id_all[tr_idx]).isdisjoint(set(audio_window_id_all[te_idx])), "Audio-window overlap train/test!"
    assert set(audio_window_id_all[va_idx]).isdisjoint(set(audio_window_id_all[te_idx])), "Audio-window overlap val/test!"

    # Reporting groups (subject×stimulus×session)
    train_groups = set(np.unique(group_ids_all[tr_idx]))
    val_groups   = set(np.unique(group_ids_all[va_idx]))
    test_groups  = set(np.unique(group_ids_all[te_idx]))

    # EEG normalization (fold-wise if leak-safe; else reuse global)
    if LEAK_SAFE:
        np.copyto(X_eeg_all, X_eeg_backup)  # restore base
        X_eeg_fold = X_eeg_all
        if NORM_SUBJECT:
            X_eeg_fold = apply_norm_subject_trainstats_scalar(X_eeg_fold, subjects_all, train_idx=tr_idx)
        if NORM_CHANNEL:
            X_eeg_fold = apply_norm_channel_trainstats(X_eeg_fold, train_idx=tr_idx)
    else:
        X_eeg_fold = X_eeg_norm_all

    # Datasets / loaders (weights only used for TRAIN)
    ds_tr = EEGAudioDS(X_eeg_fold, X_audio_all, y_eeg_all, y_audio_all, subjects_all, tr_idx)
    ds_va = EEGAudioDS(X_eeg_fold, X_audio_all, y_eeg_all, y_audio_all, subjects_all, va_idx)
    ds_te = EEGAudioDS(X_eeg_fold, X_audio_all, y_eeg_all, y_audio_all, subjects_all, te_idx)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = MoEFusionModelStrategy1(
        window_sec=WINDOW_SEC,
        p_drop=0.3,
        eeg_model_size=EEG_MODEL_SIZE,
        audio_adapter_type=AUDIO_ADAPTER_TYPE,
        expert_hidden=EXPERT_HIDDEN_DIM,
        gate_hidden=GATE_HIDDEN_DIM,
    ).to(device)

    md_tot, md_trn = count_params(model)
    print(
        f"\n===== Fold {fold_idx} (CONTENT-HELD-OUT) "
        f"Ntr={len(tr_idx):,} Nva={len(va_idx):,} Nte={len(te_idx):,} | "
        f"stim(tr/va/te)={len(train_stim)}/{len(val_stim)}/{len(test_stim)} | "
        f"groups(tr/va/te)={len(train_groups)}/{len(val_groups)}/{len(test_groups)} ====="
    )
    print(f"Model params: total={md_tot:,} trainable={md_trn:,}")

    # Losses
    loss_fn_train = CCCLoss().to(device)
    loss_fn_eval  = CCCLoss().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and USE_AMP))

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_epoch(
            model, dl_tr, opt, scaler, loss_fn_train,
            clip_grad=CLIP_GRAD
        )

        ev_tr = evaluate(model, dl_tr, loss_fn_eval)
        ev_va = evaluate(model, dl_va, loss_fn_eval)

        key = ev_va["loss"]
        improved = key < best_val - 1e-9
        if improved:
            best_val = key
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": ep
            }
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"ep{ep:03d} | train_obj={tr_loss:.4f} | "
            f"train:[{mstr(ev_tr['sample'])}] | "
            f"val_loss={ev_va['loss']:.4f} [{mstr(ev_va['sample'])}] | "
            f"{'★' if improved else ''}"
        )

        if (ep % 10 == 0) or improved:
            log_resource_usage(f"fold={fold_idx} ep={ep}")

        if no_improve >= PATIENCE:
            print(f"Early stop at ep{ep:03d} (best ep {best_state['epoch']})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    ev_te = evaluate(model, dl_te, loss_fn_eval)

    row = {
        "fold": fold_idx,
        "split_protocol": "content_held_out_5fold",
        "stimulus_key_col": STIM_COL,
        "final_target": TARGET_FINAL_KIND,
        "eeg_target": TARGET_EEG_KIND,
        "audio_target": TARGET_AUDIO_KIND,
        "eeg_model": EEG_MODEL_SIZE,
        "audio_adapter": AUDIO_ADAPTER_TYPE,
        "use_expert_loss": USE_EXPERT_LOSS,
        "lambda_eeg": LAMBDA_EEG,
        "lambda_audio": LAMBDA_AUDIO,
        "test_loss": float(ev_te["loss"]),
        "test_rmse": float(ev_te["sample"]["rmse"]),
        "test_pcc": float(ev_te["sample"]["pcc"]),
        "test_ccc": float(ev_te["sample"]["ccc"]),
        "n_te_stimuli": int(len(test_stim)),
        "n_te_groups": int(len(test_groups)),
        "best_epoch": int(best_state["epoch"]) if best_state is not None else -1,
        "trial_group_keys": "|".join([str(x) for x in GROUP_KEYS]),
        "audio_id_keys": "|".join([str(x) for x in AUDIO_ID_KEYS]),
    }
    per_fold_rows.append(row)
    all_results_rows.append(row)

    print(
        f"TEST: loss={ev_te['loss']:.4f} "
        f"RMSE={ev_te['sample']['rmse']:.4f} "
        f"PCC={ev_te['sample']['pcc']:.4f} "
        f"CCC={ev_te['sample']['ccc']:.4f} | "
        f"⏱ {(time.time() - t0) / 60:.1f} min"
    )
    log_resource_usage(f"end_fold {fold_idx}")

    # cleanup
    del model, opt, scaler, loss_fn_train, loss_fn_eval
    del ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te
    del best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------
# Summary + Save
# ------------------------------------------------
df = pd.DataFrame(per_fold_rows).set_index("fold").sort_index()

print("\n===== PER-FOLD TEST (CONTENT-HELD-OUT EEG+Audio, CCC) =====")
print(" fold  test_loss  test_rmse  test_pcc  test_ccc  n_te_stimuli  n_te_groups  best_epoch")
for f, r in df.iterrows():
    print(
        f"{f:5d}  {r['test_loss']:.6f}   {r['test_rmse']:.6f}   "
        f"{r['test_pcc']:.6f}  {r['test_ccc']:.6f}  "
        f"{int(r['n_te_stimuli']):11d}  {int(r['n_te_groups']):10d}  {int(r['best_epoch']):9d}"
    )

print("\n===== TEST MEAN ± STD (CONTENT-HELD-OUT) =====")
stats = {}
for k in ["test_loss", "test_rmse", "test_pcc", "test_ccc"]:
    mu = df[k].mean()
    sd = df[k].std(ddof=0)
    stats[k + "_mean"] = float(mu)
    stats[k + "_std"]  = float(sd)
    print(f"{k}: {mu:.4f} ± {sd:.4f}")

df_all = pd.DataFrame(all_results_rows)
per_fold_path = os.path.join(RESULTS_DIR, "per_fold_results_moe_strategy1_ssl_CONTENT_HELD_OUT_CCC.csv")
summary_path  = os.path.join(RESULTS_DIR, "summary_results_moe_strategy1_ssl_STIMULUS_DISJOINT_CCC.csv")

df_all.to_csv(per_fold_path, index=False)

pd.DataFrame([{
    "split_protocol": "content_held_out_5fold",
    "stimulus_key_col": STIM_COL,
    "final_target": TARGET_FINAL_KIND,
    "eeg_target": TARGET_EEG_KIND,
    "audio_target": TARGET_AUDIO_KIND,
    "eeg_model": EEG_MODEL_SIZE,
    "audio_adapter": AUDIO_ADAPTER_TYPE,
    "use_expert_loss": USE_EXPERT_LOSS,
    "lambda_eeg": LAMBDA_EEG,
    "lambda_audio": LAMBDA_AUDIO,
    "trial_group_keys": "|".join([str(x) for x in GROUP_KEYS]),
    "audio_id_keys": "|".join([str(x) for x in AUDIO_ID_KEYS]),
    "n_stimuli_total": int(len(stim_unique)),
    "n_groups_total": int(len(np.unique(group_ids_all))),
    "n_windows_total": int(len(stimulus_id_all)),
    "n_unique_audio_windows_total": int(len(np.unique(audio_window_id_all))),
    **stats
}]).to_csv(summary_path, index=False)

print(f"\nSaved per-fold results to: {per_fold_path}")
print(f"Saved summary results to:  {summary_path}")
print(f"Resource usage log:        {RESOURCE_LOG_PATH}")
