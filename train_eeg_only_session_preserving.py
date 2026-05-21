# ============================================================
# EEG-ONLY REGRESSION — SESSION-PRESERVING 5-FOLD CV (outer+inner)
# + LEAK-SAFE EEG NORMALIZATION
#
# EEG-only: uses ONLY X_eeg and y_eeg (no audio load, no alignment).
#
# Diagnose-first NaN/Inf handling:
#   NAN_POLICY:
#     "diagnose"  -> do NOT modify data; only locate/report offenders (default)
#     "drop"      -> drop windows that have non-finite X or y_target
#     "replace_x" -> replace non-finite EEG values with 0.0 (keep labels)
#     "clip_x"    -> clip EEG to [-CLIP_VAL, CLIP_VAL] after normalization (stability)
# ============================================================

import os, time, random, warnings, subprocess, gc
import numpy as np
import pandas as pd
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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
EEG_ROOT = os.path.join(BASE_DIR, "Continuous EEG Time-series dataset")

WINDOW_SEC     = 10
EEG_ICA_STATUS = "WithICA"
EEG_FOLDER     = f"Cont_{EEG_ICA_STATUS}_{WINDOW_SEC}sec"
EEG_DATA_DIR   = os.path.join(EEG_ROOT, EEG_FOLDER)

print("EEG data dir  :", EEG_DATA_DIR)

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
LEAK_SAFE      = True
NORM_KIND      = "bn"          # {'bn','gn'} (if you see instability, try "gn")
USE_TEMP_POOLS = False

# ---- Model config ----
EEG_MODEL_SIZE = "base"  # {"lite","base"}

# CV config
OUTER_FOLDS = 5
INNER_FOLDS = 5
MAX_BINS    = 5

RESULTS_DIR = os.path.join(
    BASE_DIR,
    f"results_EEG_only_SESSION_PRESERVING_{WINDOW_SEC}sec_target_eeg_CCC"
)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESOURCE_LOG_PATH = os.path.join(RESULTS_DIR, "resource_usage.log")


# -------------------------
# NaN/Inf diagnosis + policy
# -------------------------
NAN_POLICY = "diagnose"  # "diagnose" | "drop" | "replace_x" | "clip_x"
CLIP_VAL   = 10.0
MAX_REPORT = 20


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
# NaN/Inf diagnostics
# ------------------------------------------------
def _nonfinite_mask_eeg(X):
    # X: (N, C, T)
    return ~np.isfinite(X).all(axis=(1, 2))

def _report_nonfinite_windows(tag, bad_idx, group_id_all=None, y=None, subs=None):
    nbad = len(bad_idx)
    if nbad == 0:
        print(f"[SANITY:{tag}] OK (no non-finite windows)")
        return

    print(f"[SANITY:{tag}] FOUND {nbad} non-finite windows.")
    show = bad_idx[:MAX_REPORT]
    rows = []
    for i in show:
        row = {"idx": int(i)}
        if group_id_all is not None:
            row["stimulus"] = str(group_id_all[i])
        if subs is not None:
            row["subj"] = str(subs[i])
        if y is not None:
            row["y"] = float(y[i]) if np.isfinite(y[i]) else str(y[i])
        rows.append(row)
    print(pd.DataFrame(rows))

def sanity_check_arrays(stage, X_eeg, y_target, group_id_all=None, subs=None):
    bad_x = np.where(_nonfinite_mask_eeg(X_eeg))[0]
    bad_y = np.where(~np.isfinite(y_target))[0]

    if len(bad_x) == 0 and len(bad_y) == 0:
        print(f"[SANITY:{stage}] OK (X and y are finite)")
        return bad_x, bad_y

    if len(bad_x) > 0:
        _report_nonfinite_windows(stage + ":X", bad_x, group_id_all, y_target, subs)
    if len(bad_y) > 0:
        _report_nonfinite_windows(stage + ":y", bad_y, group_id_all, y_target, subs)
    return bad_x, bad_y

def apply_nan_policy(X_eeg, y_target, group_id_all, subs):
    bad_x = _nonfinite_mask_eeg(X_eeg)
    bad_y = ~np.isfinite(y_target)
    bad = bad_x | bad_y
    nbad = int(bad.sum())

    if nbad == 0:
        return X_eeg, y_target, group_id_all, subs

    print(f"[NAN_POLICY] Detected {nbad} bad windows. Policy='{NAN_POLICY}'")

    if NAN_POLICY == "diagnose":
        print("[NAN_POLICY] diagnose -> no modification applied.")
        return X_eeg, y_target, group_id_all, subs

    if NAN_POLICY == "drop":
        keep = ~bad
        print(f"[NAN_POLICY] drop -> keeping {int(keep.sum())} / {len(keep)} windows.")
        return X_eeg[keep], y_target[keep], group_id_all[keep], subs[keep]

    if NAN_POLICY == "replace_x":
        X_eeg = X_eeg.copy()
        mask = ~np.isfinite(X_eeg)
        cnt = int(mask.sum())
        print(f"[NAN_POLICY] replace_x -> replacing {cnt} EEG elements with 0.0")
        X_eeg[mask] = 0.0
        if bad_y.any():
            print("[NAN_POLICY] WARNING: y_target has non-finite entries; consider NAN_POLICY='drop'.")
        return X_eeg, y_target, group_id_all, subs

    if NAN_POLICY == "clip_x":
        print("[NAN_POLICY] clip_x -> will clip per-fold after normalization.")
        return X_eeg, y_target, group_id_all, subs

    raise ValueError(f"Unknown NAN_POLICY: {NAN_POLICY}")


# ------------------------------------------------
# Load EEG dataset (EEG-only)
# ------------------------------------------------
print("\nLoading EEG dataset...")
X_eeg    = np.load(os.path.join(EEG_DATA_DIR, "X_input.npy")).astype("float32")
y_eeg    = np.load(os.path.join(EEG_DATA_DIR, "y_target.npy")).astype("float32")
meta_eeg = pd.read_csv(os.path.join(EEG_DATA_DIR, "metadata.csv"))

if y_eeg.ndim == 2:
    y_eeg = y_eeg.squeeze(-1)

N, C, T = X_eeg.shape
print("  X_eeg shape:", X_eeg.shape, "y_eeg shape:", y_eeg.shape)
print("  eeg meta columns:", meta_eeg.columns.tolist())
assert C == 32, f"Expected 32 EEG channels, got {C}"

# subjects
if "subject_id" in meta_eeg.columns:
    subjects_all = meta_eeg["subject_id"].astype(str).values
else:
    subjects_all = np.array(["0"] * len(X_eeg), dtype=str)

# session_id/group_id (EEG-only)
def build_session_group_id(meta: pd.DataFrame):
    """Session-preserving group: all windows from one (subject, trial) stay together."""
    keys = [k for k in ["subject_id", "trial_id"] if k in meta.columns]
    if len(keys) < 2:
        keys = [k for k in ["session_id", "subject_id", "trial_id"] if k in meta.columns]
    if len(keys) == 0:
        raise RuntimeError("Cannot build session-preserving groups: missing subject_id/trial_id/session_id.")
    return meta[keys].astype(str).agg("_".join, axis=1).values, keys

group_id_all, GROUP_KEYS = build_session_group_id(meta_eeg)  # variable name kept for downstream compatibility
y_target_all    = y_eeg.astype("float32")

print(f"\n[TARGET] EEG-only | y_target shape = {y_target_all.shape}")
print(f"[SESSION] Unique session groups: {len(np.unique(group_id_all))}")
print(f"[SESSION] Group keys: {GROUP_KEYS}")

# sanity (no changes yet)
_ = sanity_check_arrays("post_load", X_eeg, y_target_all, group_id_all, subjects_all)

# apply optional policy (default diagnose -> no change)
X_eeg, y_target_all, group_id_all, subjects_all = apply_nan_policy(
    X_eeg, y_target_all, group_id_all, subjects_all
)

log_resource_usage("after_eeg_load")


# ------------------------------------------------
# Robust binning for stratified splitting
# ------------------------------------------------
def make_group_bins(y_group: np.ndarray, n_splits: int, max_bins: int):
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
# Build OUTER splits at session level (stratified on session-mean target)
# ------------------------------------------------
sdf = pd.DataFrame({"group_id": group_id_all, "y": y_target_all})
session_mean   = sdf.groupby("group_id")["y"].mean()
sessions_unique  = session_mean.index.to_numpy()
y_session          = session_mean.to_numpy()

outer_bins, outer_q = make_group_bins(y_session, n_splits=OUTER_FOLDS, max_bins=MAX_BINS)

if outer_bins is not None:
    print(f"[SPLIT] OUTER: StratifiedKFold on {len(sessions_unique)} sessions (bins={outer_q})")
    outer_splitter = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_iter = outer_splitter.split(sessions_unique, outer_bins)
else:
    print(f"[SPLIT] OUTER: Fallback to KFold on {len(sessions_unique)} sessions")
    outer_splitter = KFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_iter = outer_splitter.split(sessions_unique)

outer_splits = []
for s_tr, s_te in outer_iter:
    tr_sessions = set(sessions_unique[s_tr])
    te_sessions = set(sessions_unique[s_te])

    trv_idx = np.where(np.isin(group_id_all, list(tr_sessions)))[0]
    te_idx  = np.where(np.isin(group_id_all, list(te_sessions)))[0]

    assert set(group_id_all[trv_idx]).isdisjoint(set(group_id_all[te_idx])), \
        "GROUP OVERLAP: train and test share a session group!"

    outer_splits.append((trv_idx, te_idx))

print(f"[SESSION-PRESERVING CV] unique groups={len(sessions_unique)} | windows={len(group_id_all)}")


# ------------------------------------------------
# Loss & metrics (stable CCC)
# ------------------------------------------------
class CCCLossStable(nn.Module):
    """1 - CCC using direct mean/var/cov (numerically stable)."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)

        mx = x.mean()
        my = y.mean()
        vx = ((x - mx) ** 2).mean()
        vy = ((y - my) ** 2).mean()
        cov = ((x - mx) * (y - my)).mean()

        ccc = (2.0 * cov) / (vx + vy + (mx - my) ** 2 + self.eps)
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
        mu = X_all[idx].mean(axis=(0, 1, 2), keepdims=True)
        sd = X_all[idx].std (axis=(0, 1, 2), keepdims=True) + eps
        X_all[idx] = (X_all[idx] - mu) / sd
    return X_all


def apply_norm_subject_trainstats_scalar(X_all, subs_all, train_idx, eps=1e-7):
    train_mask = np.zeros(len(X_all), dtype=bool)
    train_mask[train_idx] = True

    mu_global = X_all[train_idx].mean(axis=(0, 1, 2), keepdims=True)
    sd_global = X_all[train_idx].std (axis=(0, 1, 2), keepdims=True) + eps

    for sid in np.unique(subs_all):
        idx_all = np.where(subs_all == sid)[0]
        idx_tr  = idx_all[train_mask[idx_all]]
        if len(idx_tr) == 0:
            mu, sd = mu_global, sd_global
        else:
            mu = X_all[idx_tr].mean(axis=(0, 1, 2), keepdims=True)
            sd = X_all[idx_tr].std (axis=(0, 1, 2), keepdims=True) + eps
        X_all[idx_all] = (X_all[idx_all] - mu) / sd
    return X_all


def apply_norm_channel_global(X_all, eps=1e-7):
    mu = X_all.mean(axis=(0, 2), keepdims=True)
    sd = X_all.std (axis=(0, 2), keepdims=True) + eps
    X_all[:] = (X_all - mu) / sd
    return X_all


def apply_norm_channel_trainstats(X_all, train_idx, eps=1e-7):
    mu = X_all[train_idx].mean(axis=(0, 2), keepdims=True)
    sd = X_all[train_idx].std (axis=(0, 2), keepdims=True) + eps
    X_all[:] = (X_all - mu) / sd
    return X_all


# Precompute EEG normalization when LEAK_SAFE = False
if not LEAK_SAFE:
    print("\nApplying global EEG normalization (subject/channel)...")
    X_eeg_norm_all = X_eeg.copy()
    if NORM_SUBJECT:
        X_eeg_norm_all = apply_norm_subject_global_scalar(X_eeg_norm_all, subjects_all)
    if NORM_CHANNEL:
        X_eeg_norm_all = apply_norm_channel_global(X_eeg_norm_all)
    if NAN_POLICY == "clip_x":
        X_eeg_norm_all = np.clip(X_eeg_norm_all, -CLIP_VAL, CLIP_VAL)
    log_resource_usage("after_global_eeg_norm")
else:
    X_eeg_norm_all = None
    X_eeg_backup = X_eeg.copy()  # restore each fold


# ------------------------------------------------
# EEG encoder (Stage-A) (kernel/stride adapts mildly to T)
# ------------------------------------------------
def _gn(num_channels, groups):
    groups = min(groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)

def _norm1d(in_channels):
    return nn.BatchNorm1d(in_channels) if NORM_KIND == "bn" else _gn(in_channels, groups=32)

def _auto_k0(time_len: int):
    # pick an odd kernel <= 41 and <= time_len (at least 7 if possible)
    k = min(41, time_len if time_len % 2 == 1 else time_len - 1)
    k = max(7, k)
    if k > time_len:
        k = time_len if time_len % 2 == 1 else max(1, time_len - 1)
    return int(max(1, k))

def _auto_s0(time_len: int):
    if time_len >= 512:
        return 4
    if time_len >= 256:
        return 2
    return 1

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
        B, CF, Tt = x.shape
        C, Fp = self.C, self.F
        scores = self.scorer(x)
        alpha  = torch.softmax(scores, dim=-1)
        x4 = x.view(B, C, Fp, Tt)
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
    def __init__(self, C=32, time_len=1280, dropout=0.1, D_embed=256, model_size="base"):
        super().__init__()

        if model_size == "base":
            F_per, D_gat, nheads = 16, 64, 4
            dilations = [1,2,4,8,16,32,64,128] if time_len >= 512 else [1,2,4,8,16,32,64]
        elif model_size == "lite":
            F_per, D_gat, nheads = 8, 32, 2
            dilations = [1,2,4,8,16,32,64] if time_len >= 256 else [1,2,4,8,16,32]
        else:
            raise ValueError(f"Unknown EEG model_size: {model_size}")

        k0 = _auto_k0(time_len)
        s0 = _auto_s0(time_len)

        self.C, self.F_per = C, F_per
        self.stem  = DepthwiseTemporalStem(C=C, F_per=F_per, k0=k0, s0=s0, dropout=0.05)
        self.pool1 = nn.AvgPool1d(4) if USE_TEMP_POOLS else nn.Identity()
        self.tcn   = TCNStackDW(C=C, F_per=F_per, dilations=dilations, dropout=dropout, k=3)
        self.pool2 = nn.AvgPool1d(8) if USE_TEMP_POOLS else nn.Identity()
        self.tpool = TemporalAttnPool1D(C=C, F_per=F_per)
        self.gat   = ChannelGAT(C=C, F_in=F_per, D_gat=D_gat, nheads=nheads, dropout=dropout)
        self.cpool = ChannelAttnPool(D_gat=D_gat, dropout=dropout)
        self.proj  = nn.Sequential(nn.LayerNorm(D_gat), nn.Linear(D_gat, D_embed))

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(x)
        x = self.tcn(x)
        x = self.pool2(x)
        h = self.tpool(x)
        h = self.gat(h)
        z = self.cpool(h)
        return self.proj(z)

class EEGRegressor(nn.Module):
    def __init__(self, time_len=1280, p_drop=0.3, eeg_model_size="lite"):
        super().__init__()
        self.D_eeg = 256 if eeg_model_size == "base" else 128
        self.enc = EEGEncoderStageA(C=32, time_len=time_len, dropout=0.1, D_embed=self.D_eeg, model_size=eeg_model_size)
        self.head = nn.Sequential(
            nn.LayerNorm(self.D_eeg),
            nn.Linear(self.D_eeg, 128),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 1),
        )

    def forward(self, x_eeg):
        z = self.enc(x_eeg)
        y = self.head(z)
        return y


# ------------------------------------------------
# Dataset
# ------------------------------------------------
class EEGDS(Dataset):
    def __init__(self, X_eeg, y, subs, stim, idx):
        self.X_eeg = X_eeg
        self.y = y
        self.idx = np.asarray(idx, dtype=int)
        self.subs = subs[self.idx]
        self.stim = stim[self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gi = int(self.idx[i])
        return (
            torch.from_numpy(self.X_eeg[gi]),
            torch.tensor(self.y[gi], dtype=torch.float32),
            self.subs[i],
            self.stim[i],
            gi,   # global index for debug
        )


# ------------------------------------------------
# Train / Eval with NaN guards
# ------------------------------------------------
def _isfinite_torch(t):
    return torch.isfinite(t).all().item()

def train_epoch(model, loader, opt, scaler, loss_fn, clip_grad=None):
    model.train()
    loss_sum, n_samples = 0.0, 0
    use_amp = (device.type == "cuda" and USE_AMP)

    for xb_eeg, yb, _, _, gidx in loader:
        xb_eeg = xb_eeg.to(device, non_blocking=True)
        yb     = yb.to(device, non_blocking=True).reshape(-1)

        if not _isfinite_torch(xb_eeg):
            print("[NAN-GUARD] Non-finite xb_eeg in train batch. gidx[:10]=", gidx[:10].tolist())
            return float("nan")

        opt.zero_grad(set_to_none=True)

        ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            preds = model(xb_eeg).reshape(-1)

            if not _isfinite_torch(preds):
                print("[NAN-GUARD] Non-finite preds in train batch. gidx[:10]=", gidx[:10].tolist())
                return float("nan")

            loss = loss_fn(preds, yb)
            if not torch.isfinite(loss).item():
                print("[NAN-GUARD] Non-finite loss in train batch. gidx[:10]=", gidx[:10].tolist())
                return float("nan")

        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        bs = xb_eeg.size(0)
        loss_sum  += float(loss.item()) * bs
        n_samples += bs

    return loss_sum / max(1, n_samples)

def evaluate(model, loader, loss_fn):
    model.eval()
    n_samples, loss_sum = 0, 0.0
    all_t, all_p = [], []
    use_amp = (device.type == "cuda" and USE_AMP)

    with torch.no_grad():
        ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            for xb_eeg, yb, _, _, gidx in loader:
                xb_eeg = xb_eeg.to(device, non_blocking=True)
                yb     = yb.to(device, non_blocking=True).reshape(-1)

                if not _isfinite_torch(xb_eeg):
                    print("[NAN-GUARD] Non-finite xb_eeg in eval batch. gidx[:10]=", gidx[:10].tolist())
                    return {"loss": float("nan"), "sample": {"rmse": float("nan"), "pcc": 0.0, "ccc": float("nan")}}

                preds = model(xb_eeg).reshape(-1)
                if not _isfinite_torch(preds):
                    print("[NAN-GUARD] Non-finite preds in eval batch. gidx[:10]=", gidx[:10].tolist())
                    return {"loss": float("nan"), "sample": {"rmse": float("nan"), "pcc": 0.0, "ccc": float("nan")}}

                loss  = loss_fn(preds, yb)
                if not torch.isfinite(loss).item():
                    print("[NAN-GUARD] Non-finite loss in eval batch. gidx[:10]=", gidx[:10].tolist())
                    return {"loss": float("nan"), "sample": {"rmse": float("nan"), "pcc": 0.0, "ccc": float("nan")}}

                all_t.append(yb.detach().cpu().numpy().reshape(-1))
                all_p.append(preds.detach().float().cpu().numpy().reshape(-1))

                bs = xb_eeg.size(0)
                loss_sum  += float(loss.item()) * bs
                n_samples += bs

    yt = np.concatenate(all_t, axis=0)
    yp = np.concatenate(all_p, axis=0)
    return {"loss": loss_sum / max(1, n_samples), "sample": metrics_from_preds_reg(yt, yp)}


# ------------------------------------------------
# Run SESSION-PRESERVING OUTER + INNER CV
# ------------------------------------------------
all_results_rows = []
per_fold_rows = []

log_resource_usage("before_cv")

for fold_idx, (trv_idx, te_idx) in enumerate(outer_splits, 1):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # INNER split: group-level within trv
    trv_sessions_unique = np.unique(group_id_all[trv_idx])
    y_trv_stim = session_mean.loc[trv_sessions_unique].to_numpy()

    inner_bins, inner_q = make_group_bins(y_trv_stim, n_splits=INNER_FOLDS, max_bins=MAX_BINS)
    if inner_bins is not None:
        inner_splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold_idx)
        s_tr2, s_va2 = next(inner_splitter.split(trv_sessions_unique, inner_bins))
    else:
        inner_splitter = KFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold_idx)
        s_tr2, s_va2 = next(inner_splitter.split(trv_sessions_unique))

    train_sessions = set(trv_sessions_unique[s_tr2])
    val_sessions   = set(trv_sessions_unique[s_va2])
    test_sessions  = set(np.unique(group_id_all[te_idx]))

    tr_idx = np.where(np.isin(group_id_all, list(train_sessions)))[0]
    va_idx = np.where(np.isin(group_id_all, list(val_sessions)))[0]

    assert set(group_id_all[tr_idx]).isdisjoint(set(group_id_all[va_idx]))
    assert set(group_id_all[tr_idx]).isdisjoint(set(group_id_all[te_idx]))
    assert set(group_id_all[va_idx]).isdisjoint(set(group_id_all[te_idx]))

    # Restore raw EEG each fold (LEAK_SAFE)
    if LEAK_SAFE:
        np.copyto(X_eeg, X_eeg_backup)

    # --- fold diagnosis BEFORE normalization ---
    print(f"\n[FOLD {fold_idx}] Pre-norm sanity:")
    sanity_check_arrays(
        f"fold{fold_idx}_pre_norm",
        X_eeg, y_target_all,
        group_id_all, subjects_all
    )

    # EEG normalization (fold-wise if leak-safe; else reuse global)
    if LEAK_SAFE:
        X_eeg_fold = X_eeg
        if NORM_SUBJECT:
            X_eeg_fold = apply_norm_subject_trainstats_scalar(X_eeg_fold, subjects_all, train_idx=tr_idx)
        if NORM_CHANNEL:
            X_eeg_fold = apply_norm_channel_trainstats(X_eeg_fold, train_idx=tr_idx)
        if NAN_POLICY == "clip_x":
            X_eeg_fold = np.clip(X_eeg_fold, -CLIP_VAL, CLIP_VAL)
    else:
        X_eeg_fold = X_eeg_norm_all

    # --- fold diagnosis AFTER normalization ---
    print(f"[FOLD {fold_idx}] Post-norm sanity:")
    sanity_check_arrays(
        f"fold{fold_idx}_post_norm",
        X_eeg_fold, y_target_all,
        group_id_all, subjects_all
    )

    # Datasets / loaders
    ds_tr = EEGDS(X_eeg_fold, y_target_all, subjects_all, group_id_all, tr_idx)
    ds_va = EEGDS(X_eeg_fold, y_target_all, subjects_all, group_id_all, va_idx)
    ds_te = EEGDS(X_eeg_fold, y_target_all, subjects_all, group_id_all, te_idx)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = EEGRegressor(time_len=T, p_drop=0.3, eeg_model_size=EEG_MODEL_SIZE).to(device)
    md_tot, md_trn = count_params(model)

    print(
        f"\n===== Fold {fold_idx} (EEG-ONLY SESSION-PRESERVING) "
        f"Ntr={len(tr_idx):,} Nva={len(va_idx):,} Nte={len(te_idx):,} | "
        f"groups_tr={len(train_sessions)} groups_va={len(val_sessions)} groups_te={len(test_sessions)} ====="
    )
    print(f"Model params: total={md_tot:,} trainable={md_trn:,}")

    loss_fn = CCCLossStable().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and USE_AMP))

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    aborted = False

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, dl_tr, opt, scaler, loss_fn, clip_grad=CLIP_GRAD)
        ev_tr = evaluate(model, dl_tr, loss_fn)
        ev_va = evaluate(model, dl_va, loss_fn)

        if (not np.isfinite(tr_loss)) or (not np.isfinite(ev_va["loss"])) or (not np.isfinite(ev_tr["loss"])):
            print("\n[ABORT] NaN/Inf detected in training/eval metrics.")
            print("  Fold:", fold_idx)
            print("  Train groups:", sorted(list(train_sessions))[:20], "..." if len(train_sessions) > 20 else "")
            print("  Val groups  :", sorted(list(val_sessions))[:20], "..." if len(val_sessions) > 20 else "")
            print("  Test groups :", sorted(list(test_sessions))[:20], "..." if len(test_sessions) > 20 else "")
            aborted = True
            break

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

    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        best_epoch = int(best_state["epoch"])
    else:
        best_epoch = -1

    ev_te = evaluate(model, dl_te, loss_fn)

    row = {
        "fold": fold_idx,
        "eeg_model": EEG_MODEL_SIZE,
        "nan_policy": NAN_POLICY,
        "norm_kind": NORM_KIND,
        "test_loss": float(ev_te["loss"]),
        "test_rmse": float(ev_te["sample"]["rmse"]),
        "test_pcc": float(ev_te["sample"]["pcc"]),
        "test_ccc": float(ev_te["sample"]["ccc"]),
        "n_te_groups": int(len(test_sessions)),
        "best_epoch": best_epoch,
        "aborted": bool(aborted),
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
    del model, opt, scaler, loss_fn
    del ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te
    del best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------
# Summary + Save
# ------------------------------------------------
df = pd.DataFrame(per_fold_rows).set_index("fold").sort_index()

print("\n===== PER-FOLD TEST (EEG-ONLY SESSION-PRESERVING) =====")
print(" fold  test_loss  test_rmse  test_pcc  test_ccc  n_te_groups  best_epoch  aborted  nan_policy  norm_kind")
for f, r in df.iterrows():
    print(
        f"{f:5d}  {r['test_loss']:.6f}   {r['test_rmse']:.6f}   "
        f"{r['test_pcc']:.6f}  {r['test_ccc']:.6f}  "
        f"{int(r['n_te_groups']):10d}  {int(r['best_epoch']):9d}  "
        f"{str(r['aborted']):7s}  {r['nan_policy']:9s}  {r['norm_kind']}"
    )

print("\n===== TEST MEAN ± STD (EEG-ONLY SESSION-PRESERVING) =====")
stats = {}
for k in ["test_loss", "test_rmse", "test_pcc", "test_ccc"]:
    mu = df[k].mean()
    sd = df[k].std(ddof=0)
    stats[k + "_mean"] = float(mu)
    stats[k + "_std"]  = float(sd)
    print(f"{k}: {mu:.4f} ± {sd:.4f}")

df_all = pd.DataFrame(all_results_rows)
per_fold_path = os.path.join(RESULTS_DIR, "per_fold_results_EEG_only_session_preserving.csv")
summary_path  = os.path.join(RESULTS_DIR, "summary_results_EEG_only_session_preserving.csv")

df_all.to_csv(per_fold_path, index=False)
pd.DataFrame([{
    "eeg_model": EEG_MODEL_SIZE,
    "nan_policy": NAN_POLICY,
    "norm_kind": NORM_KIND,
    "n_groups_total": int(len(sessions_unique)),
    "n_windows_total": int(len(group_id_all)),
    **stats
}]).to_csv(summary_path, index=False)

print(f"\nSaved per-fold results to: {per_fold_path}")
print(f"Saved summary results to:  {summary_path}")
print(f"Resource usage log:        {RESOURCE_LOG_PATH}")
