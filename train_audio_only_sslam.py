import os, time, math, random, warnings
import numpy as np
import pandas as pd
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================================================
# Repro / Device
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config
# ============================================================

BASE_DIR   = os.environ.get("MAHNOB_BASE_DIR", r"/path/to/HCI_Tagging_Database")
AUDIO_ROOT = os.path.join(BASE_DIR, "Continuous_Audio_32k")

WINDOW_SEC = 10          # 10-sec clips

FOLDER   = f"{WINDOW_SEC}sec"
DATA_DIR = os.path.join(AUDIO_ROOT, FOLDER)

# Paths
#   - embeddings: precomputed SSLAM embeddings (from compute_sslam_embeddings.py)
#   - labels:     per-window labels
#   - metadata:   per-window meta (subject_id, trial_id, media_file, label_idx, ...)
EMB_PATH       = os.path.join(DATA_DIR, "audio_emb_sslam_10sec.npy")
Y_PATH         = os.path.join(DATA_DIR, "Cont_Audio_10sec_y_32k.npy")
META_PATH      = os.path.join(DATA_DIR, f"metadata_audio_{WINDOW_SEC}sec.csv")

print("Data dir :", DATA_DIR)
print("Emb path :", EMB_PATH)
print("y path   :", Y_PATH)
print("meta     :", META_PATH)

BATCH_SIZE    = 64
LR            = 3e-4
WD            = 1e-4
EPOCHS        = 1000
PATIENCE      = 20

USE_AMP       = False

# Choose model variant: "base" (with hidden) or "lite" (no hidden)
MODEL_VARIANT = "lite"    # change to "base" if you want a larger head

# ============================================================
# Load SSLAM embeddings, labels, metadata
# ============================================================

print("\nLoading SSLAM embeddings into RAM...")
X = np.load(EMB_PATH).astype("float32")      # (N, D_ssl)
print("  X_emb shape:", X.shape)

print("Loading per-window targets...")
y = np.load(Y_PATH).astype("float32")   # (N,)
if y.ndim == 2:
    y = y.squeeze(-1)
print("  y shape:", y.shape)

print("Loading metadata...")
meta = pd.read_csv(META_PATH)
print("  meta columns:", meta.columns.tolist())

if "subject_id" in meta.columns:
    subjects = meta["subject_id"].astype(str).values
else:
    # subject info not needed for this task, but keep a dummy column for DS return
    subjects = np.array(["0"] * len(y), dtype=str)

N, D = X.shape
assert len(y) == N, "Mismatch between X and y length!"
EMB_DIM = D
print(f"\nDataset (10s audio, SSLAM emb, per-window labels): N={N}, D_ssl={D}")

# ============================================================
# Stratify session-level label summaries into bins for grouped CV
# ============================================================

num_bins = 5
try:
    y_binned = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
except ValueError:
    y_binned = pd.cut(y, bins=num_bins, labels=False)
y_binned = np.asarray(y_binned, dtype=int)

# ============================================================
# Loss and metrics
# ============================================================

class CCCLoss(nn.Module):
    """1 - Concordance Correlation Coefficient."""
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
    return 0.0 if np.isnan(r) else r

def concordance_correlation_coefficient(y_true, y_pred):
    cor = safe_pearsonr(y_true, y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.sqrt(var_true)
    sd_pred = np.sqrt(var_pred)
    num = 2 * cor * sd_true * sd_pred
    den = var_true + var_pred + (mean_true - mean_pred) ** 2
    return num / (den + 1e-8)

def metrics_from_preds_reg(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pcc  = safe_pearsonr(y_true, y_pred)
    ccc  = concordance_correlation_coefficient(y_true, y_pred)
    return {"rmse": rmse, "pcc": pcc, "ccc": ccc}

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train

# ============================================================
# Regression heads on top of SSLAM embeddings
# ============================================================

class AudioSSLAMHeadBase(nn.Module):
    """
    Base MLP head: D_ssl -> 512 -> 1.
    Predicts per-window valence for the 10s window.
    """
    def __init__(self, D_in, hidden=512, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(D_in),
            nn.Linear(D_in, hidden),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):  # [B, D_in]
        return self.net(x)  # [B, 1]

class AudioSSLAMHeadLite(nn.Module):
    """
    Lite head: D_ssl -> 1 (no hidden layer, far fewer parameters).
    Still uses LayerNorm for a bit of stabilization.
    """
    def __init__(self, D_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(D_in),
            nn.Linear(D_in, 1),
        )
    def forward(self, x):  # [B, D_in]
        return self.net(x)  # [B, 1]

# ============================================================
# Dataset
# ============================================================

class AudioEmbDS(Dataset):
    def __init__(self, X, y, subjects, idx):
        self.X = X
        self.y = y
        self.idx = np.array(idx)
        self.subs = subjects[self.idx]
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        gi = self.idx[i]
        x = self.X[gi].astype("float32")       # (D_ssl,)
        y_val = float(self.y[gi])
        sub = self.subs[i]
        return (
            torch.from_numpy(x),              # [D_ssl]
            torch.tensor(y_val, dtype=torch.float32),
            sub,
        )

# ============================================================
# Train / eval
# ============================================================

def evaluate(model, loader, loss_fn):
    model.eval()
    n_samples = 0
    loss_sum = 0.0
    all_t, all_p = [], []
    use_amp = (device.type == "cuda" and USE_AMP)

    with torch.no_grad():
        ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            for xb, yb, _ in loader:
                xb = xb.to(device, non_blocking=True)  # [B,D_ssl]
                yb = yb.to(device, non_blocking=True)  # [B]

                preds = model(xb).squeeze()            # [B]
                loss  = loss_fn(preds, yb)

                all_t.append(yb.cpu().numpy())
                all_p.append(preds.cpu().numpy())

                loss_sum += float(loss.item()) * xb.size(0)
                n_samples += xb.size(0)

    yt = np.concatenate(all_t)
    yp = np.concatenate(all_p)
    m_sample = metrics_from_preds_reg(yt, yp)
    return {"loss": loss_sum / max(1, n_samples), "sample": m_sample}

def train_epoch(model, loader, opt, scaler, loss_fn, clip_grad=None):
    model.train()
    loss_sum = 0.0
    n_samples = 0
    use_amp = (device.type == "cuda" and USE_AMP)

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            preds = model(xb).squeeze()
            loss  = loss_fn(preds, yb.squeeze())

        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        loss_sum += float(loss.item()) * xb.size(0)
        n_samples += xb.size(0)

    return loss_sum / max(1, n_samples)

# ============================================================
# Session-preserving 5-fold CV
# All windows from the same (subject_id, trial_id) session stay in one fold.
# ============================================================

def build_session_groups(meta_df: pd.DataFrame):
    keys = [k for k in ["subject_id", "trial_id"] if k in meta_df.columns]
    if len(keys) < 2:
        keys = [k for k in ["session_id", "subject_id", "trial_id"] if k in meta_df.columns]
    if len(keys) == 0:
        raise RuntimeError("Cannot build session groups: metadata lacks subject_id/trial_id/session_id.")
    return meta_df[keys].astype(str).agg("_".join, axis=1).values, keys


def make_group_bins(y_group: np.ndarray, n_splits: int, max_bins: int = 5):
    for q in range(min(max_bins, len(y_group)), 1, -1):
        try:
            bins = pd.qcut(y_group, q=q, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.cut(y_group, bins=q, labels=False)
        bins = np.asarray(bins, dtype=int)
        counts = np.bincount(bins) if bins.size else np.array([0])
        if counts.min() >= n_splits:
            return bins
    return None

from sklearn.model_selection import KFold

group_ids_all, GROUP_KEYS = build_session_groups(meta)
gdf = pd.DataFrame({"group_id": group_ids_all, "y": y})
group_mean = gdf.groupby("group_id")["y"].mean()
gids_unique = group_mean.index.to_numpy()
y_group = group_mean.to_numpy()
y_group_bins = make_group_bins(y_group, 5, 5)

if y_group_bins is not None:
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    outer_iter = outer.split(gids_unique, y_group_bins)
else:
    outer = KFold(5, shuffle=True, random_state=SEED)
    outer_iter = outer.split(gids_unique)

outer_splits = []
for g_trv, g_te in outer_iter:
    trv_groups = set(gids_unique[g_trv])
    te_groups = set(gids_unique[g_te])
    trv_idx = np.where(np.isin(group_ids_all, list(trv_groups)))[0]
    te_idx = np.where(np.isin(group_ids_all, list(te_groups)))[0]
    assert set(group_ids_all[trv_idx]).isdisjoint(set(group_ids_all[te_idx]))
    outer_splits.append((trv_idx, te_idx))

per_fold_rows = []

for f, (trv_idx, te_idx) in enumerate(outer_splits, 1):
    trv_groups = np.unique(group_ids_all[trv_idx])
    y_trv_group = group_mean.loc[trv_groups].to_numpy()
    inner_bins = make_group_bins(y_trv_group, 5, 5)
    if inner_bins is not None:
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + f)
        g_tr, g_va = next(inner.split(trv_groups, inner_bins))
    else:
        inner = KFold(5, shuffle=True, random_state=SEED + f)
        g_tr, g_va = next(inner.split(trv_groups))
    train_groups = set(trv_groups[g_tr])
    val_groups = set(trv_groups[g_va])
    tr_idx = np.where(np.isin(group_ids_all, list(train_groups)))[0]
    va_idx = np.where(np.isin(group_ids_all, list(val_groups)))[0]
    assert set(group_ids_all[tr_idx]).isdisjoint(set(group_ids_all[va_idx]))
    assert set(group_ids_all[tr_idx]).isdisjoint(set(group_ids_all[te_idx]))

    ds_tr = AudioEmbDS(X, y, subjects, tr_idx)
    ds_va = AudioEmbDS(X, y, subjects, va_idx)
    ds_te = AudioEmbDS(X, y, subjects, te_idx)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2, pin_memory=True)

    # Choose model variant
    if MODEL_VARIANT.lower() == "lite":
        model = AudioSSLAMHeadLite(D_in=EMB_DIM).to(device)
    else:
        model = AudioSSLAMHeadBase(D_in=EMB_DIM, hidden=512, p_drop=0.3).to(device)

    md_tot, md_trn = count_params(model)
    print(
        f"\n===== Fold {f}  (Ntr={len(tr_idx):,}, Nva={len(va_idx):,}, Nte={len(te_idx):,})  "
        f"task=VALENCE(REGRESSION, SSLAM-EMB, SESSION-PRESERVING)  use_amp={USE_AMP}  variant={MODEL_VARIANT} ====="
    )
    print(f"Model params: total={md_tot:,} trainable={md_trn:,}")

    loss_fn = CCCLoss().to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler  = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and USE_AMP))

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, dl_tr, opt, scaler, loss_fn, clip_grad=1.0)
        ev_tr   = evaluate(model, dl_tr, loss_fn)
        ev_va   = evaluate(model, dl_va, loss_fn)

        key = ev_va["loss"]
        improved = key < best_val - 1e-9
        if improved:
            best_val = key
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": ep,
            }
            no_improve = 0
        else:
            no_improve += 1

        def mstr(m):
            return f"rmse={m['rmse']:.4f} pcc={m['pcc']:.4f} ccc={m['ccc']:.4f}"

        print(
            f"ep{ep:03d} | "
            f"train: loss={tr_loss:.4f} [{mstr(ev_tr['sample'])}] | "
            f"val: loss={ev_va['loss']:.4f} [{mstr(ev_va['sample'])}] | "
            f"{'★' if improved else ''}"
        )
        if no_improve >= PATIENCE:
            print(f"Early stop at ep{ep:03d} (best ep {best_state['epoch']})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    ev_te = evaluate(model, dl_te, loss_fn)
    row = {
        "fold": f,
        "test_loss": ev_te["loss"],
        "test_rmse": ev_te["sample"]["rmse"],
        "test_pcc":  ev_te["sample"]["pcc"],
        "test_ccc":  ev_te["sample"]["ccc"],
    }
    per_fold_rows.append(row)

    print(
        f"TEST: loss={ev_te['loss']:.4f} "
        f"RMSE={ev_te['sample']['rmse']:.4f} "
        f"PCC={ev_te['sample']['pcc']:.4f} "
        f"CCC={ev_te['sample']['ccc']:.4f} | "
        f"⏱ {(time.time()-t0)/60:.1f} min"
    )

# ============================================================
# Summary
# ============================================================

df = pd.DataFrame(per_fold_rows).set_index("fold").sort_index()

print("\n===== PER-FOLD TEST (VALENCE REGRESSION, AUDIO SSLAM-EMB, SESSION-PRESERVING) "
      f"[variant={MODEL_VARIANT}] =====")
print(" fold  test_loss  test_rmse  test_pcc  test_ccc")
for f, r in df.iterrows():
    print(
        f"{f:5d}  {r['test_loss']:.6f}   {r['test_rmse']:.6f}   "
        f"{r['test_pcc']:.6f}  {r['test_ccc']:.6f}"
    )

print("\n===== TEST MEAN ± STD =====")
for k in ["test_loss", "test_rmse", "test_pcc", "test_ccc"]:
    mu = df[k].mean()
    sd = df[k].std(ddof=0)
    print(f"{k}: {mu:.4f} ± {sd:.4f}")


RESULTS_DIR = os.path.join(DATA_DIR, "results_audio_only_session_preserving")
os.makedirs(RESULTS_DIR, exist_ok=True)
df.to_csv(os.path.join(RESULTS_DIR, "per_fold_results_audio_only_session_preserving.csv"))
