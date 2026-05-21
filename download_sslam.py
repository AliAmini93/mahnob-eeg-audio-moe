# download_sslam_pretrain.py

import os
import torch
from transformers import AutoModel

# -------------------------
# Config
# -------------------------
BASE_DIR = os.environ.get("MAHNOB_BASE_DIR", "/path/to/HCI_Tagging_Database")
MODEL_ID = "ta012/SSLAM_pretrain"  # HF repo id
SAVE_DIR = os.environ.get("SSLAM_SAVE_DIR", os.path.join(BASE_DIR, "pretrained", "SSLAM_pretrain"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print(f"Loading SSLAM model: {MODEL_ID}")
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
model.to(device)
model.eval()

hidden_size = getattr(model.config, "hidden_size", None)
print("Model config:", model.config)
print("Hidden size (embedding dim) =", hidden_size)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters   : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Saving model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR)

print("SSLAM download and save finished.")
