#!/usr/bin/env python3
"""
Cell-type-specific models for predicting age from single-cell gene expression.
Uses harmonized discovery data (TM + Calico); one model per eligible cell type.
- On GPU: LSTM (sequence + attention over genes).
- On CPU (default): MLP (flat vector → dense layers), faster but different model; set USE_MLP_ON_CPU = False to use LSTM on CPU.
Run after preprocessing pipeline (00–05); expects data/discovery_combined.h5ad.
"""

import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

# Discovery data (harmonized TM + Calico)
DISCOVERY_PATH = getattr(config, "DISCOVERY_COMBINED_PATH", config.DATA_DIR / "discovery_combined.h5ad")
RESULTS_MODELS = getattr(config, "RESULTS_MODELS", config.OUTPUT_DIR / "models")
CELL_TYPE_COL = "cell_type_raw"
AGE_COL = "age_months"
MIN_CELLS = 100
MIN_TIMEPOINTS = 4
TRAIN_FRAC = 0.8
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 5
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 10
MAX_EPOCHS = 100
SEED = 42
# DataLoader workers: 0 = single process (safest with CUDA); 4+ = parallel loading (faster on CPU)
NUM_WORKERS = 24
# When on CPU: use a small MLP instead of LSTM (much faster; no recurrence over genes).
# Set True for faster CPU runs; False to always use LSTM (slower on CPU, same model as GPU).
USE_MLP_ON_CPU = False
# When on CPU and still using LSTM: cap genes to this many (shorter sequence = faster)
MAX_GENES_LSTM_CPU = 400


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AgingLSTM(nn.Module):
    """LSTM with self-attention for age regression from gene expression (sequence of genes)."""

    def __init__(self, input_dim=1, hidden_size=256, num_layers=2, dropout=0.3, projection_size=512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, projection_size),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            projection_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attn_proj = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)  # (batch, seq_len, projection_size)
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden_size*2)
        attn_logits = self.attn_proj(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_size*2)
        pred = self.fc_out(context).squeeze(-1)  # (batch,)
        return pred, attn_weights.squeeze(-1)

    @torch.no_grad()
    def get_attention_weights(self, x):
        _, attn_weights = self.forward(x)
        return attn_weights


class AgingMLP(nn.Module):
    """Small MLP for age regression from gene vector. Much faster than LSTM on CPU."""

    def __init__(self, n_genes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1) from dataset; flatten to (batch, seq_len)
        if x.dim() == 3:
            x = x.squeeze(-1)
        pred = self.net(x).squeeze(-1)
        return pred, None  # no attention for MLP

    @torch.no_grad()
    def get_attention_weights(self, x):
        return None


class AgingDataset(Dataset):
    """Dataset of gene expression and age for one cell type; age normalized to [0,1]."""

    def __init__(self, adata, cell_type, hvg_names, cell_type_col=None):
        col = cell_type_col if cell_type_col is not None else CELL_TYPE_COL
        ct_mask = (adata.obs[col].astype(str) == cell_type)
        self.adata = adata[ct_mask].copy()
        self.cell_type = cell_type
        # Restrict to HVGs present in adata
        self.genes = [g for g in hvg_names if g in self.adata.var_names]
        if len(self.genes) != len(hvg_names):
            self.adata = self.adata[:, self.genes].copy()
        else:
            self.adata = self.adata[:, self.genes].copy()
        X = self.adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = np.asarray(X, dtype=np.float32)
        age = self.adata.obs[AGE_COL].astype(float).values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.age_normalized = self.scaler.fit_transform(age.reshape(-1, 1)).ravel().astype(np.float32)
        self.age_raw = age.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return (seq_len, 1) for LSTM: each gene is a time step with one feature
        x = torch.from_numpy(self.X[idx]).unsqueeze(-1)  # (seq_len, 1)
        y = torch.tensor(self.age_normalized[idx], dtype=torch.float32)
        return x, y


def _stratified_split(age_months, train_frac, seed):
    """Train/test split stratified by age (binned) so timepoints are represented in both."""
    age = np.asarray(age_months, dtype=float)
    # Bin ages for stratification (use few bins if few unique values)
    n_bins = min(5, max(2, len(np.unique(age))))
    bins = np.percentile(age, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 1e-6
    strat = np.digitize(age, bins) - 1
    strat = np.clip(strat, 0, n_bins - 1)
    idx = np.arange(len(age))
    i_train, i_test = train_test_split(idx, test_size=1 - train_frac, stratify=strat, random_state=seed)
    return i_train, i_test


def train_lstm_per_cell_type(
    adata,
    cell_type,
    hvg_names,
    device,
    save_dir,
    figures_dir,
    cell_type_col=None,
    num_workers=0,
    use_mlp=False,
):
    """Train one model (LSTM or MLP) for the given cell type. Eligibility: >=MIN_CELLS, >=MIN_TIMEPOINTS."""
    col = cell_type_col if cell_type_col is not None else CELL_TYPE_COL
    ct_mask = (adata.obs[col].astype(str) == cell_type)
    n_cells = ct_mask.sum()
    ages = adata.obs.loc[ct_mask, AGE_COL].dropna().astype(float)
    n_timepoints = ages.nunique()

    if n_cells < MIN_CELLS or n_timepoints < MIN_TIMEPOINTS:
        return None, None, None, None, None

    full_ds = AgingDataset(adata, cell_type, hvg_names, cell_type_col=col)
    n_genes = len(full_ds.genes)
    if n_genes == 0:
        return None, None, None, None, None

    model_name = "MLP" if use_mlp else "LSTM"
    i_train, i_test = _stratified_split(full_ds.age_raw, TRAIN_FRAC, SEED)
    # Subset datasets
    train_ds = torch.utils.data.Subset(full_ds, i_train)
    test_ds = torch.utils.data.Subset(full_ds, i_test)
    # Further split train into train/val (80% of train = 64% total)
    strat = np.digitize(full_ds.age_raw[i_train], np.percentile(full_ds.age_raw, np.linspace(0, 100, 6)))
    i_tr, i_val = train_test_split(np.arange(len(i_train)), test_size=0.2, stratify=strat, random_state=SEED)
    train_ds = torch.utils.data.Subset(full_ds, i_train[i_tr])
    val_ds = torch.utils.data.Subset(full_ds, i_train[i_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    if use_mlp:
        model = AgingMLP(n_genes=n_genes, hidden=256, dropout=0.3).to(device)
    else:
        model = AgingLSTM(input_dim=1, hidden_size=256, num_layers=2, dropout=0.3, projection_size=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(MAX_EPOCHS), desc=cell_type[:28], unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", best=f"{best_val_loss:.4f}")
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            epoch_pbar.set_postfix_str(f"early stop @ epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test set metrics (in original age space)
    model.eval()
    preds_list, ages_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred_norm, _ = model(xb)
            pred_norm = pred_norm.cpu().numpy()
            # Inverse transform: full_ds.scaler is fit on all cells of this type
            pred_age = full_ds.scaler.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
            preds_list.append(pred_age)
            ages_list.append(yb.numpy())
    preds = np.concatenate(preds_list)
    # True ages (normalized then inverse transform to get raw)
    true_norm = np.concatenate(ages_list)
    true_age = full_ds.scaler.inverse_transform(true_norm.reshape(-1, 1)).ravel()
    test_r2 = r2_score(true_age, preds)
    test_mae = mean_absolute_error(true_age, preds)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{cell_type.replace(' ', '_')}_mlp_best.pt" if use_mlp else f"{cell_type.replace(' ', '_')}_lstm_best.pt"
    torch.save(
        {"model_state": model.cpu().state_dict(), "scaler_min": full_ds.scaler.min_, "scaler_scale": full_ds.scaler.scale_, "genes": full_ds.genes, "use_mlp": use_mlp},
        save_dir / ckpt_name,
    )

    # Loss curves
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized age)")
    ax.legend()
    ax.set_title(f"{model_name} loss — {cell_type}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"lstm_loss_{cell_type.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return model, train_losses, val_losses, test_r2, test_mae


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use 0 workers with CUDA (avoid fork); use NUM_WORKERS on CPU for parallel data loading
    num_workers = 0 if device.type == "cuda" else min(NUM_WORKERS, 16)
    use_mlp_on_cpu = (device.type == "cpu" and USE_MLP_ON_CPU)
    max_genes_cpu = MAX_GENES_LSTM_CPU
    print(f"Device: {device}  |  DataLoader workers: {num_workers}")
    if device.type == "cpu":
        if use_mlp_on_cpu:
            print("  Using MLP (fast CPU mode). Set USE_MLP_ON_CPU = False in script to use LSTM.")
        else:
            print(f"  Using LSTM with max {max_genes_cpu} genes on CPU. Set USE_MLP_ON_CPU = True for faster MLP.")

    discovery_path = Path(DISCOVERY_PATH)
    if not discovery_path.exists():
        print(f"Discovery data not found: {discovery_path}. Run preprocessing 00–05 first.")
        return 1

    import scanpy as sc
    adata = sc.read_h5ad(discovery_path)
    cell_type_col = CELL_TYPE_COL
    if cell_type_col not in adata.obs.columns:
        alt = "cell_ontology_class"
        if alt not in adata.obs.columns:
            print(f"Neither {CELL_TYPE_COL} nor {alt} found in adata.obs.")
            return 1
        cell_type_col = alt
    if AGE_COL not in adata.obs.columns:
        print(f"Column {AGE_COL} not found.")
        return 1

    # HVG list: use var_names (discovery may already be HVG-subset) or highly_variable
    if "highly_variable" in adata.var.columns:
        hvg_names = adata.var_names[adata.var["highly_variable"]].tolist()
    else:
        hvg_names = adata.var_names.tolist()
    if len(hvg_names) == 0:
        print("No genes found.")
        return 1
    # On CPU with LSTM, cap genes to speed up; MLP uses all
    if device.type == "cpu" and not use_mlp_on_cpu and len(hvg_names) > max_genes_cpu:
        hvg_names = hvg_names[:max_genes_cpu]
        print(f"Using {len(hvg_names)} genes (capped for CPU LSTM).")
    else:
        print(f"Using {len(hvg_names)} genes for {'MLP' if use_mlp_on_cpu else 'LSTM'} input.")

    # Eligible cell types
    ct_counts = adata.obs[cell_type_col].astype(str).value_counts()
    timepoints_per_ct = adata.obs.groupby(cell_type_col)[AGE_COL].nunique()
    eligible = [
        ct for ct in ct_counts.index
        if ct_counts[ct] >= MIN_CELLS and timepoints_per_ct.get(ct, 0) >= MIN_TIMEPOINTS
    ]
    print(f"Eligible cell types (n_cells>={MIN_CELLS}, n_timepoints>={MIN_TIMEPOINTS}): {len(eligible)}")
    if not eligible:
        print("No eligible cell types. Relax MIN_CELLS or MIN_TIMEPOINTS if needed.")
        return 0

    results = []
    pbar = tqdm(eligible, desc="Cell types", unit="model")
    for cell_type in pbar:
        pbar.set_postfix_str(cell_type[:40])
        model, train_l, val_l, test_r2, test_mae = train_lstm_per_cell_type(
            adata, cell_type, hvg_names, device,
            save_dir=RESULTS_MODELS,
            figures_dir=config.FIGURES_DIR,
            cell_type_col=cell_type_col,
            num_workers=num_workers,
            use_mlp=use_mlp_on_cpu,
        )
        n_cells = (adata.obs[cell_type_col].astype(str) == cell_type).sum()
        n_tp = adata.obs.loc[adata.obs[cell_type_col].astype(str) == cell_type, AGE_COL].nunique()
        results.append({
            "cell_type": cell_type,
            "n_cells": int(n_cells),
            "n_timepoints": int(n_tp),
            "test_R2": float(test_r2) if test_r2 is not None else np.nan,
            "test_MAE": float(test_mae) if test_mae is not None else np.nan,
        })

    summary_path = config.RESULTS_HARMONIZATION.parent / "lstm_training_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
