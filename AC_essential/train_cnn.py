#!/usr/bin/env python3
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── 1) load your pre-saved data ─────────────────────────────────────────────
train_df   = pd.read_pickle('train_df.pkl')
test_df    = pd.read_pickle('test_df.pkl')
df_global  = pd.read_pickle('df_global.pkl')
lat_vals   = np.load('lat_vals.npy')    # shape (H,)
lon_vals   = np.load('lon_vals.npy')    # shape (W,)
scaler     = joblib.load('sst_scaler.pkl')

H, W = len(lat_vals), len(lon_vals)

# ─── 2) helper to build sparse→dense maps on the fly ─────────────────────────
def build_maps(year, pts_df, glob_df):
    # returns inp (2×H×W) and out (1×H×W) numpy arrays
    inp = np.zeros((2, H, W), dtype=np.float32)
    out = np.full((1, H, W), np.nan, dtype=np.float32)

    # channel 0 = scaled SST at the 86 points, channel 1 = mask
    sub = pts_df[pts_df['year']==year]
    for _, r in sub.iterrows():
        i = np.searchsorted(lat_vals, r['latitude'])
        j = np.searchsorted(lon_vals, r['longitude'])
        inp[0, i, j] = r['sst_scaled']
        inp[1, i, j] = 1.0

    # fill true full map
    glob = glob_df[glob_df['year']==year]
    for _, r in glob.iterrows():
        i = np.searchsorted(lat_vals, r['latitude'])
        j = np.searchsorted(lon_vals, r['longitude'])
        out[0, i, j] = r['sst_scaled']

    return inp, out

# ─── 3) Dataset that reads per-year data on demand ───────────────────────────
class SSTYearDataset(Dataset):
    def __init__(self, years, pts_df, glob_df):
        self.years   = years
        self.pts_df  = pts_df
        self.glob_df = glob_df
    def __len__(self):
        return len(self.years)
    def __getitem__(self, idx):
        yr = self.years[idx]
        x, y = build_maps(yr, self.pts_df, self.glob_df)
        return torch.from_numpy(x), torch.from_numpy(y)

# ─── 4) simple encoder–decoder CNN ──────────────────────────────────────────
class CNNInterp(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,16, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Conv2d(16,1,3, padding=1)
    def forward(self, x):
        return self.dec(self.enc(x))

# ─── 5) masked MSE loss ignores NaNs ────────────────────────────────────────
def masked_mse(pred, true):
    mask = ~torch.isnan(true)
    diff = (pred - torch.where(mask, true, torch.zeros_like(true)))**2
    return diff[mask].mean()

# ─── 6) main training routine ───────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_years = np.sort(train_df['year'].unique())
    test_years  = np.sort(test_df ['year'].unique())

    train_ds = SSTYearDataset(train_years, train_df, df_global)
    test_ds  = SSTYearDataset(test_years,  test_df,  df_global)

    # SMALLER BATCH
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0)

    model = CNNInterp().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    best_val = np.inf
    patience = 5
    wait = 0

    for epoch in range(1, 101):
        # training
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = masked_mse(model(xb), yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vals.append(masked_mse(model(xb), yb).item())
        val_loss = float(np.mean(vals))
        print(f"Epoch {epoch:03d} — val_loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    print("Training complete. Best val_loss =", best_val)

if __name__ == '__main__':
    main()
