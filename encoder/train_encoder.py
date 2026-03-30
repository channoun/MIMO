"""
Train DJSCC codec pair {f_γ, g_β} without channel (clean training).

Minimizes reconstruction loss on source images over the FFHQ-256 dataset.
After training, f_γ is used as the fixed encoder in PVD inference.

Usage:
    python -m encoder.train_encoder --config configs/rayleigh_4x1_Nu4.yaml
"""
import argparse
import os
import sys
import math
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .swin_jscc import DJSCCEncoder, DJSCCDecoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FFHQDataset(Dataset):
    """FFHQ-256 dataset. Expects PNG/JPG images in a flat directory."""

    def __init__(self, root: str, split: str = "train", val_ratio: float = 0.05):
        self.root = root
        all_files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        n_val = max(1, int(len(all_files) * val_ratio))
        if split == "train":
            self.files = all_files[n_val:]
        else:
            self.files = all_files[:n_val]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),           # [0, 1]
            transforms.Normalize([0.5]*3, [0.5]*3),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def codec_loss(D_hat: torch.Tensor, D0: torch.Tensor) -> torch.Tensor:
    """MS-SSIM + MSE composite loss."""
    mse = nn.functional.mse_loss(D_hat, D0)
    # Simple L1 + MSE for stability
    l1 = nn.functional.l1_loss(D_hat, D0)
    return mse + 0.1 * l1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Build model
    enc = DJSCCEncoder(
        embed_dim=cfg.get("embed_dim", 96),
        depths=cfg.get("depths", [2, 2, 6, 2]),
        num_heads=cfg.get("num_heads", [3, 6, 12, 24]),
        Nt=cfg.get("Nt", 1),
        K=cfg.get("K", 192),
        T=cfg.get("T", 24),
        Nu=cfg.get("Nu", 1),
        power=cfg.get("power", 1.0),
    ).to(device)
    dec = DJSCCDecoder(
        embed_dim=cfg.get("embed_dim", 96),
        depths=cfg.get("dec_depths", [2, 6, 2, 2]),
        num_heads=cfg.get("dec_num_heads", [24, 12, 6, 3]),
        Nt=cfg.get("Nt", 1),
        K=cfg.get("K", 192),
        T=cfg.get("T", 24),
    ).to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = optim.AdamW(params, lr=cfg.get("lr", 1e-4), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 100)
    )

    data_root = cfg.get("data_root", "data/ffhq256")
    train_ds = FFHQDataset(data_root, split="train")
    val_ds = FFHQDataset(data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16),
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 16),
                            shuffle=False, num_workers=4, pin_memory=True)

    ckpt_dir = cfg.get("encoder_ckpt_dir", "encoder/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg.get("epochs", 100)):
        enc.train(); dec.train()
        total_loss = 0.0
        for D0 in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            D0 = D0.to(device)
            X = enc(D0)
            D_hat = dec(X)
            loss = codec_loss(D_hat, D0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        enc.eval(); dec.eval()
        val_loss = 0.0
        with torch.no_grad():
            for D0 in val_loader:
                D0 = D0.to(device)
                D_hat = dec(enc(D0))
                val_loss += codec_loss(D_hat, D0).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:3d} | train={avg_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(ckpt_dir, "best.pt"))

    torch.save({
        "epoch": cfg.get("epochs", 100),
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
    }, os.path.join(ckpt_dir, "final.pt"))
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
