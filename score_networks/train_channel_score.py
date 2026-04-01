"""
Train channel score network q_{θ_H}.

Learns ∇_{H_j} ln q(H_j | σ_j) via denoising score matching (DSM):
    L(θ) = E_{H0, ε, j} [ ||σ_j * s_θ(H0 + σ_j ε, σ_j) + ε||^2 ]

H0 is drawn from the Rayleigh channel prior (i.i.d. CN(0, I)).

Usage:
    python -m score_networks.train_channel_score --config configs/rayleigh_4x1_Nu4.yaml
"""
import argparse
import os
import math
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .ncsnpp import ChannelScoreNet, ChannelScoreNet2ndOrder, get_sigmas
from channels.rayleigh import generate_rayleigh_channel, noise_schedule_exponential


def dsm_loss(
    net: nn.Module,
    H0: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Denoising score matching loss for channel prior."""
    B = H0.shape[0]
    j = torch.randint(1, len(sigmas), (B,), device=device)
    sigma_j = sigmas[j]  # (B,)

    # Noisy channel: H_j = H0 + sigma_j * eps
    eps_real = torch.randn_like(H0.real)
    eps_imag = torch.randn_like(H0.imag)

    scale = sigma_j[:, None, None]
    scale_4d = sigma_j[:, None, None, None]

    H_j_real = H0.real + scale * eps_real
    H_j_imag = H0.imag + scale * eps_imag
    H_j = torch.stack([H_j_real, H_j_imag], dim=1)  # (B, 2, NrK, NtK)

        # Normalize input
    H_j_normalized = H_j / scale_4d

    # New target
    score_target =  torch.stack([-eps_real, -eps_imag], dim=1)

    # Forward
    score_pred = net(H_j_normalized, sigma_j)

    # Loss
    loss = ((score_pred - score_target) ** 2).mean()

    # Target score: -eps / sigma_j
    # score_target = torch.stack([-eps_real, -eps_imag], dim=1) / scale_4d

    # # Predicted score
    # score_pred = net(H_j, sigma_j)

    # # Weighted MSE (sigma^2 weighting)
    # loss = (sigma_j[:, None, None, None] ** 2 * (score_pred - score_target) ** 2).mean()
    return loss


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nr = cfg.get("Nr", 4)
    Nt = cfg.get("Nt", 1)
    K = cfg.get("K", 192)
    Nu = cfg.get("Nu", 1)
    J = cfg.get("J", 30)
    sigma_1 = cfg.get("sigma_H_1", 0.01)
    sigma_J = cfg.get("sigma_H_J", 100.0)

    sigmas = noise_schedule_exponential(sigma_1, sigma_J, J, device)

    net = ChannelScoreNet(Nr=Nr, Nt=Nt, K=K).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.get("score_lr", 2e-4))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = cfg.get("score_epochs", 200)
    batch_size = cfg.get("score_batch_size", 256)

    best_loss = float("inf")
    for epoch in range(epochs):
        net.train()
        # Generate fresh channel samples each epoch
        H0 = generate_rayleigh_channel(batch_size * 10, Nu, Nr, Nt, K, device)
        H0 = H0[:, 0]  # (B, NrK, NtK)
        ds = TensorDataset(H0)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        total = 0.0
        for (h,) in loader:
            loss = dsm_loss(net, h, sigmas, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(net.state_dict(), os.path.join(ckpt_dir, "channel_score_best.pt"))

    torch.save(net.state_dict(), os.path.join(ckpt_dir, "channel_score_final.pt"))
    print("Channel score network training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
