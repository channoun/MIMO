"""
Main evaluation script for Blind-MIMOSC.

Runs PVD and all baselines over Monte Carlo trials and reports metrics.

Usage:
    python eval.py --config configs/rayleigh_4x1_Nu4.yaml --snr 10
    python eval.py --config configs/rayleigh_4x1_Nu4.yaml --all-snr
    python eval.py --config configs/stable_noise.yaml --snr 10 --stable
"""
import argparse
import os
import math
from unittest import loader
from pathlib import Path
from torchvision import transforms
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

from encoder.swin_jscc import DJSCCEncoder, DJSCCDecoder
from score_networks.ncsnpp import NCSNpp, ChannelScoreNet, ChannelScoreNet2ndOrder, ImageScoreNet2ndOrder
from pvd.pvd import PVDSolver
from channels.rayleigh import generate_rayleigh_channel, apply_channel, noise_schedule_exponential
from channels.cdl_c import load_cdlc_channel
from baselines.djscc_mimo import DJSCCMIMOBaseline
from baselines.dps_mimo import DPSMIMOBaseline
from metrics.ms_ssim import ms_ssim
from metrics.nmse import nmse_db


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_encoder(cfg: dict, device: torch.device) -> tuple:
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

    ckpt_dir = cfg.get("encoder_ckpt_dir", "encoder/checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["encoder"])
        dec.load_state_dict(ckpt["decoder"])
        print(f"Loaded encoder/decoder from {ckpt_path}")
    else:
        print(f"WARNING: No encoder checkpoint found at {ckpt_path}. Using random weights.")
    return enc.eval(), dec.eval()


def load_score_nets(cfg: dict, device: torch.device) -> tuple:
    Nr, Nt, K = cfg.get("Nr", 4), cfg.get("Nt", 1), cfg.get("K", 192)
    ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")

    S_theta_H = ChannelScoreNet(Nr=Nr, Nt=Nt, K=K).to(device)
    S_theta_D = NCSNpp(
        in_channels=3,
        base_channels=cfg.get("score_base_channels", 128),
        ch_mults=tuple(cfg.get("score_ch_mults", [1, 2, 2, 2])),
        num_res_blocks=cfg.get("score_num_res_blocks", 2),
        attn_resolutions=tuple(cfg.get("score_attn_resolutions", [16])),
        dropout=cfg.get("score_dropout", 0.1),
    ).to(device)

    ch_ckpt = os.path.join(ckpt_dir, "channel_score_best.pt")
    img_ckpt = os.path.join(ckpt_dir, "image_score_best.pt")

    if os.path.exists(ch_ckpt):
        S_theta_H.load_state_dict(torch.load(ch_ckpt, map_location=device, weights_only=True))
        print(f"Loaded channel score from {ch_ckpt}")
    else:
        print(f"WARNING: No channel score checkpoint at {ch_ckpt}.")

    if os.path.exists(img_ckpt):
        S_theta_D.load_state_dict(torch.load(img_ckpt, map_location=device, weights_only=True))
        print(f"Loaded image score from {img_ckpt}")
    else:
        print(f"WARNING: No image score checkpoint at {img_ckpt}.")

    # Second-order networks
    s_theta_H = ChannelScoreNet2ndOrder(Nr=Nr, Nt=Nt, K=K).to(device)
    s_theta_D = ImageScoreNet2ndOrder().to(device)

    ch_2nd = os.path.join(ckpt_dir, "channel_score2nd_best.pt")
    img_2nd = os.path.join(ckpt_dir, "image_score2nd_best.pt")
    if os.path.exists(ch_2nd):
        s_theta_H.load_state_dict(torch.load(ch_2nd, map_location=device, weights_only=True))
    if os.path.exists(img_2nd):
        s_theta_D.load_state_dict(torch.load(img_2nd, map_location=device, weights_only=True))

    if not cfg.get("use_second_order", True):
        s_theta_H = None
        s_theta_D = None

    return S_theta_H.eval(), S_theta_D.eval(), s_theta_H, s_theta_D


# ---------------------------------------------------------------------------
# Channel generation
# ---------------------------------------------------------------------------

def get_channel(cfg: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    channel = cfg.get("channel", "rayleigh")
    Nr, Nt, K, Nu = cfg["Nr"], cfg["Nt"], cfg["K"], cfg.get("Nu", 1)
    if channel == "rayleigh":
        return generate_rayleigh_channel(batch_size, Nu, Nr, Nt, K, device)
    elif channel == "cdl_c":
        path = cfg.get("cdl_c_path", "data/cdl_c_channels.npy")
        return load_cdlc_channel(path, batch_size, Nr, Nt, K, device)
    else:
        raise ValueError(f"Unknown channel type: {channel}")


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------

def evaluate_pvd(
    pvd: PVDSolver,
    D0: torch.Tensor,
    H0: torch.Tensor,
    Y: torch.Tensor,
    sigma_n: float,
) -> Dict[str, float]:
    H_hat, D_hat = pvd.solve(Y, verbose=False)

    # Denormalize images from [-1,1] to [0,1]
    D0_01 = (D0.clamp(-1, 1) + 1) / 2
    D_hat_01 = (D_hat.clamp(-1, 1) + 1) / 2

    ms_ssim_val = ms_ssim(D0_01, D_hat_01).mean().item()
    nmse_val = nmse_db(H_hat, H0[:, 0]).item()

    return {
        "ms_ssim": ms_ssim_val,
        "nmse_db": nmse_val,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_at_snr(cfg: dict, snr_db: float, args, device: torch.device) -> Dict[str, List]:
    Nr, Nt, K, T, Nu = cfg["Nr"], cfg["Nt"], cfg["K"], cfg["T"], cfg.get("Nu", 1)
    n_trials = cfg.get("n_trials", 300)
    batch_size = args.batch_size

    # Load models
    enc, dec = load_encoder(cfg, device)
    S_theta_H, S_theta_D, s_theta_H, s_theta_D = load_score_nets(cfg, device)

    # Noise std
    snr_linear = 10 ** (snr_db / 10.0)
    sigma_n = math.sqrt(1.0 / snr_linear)  # unit signal power

    # Build PVD
    pvd = PVDSolver(
        f_gamma=enc, S_theta_H=S_theta_H, S_theta_D=S_theta_D,
        s_theta_H=s_theta_H, s_theta_D=s_theta_D,
        sigma_n=sigma_n, Nr=Nr, Nt=Nt, K=K, T=T, Nu=Nu,
        J=cfg.get("J", 30), J_in=cfg.get("J_in", 20),
        zeta_H=cfg.get("zeta_H", 1.0), zeta_D=cfg.get("zeta_D", 1.0),
        device=device,
        use_second_order=cfg.get("use_second_order", True),
    )

    # Baselines
    djscc_perfect = DJSCCMIMOBaseline(enc, dec, Nr, Nt, K, T, Nu, perfect_csi=True)
    djscc_pilot = DJSCCMIMOBaseline(enc, dec, Nr, Nt, K, T, Nu, perfect_csi=False)

    results = {
        "pvd": {"ms_ssim": [], "nmse_db": []},
        "djscc_perfect": {"ms_ssim": [], "nmse_db": []},
        "djscc_pilot": {"ms_ssim": [], "nmse_db": []},
    }

    n_done = 0
    pbar = tqdm(total=n_trials, desc=f"SNR={snr_db}dB")

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
    ])

    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class FlatFolderDataset(Dataset):
        def __init__(self, root, transform=None):
            self.paths = list(Path(root).glob("*.png"))
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0
        

    dataset = FlatFolderDataset("data/mnist256/all", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    data_iter = iter(loader)

    while n_done < n_trials:
        bs = min(batch_size, n_trials - n_done)

        # Generate channel and random images (random if no dataset)
        H0 = get_channel(cfg, bs, device)
        # D0 = torch.rand(bs, 3, 256, 256, device=device) * 2 - 1  # placeholder

        try:
            D0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            D0, _ = next(data_iter)

        D0 = D0.to(device)

        # Apply channel for PVD
        X = enc(D0)
        Y, _ = apply_channel(H0, X, snr_db)

        # PVD
        try:
            r_pvd = evaluate_pvd(pvd, D0, H0, Y, sigma_n)
            for k in r_pvd:
                results["pvd"][k].append(r_pvd[k])
        except Exception as e:
            print(f"PVD failed: {e}")

        # DJSCC-perfect
        try:
            D_hat_p, H_hat_p = djscc_perfect.run(D0, H0, snr_db, sigma_n)
            D0_01 = (D0 + 1) / 2
            D_hat_p_01 = (D_hat_p.clamp(-1,1) + 1) / 2
            results["djscc_perfect"]["ms_ssim"].append(ms_ssim(D0_01, D_hat_p_01).mean().item())
            results["djscc_perfect"]["nmse_db"].append(0.0)  # perfect CSI
        except Exception as e:
            print(f"DJSCC-perfect failed: {e}")

        # DJSCC-pilot
        try:
            D_hat_pi, H_hat_pi = djscc_pilot.run(D0, H0, snr_db, sigma_n)
            D0_01 = (D0 + 1) / 2
            D_hat_pi_01 = (D_hat_pi.clamp(-1,1) + 1) / 2
            results["djscc_pilot"]["ms_ssim"].append(ms_ssim(D0_01, D_hat_pi_01).mean().item())
            results["djscc_pilot"]["nmse_db"].append(nmse_db(H_hat_pi, H0[:, 0]).item())
        except Exception as e:
            print(f"DJSCC-pilot failed: {e}")

        n_done += bs
        pbar.update(bs)

    pbar.close()
    return results


def summarize(results: Dict) -> Dict:
    summary = {}
    for method, metrics in results.items():
        summary[method] = {}
        for metric, vals in metrics.items():
            if vals:
                arr = np.array(vals)
                summary[method][metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--all-snr", action="store_true")
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    if args.all_snr:
        snr_list = cfg.get("snr_db_range", [-5, 0, 5, 10, 15, 20])
    else:
        snr_list = [args.snr]

    all_results = {}
    for snr_db in snr_list:
        print(f"\n{'='*50}")
        print(f"Evaluating at SNR = {snr_db} dB")
        print(f"{'='*50}")
        results = evaluate_at_snr(cfg, snr_db, args, device)
        summary = summarize(results)
        all_results[str(snr_db)] = summary

        # Print summary table
        print(f"\nSNR = {snr_db} dB Results:")
        print(f"{'Method':<20} {'MS-SSIM':>10} {'NMSE(dB)':>10}")
        print("-" * 42)
        for method, metrics in summary.items():
            ms = metrics.get("ms_ssim", {})
            nm = metrics.get("nmse_db", {})
            ms_str = f"{ms.get('mean', 0):.4f}±{ms.get('std', 0):.4f}" if ms else "N/A"
            nm_str = f"{nm.get('mean', 0):.2f}±{nm.get('std', 0):.2f}" if nm else "N/A"
            print(f"{method:<20} {ms_str:>10} {nm_str:>10}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
