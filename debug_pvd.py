"""
Standalone PVD debug runner.

Loads models from a config, generates a single batch, and runs one PVD trial
with full per-step diagnostics printed to stdout.

Usage:
    python debug_pvd.py --config configs/rayleigh_4x1_Nu4.yaml --snr 10 --J 5 --J_in 3
    python debug_pvd.py --config configs/rayleigh_4x1_Nu4.yaml --snr 10          # full J/J_in from config
    python debug_pvd.py --config configs/rayleigh_4x1_Nu4.yaml --snr 10 --no-second-order

Flags:
    --J          Override number of outer diffusion steps (default: from config)
    --J_in       Override number of inner Langevin steps  (default: from config)
    --snr        SNR in dB
    --no-second-order   Disable second-order trace correction (simplifies eff_var)
    --no-checkpoint     Disable gradient checkpointing on encoder
"""
import argparse
import math
import yaml
import torch

from encoder.swin_jscc import DJSCCEncoder
from score_networks.ncsnpp import NCSNpp, ChannelScoreNet, ChannelScoreNet2ndOrder, ImageScoreNet2ndOrder
from pvd.pvd import PVDSolver
from channels.rayleigh import generate_rayleigh_channel, apply_channel
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--snr",     type=float, default=10.0)
    parser.add_argument("--J",       type=int,   default=None)
    parser.add_argument("--J_in",    type=int,   default=None)
    parser.add_argument("--no-second-order",         action="store_true")
    parser.add_argument("--no-checkpoint",           action="store_true")
    parser.add_argument("--analytical-channel-prior", action="store_true",
                        help="Use exact Rayleigh score instead of trained channel score net")
    parser.add_argument("--device",  type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    Nr  = cfg["Nr"]
    Nt  = cfg["Nt"]
    K   = cfg["K"]
    T   = cfg["T"]
    Nu  = cfg.get("Nu", 1)
    J   = args.J   if args.J   is not None else cfg.get("J",    30)
    J_in= args.J_in if args.J_in is not None else cfg.get("J_in", 20)

    print(f"Device : {device}")
    print(f"Config : {args.config}")
    print(f"Nr={Nr}  Nt={Nt}  K={K}  T={T}  Nu={Nu}  J={J}  J_in={J_in}  SNR={args.snr}dB")

    # ------------------------------------------------------------------ #
    # Load encoder
    # ------------------------------------------------------------------ #
    enc = DJSCCEncoder(
        embed_dim   = cfg.get("embed_dim", 96),
        depths      = cfg.get("depths",    [2, 2, 6, 2]),
        num_heads   = cfg.get("num_heads", [3, 6, 12, 24]),
        Nt=Nt, K=K, T=T, Nu=Nu,
        power       = cfg.get("power", 1.0),
    ).to(device).eval()

    ckpt_dir  = cfg.get("encoder_ckpt_dir", "encoder/checkpoints")
    enc_ckpt  = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(enc_ckpt):
        ckpt = torch.load(enc_ckpt, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["encoder"])
        print(f"Loaded encoder from {enc_ckpt}")
    else:
        print(f"WARNING: encoder checkpoint not found at {enc_ckpt} — using random weights")

    # ------------------------------------------------------------------ #
    # Load score networks
    # ------------------------------------------------------------------ #
    score_ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")

    S_theta_H = ChannelScoreNet(Nr=Nr, Nt=Nt, K=K).to(device).eval()
    ch_ckpt = os.path.join(score_ckpt_dir, "channel_score_best.pt")
    if os.path.exists(ch_ckpt):
        S_theta_H.load_state_dict(torch.load(ch_ckpt, map_location=device, weights_only=True))
        print(f"Loaded channel score from {ch_ckpt}")
    else:
        print(f"WARNING: channel score checkpoint not found — using random weights")

    S_theta_D = NCSNpp(
        in_channels        = 3,
        base_channels      = cfg.get("score_base_channels", 128),
        ch_mults           = tuple(cfg.get("score_ch_mults",  [1, 2, 2, 2])),
        num_res_blocks     = cfg.get("score_num_res_blocks",  2),
        attn_resolutions   = tuple(cfg.get("score_attn_resolutions", [16])),
        dropout            = cfg.get("score_dropout", 0.1),
    ).to(device).eval()
    img_ckpt = os.path.join(score_ckpt_dir, "image_score_best.pt")
    if os.path.exists(img_ckpt):
        S_theta_D.load_state_dict(torch.load(img_ckpt, map_location=device, weights_only=True))
        print(f"Loaded image score from {img_ckpt}")
    else:
        print(f"WARNING: image score checkpoint not found — using random weights")

    # Second-order networks
    use_second_order = not args.no_second_order and cfg.get("use_second_order", True)
    s_theta_H, s_theta_D = None, None
    if use_second_order:
        s_theta_H = ChannelScoreNet2ndOrder(Nr=Nr, Nt=Nt, K=K).to(device).eval()
        s_theta_D = ImageScoreNet2ndOrder().to(device).eval()
        ch2_ckpt  = os.path.join(score_ckpt_dir, "channel_score2nd_best.pt")
        img2_ckpt = os.path.join(score_ckpt_dir, "image_score2nd_best.pt")
        if os.path.exists(ch2_ckpt):
            s_theta_H.load_state_dict(torch.load(ch2_ckpt, map_location=device, weights_only=True))
        if os.path.exists(img2_ckpt):
            s_theta_D.load_state_dict(torch.load(img2_ckpt, map_location=device, weights_only=True))
        print(f"Second-order networks: {'loaded' if os.path.exists(ch2_ckpt) else 'random weights'}")
    else:
        print("Second-order correction: DISABLED")

    # ------------------------------------------------------------------ #
    # Generate one batch
    # ------------------------------------------------------------------ #
    snr_linear = 10 ** (args.snr / 10.0)
    sigma_n    = math.sqrt(1.0 / snr_linear)

    H0 = generate_rayleigh_channel(1, Nu, Nr, Nt, K, device)          # (1, Nu, NrK, NtK)
    D0 = torch.randn(1, 3, 256, 256, device=device)                    # placeholder image

    with torch.no_grad():
        X = enc(D0)                                                     # (1, Nu, NtK, T)
    Y, sigma_n_actual = apply_channel(H0, X, args.snr)                 # (1, NrK, T)

    print(f"\nGenerated batch:  H0 max={H0.abs().max():.4f}  "
          f"X max={X.abs().max():.4f}  Y max={Y.abs().max():.4f}  "
          f"sigma_n={sigma_n_actual:.4e}")

    # ------------------------------------------------------------------ #
    # Build PVD solver
    # ------------------------------------------------------------------ #
    use_analytical = args.analytical_channel_prior
    if use_analytical:
        print("Channel prior: ANALYTICAL (exact Rayleigh score, bypasses trained net)")
    else:
        print("Channel prior: TRAINED score network")

    pvd = PVDSolver(
        f_gamma                    = enc,
        S_theta_H                  = S_theta_H,
        S_theta_D                  = S_theta_D,
        s_theta_H                  = s_theta_H,
        s_theta_D                  = s_theta_D,
        sigma_n                    = sigma_n_actual,
        Nr=Nr, Nt=Nt, K=K, T=T, Nu=Nu,
        J                          = J,
        J_in                       = J_in,
        zeta_H                     = cfg.get("zeta_H", 1.0),
        zeta_D                     = cfg.get("zeta_D", 1.0),
        device                     = device,
        use_second_order           = use_second_order,
        use_checkpoint             = not args.no_checkpoint,
        use_analytical_channel_prior = use_analytical,
    )

    # ------------------------------------------------------------------ #
    # Run one PVD trial with full debug output
    # ------------------------------------------------------------------ #
    print("\nStarting PVD debug run...")
    H_hat, D_hat = pvd.solve(Y, verbose=False, debug=True)

    print("\n" + "="*70)
    print("FINAL ESTIMATES")
    print("="*70)
    print(f"H_hat  mean={H_hat.abs().mean():.4e}  max={H_hat.abs().max():.4e}  "
          f"NaN={torch.isnan(H_hat).any().item()}")
    print(f"D_hat  mean={D_hat.abs().mean():.4e}  max={D_hat.abs().max():.4e}  "
          f"NaN={torch.isnan(D_hat).any().item()}")


if __name__ == "__main__":
    main()
