import os
import time
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# Case 8 | LSTM-PINN | 连续训练 50000 轮主文件
# ============================================================

MODEL_TYPE = "lstm"
CASE_NAME = "case8_s_shock_interaction"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50000
LEARNING_RATE = 1e-3

# 与 case6 MLP 保持同级别采样
N_INTERIOR_TRAIN = 3200
N_INTERIOR_VAL = 1400
N_BOUNDARY_EACH_TRAIN = 560
N_BOUNDARY_EACH_VAL = 240

NX_PLOT = 201
VAL_EVERY = 200
PRINT_EVERY = 500
SAVE_EVERY = 10000

# PDE 系数
NU = 0.019
ALPHA_T = 0.020
DIFF_PHI = 0.020
ETA_UV = 0.055
C_T = 0.080
PHI_COUP = 0.110
JOULE = 0.038

# 损失权重
W_CONT = 1.0
W_MX = 1.0
W_MY = 1.0
W_T = 1.6
W_PHI = 2.3
W_BC = 8.0

# LSTM 结构
EMBED_DIM = 24
HIDDEN_DIM = 88
NUM_LAYERS = 2

OUTPUT_DIR = f"outputs_{CASE_NAME}_{MODEL_TYPE}_e{EPOCHS}_seed{SEED}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
torch.set_default_dtype(torch.float32)
# LSTM + 高阶自动微分更稳定
torch.backends.cudnn.enabled = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def grad_wrt_xy(u, xy):
    g = torch.autograd.grad(
        u, xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return g[:, 0:1], g[:, 1:2]


def to_numpy(x):
    return x.detach().cpu().numpy()


# ============================================================
# Case 8 真解（与 case6 MLP 一致）
# ============================================================

def psi_true(x, y):
    s_center = 0.50 + 0.18 * torch.sin(1.10 * math.pi * x + 0.10) - 0.09 * torch.sin(2.10 * math.pi * x + 0.25)
    lower_branch = 0.22 + 0.28 * x + 0.04 * torch.cos(1.25 * math.pi * x + 0.18)
    upper_branch = 0.82 - 0.30 * x + 0.05 * torch.sin(1.05 * math.pi * x + 0.32)

    return (
        0.048 * torch.tanh(15.0 * (y - s_center))
        + 0.020 * torch.tanh(11.0 * (y - lower_branch))
        - 0.022 * torch.tanh(10.5 * (y - upper_branch))
        + 0.020 * torch.sin(1.26 * math.pi * x + 0.10) * torch.sin(1.00 * math.pi * y + 0.08)
        + 0.024 * torch.exp(-27.0 * ((x - 0.24) ** 2 + (y - 0.22) ** 2))
        - 0.018 * torch.exp(-25.0 * ((x - 0.74) ** 2 + (y - 0.78) ** 2))
        + 0.016 * torch.exp(-18.0 * ((x - 0.52) ** 2 + (y - 0.50) ** 2))
    )


def p_true_func(x, y):
    return (
        0.112 * torch.cos(0.98 * math.pi * x + 0.16) * torch.sin(0.92 * math.pi * y + 0.12)
        + 0.110 * torch.exp(-30.0 * ((x - 0.78) ** 2 + (y - 0.28) ** 2))
        - 0.088 * torch.exp(-28.0 * ((x - 0.30) ** 2 + (y - 0.72) ** 2))
        + 0.034 * torch.exp(-16.0 * ((x - 0.56) ** 2 + (y - 0.46) ** 2))
    )


def T_true_func(x, y):
    front_T = 0.52 + 0.12 * torch.sin(1.05 * math.pi * x + 0.06) - 0.16 * torch.exp(-12.0 * (x - 0.65) ** 2)

    return (
        0.83
        + 0.10 * x - 0.09 * y
        + 0.26 * torch.tanh(10.0 * (y - front_T))
        + 0.058 * torch.exp(-24.0 * ((x - 0.34) ** 2 + (y - 0.70) ** 2))
        + 0.044 * torch.exp(-22.0 * ((x - 0.82) ** 2 + (y - 0.24) ** 2))
        + 0.030 * torch.sin(0.90 * math.pi * x + 0.07) * torch.sin(0.86 * math.pi * y + 0.09)
    )


def phi_true_func(x, y):
    front_phi = 0.50 - 0.22 * torch.exp(-10.0 * (x - 0.48) ** 2) + 0.10 * torch.sin(1.72 * math.pi * x + 0.14)

    return (
        0.45 * torch.tanh(11.3 * (y - front_phi))
        + 0.040 * torch.exp(-30.0 * ((x - 0.64) ** 2 + (y - 0.30) ** 2))
        - 0.034 * torch.exp(-24.0 * ((x - 0.20) ** 2 + (y - 0.62) ** 2))
        + 0.034 * torch.sin(1.32 * math.pi * x + 0.03) * torch.sin(1.06 * math.pi * y + 0.05)
        + 0.018 * x + 0.018 * y
    )


def exact_fields_from_xy(xy, need_grad=True):
    with torch.enable_grad():
        xy_local = xy.clone().detach().requires_grad_(True)
        x = xy_local[:, 0:1]
        y = xy_local[:, 1:2]

        psi = psi_true(x, y)
        psix, psiy = grad_wrt_xy(psi, xy_local)

        u = psiy
        v = -psix
        p = p_true_func(x, y)
        T = T_true_func(x, y)
        phi = phi_true_func(x, y)

    if need_grad:
        return xy_local, x, y, u, v, p, T, phi

    return (
        xy_local.detach(),
        x.detach(),
        y.detach(),
        u.detach(),
        v.detach(),
        p.detach(),
        T.detach(),
        phi.detach()
    )


# ============================================================
# LSTM-PINN
# ============================================================

class LSTMPINN(nn.Module):
    def __init__(self, out_dim=5, embed_dim=24, hidden_dim=88, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, xy):
        seq = xy.unsqueeze(-1)
        emb = torch.tanh(self.embed(seq))
        _, (h_n, _) = self.lstm(emb)
        h_last = h_n[-1]
        feat = torch.cat([h_last, xy], dim=1)
        return self.head(feat)


# ============================================================
# 采样
# ============================================================

def sample_interior(n):
    pts = np.random.rand(n, 2).astype(np.float32)
    return torch.tensor(pts, dtype=torch.float32, device=DEVICE)


def sample_boundary_side(n, side):
    s = np.random.rand(n, 1).astype(np.float32)
    if side == "left":
        pts = np.concatenate([np.zeros_like(s), s], axis=1)
    elif side == "right":
        pts = np.concatenate([np.ones_like(s), s], axis=1)
    elif side == "bottom":
        pts = np.concatenate([s, np.zeros_like(s)], axis=1)
    elif side == "top":
        pts = np.concatenate([s, np.ones_like(s)], axis=1)
    else:
        raise ValueError("Unknown side")
    return torch.tensor(pts, dtype=torch.float32, device=DEVICE)


@dataclass
class DataPack:
    interior_train: torch.Tensor
    interior_val: torch.Tensor
    boundary_train: torch.Tensor
    boundary_val: torch.Tensor


def build_dataset():
    interior_train = sample_interior(N_INTERIOR_TRAIN)
    interior_val = sample_interior(N_INTERIOR_VAL)

    boundary_train = torch.cat([
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "left"),
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "right"),
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "bottom"),
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "top"),
    ], dim=0)

    boundary_val = torch.cat([
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "left"),
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "right"),
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "bottom"),
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "top"),
    ], dim=0)

    return DataPack(interior_train, interior_val, boundary_train, boundary_val)


# ============================================================
# Manufactured source terms
# ============================================================

def compute_sources_from_exact(xy):
    xy, x, y, u, v, p, T, phi = exact_fields_from_xy(xy, need_grad=True)

    ux, uy = grad_wrt_xy(u, xy)
    vx, vy = grad_wrt_xy(v, xy)
    px, py = grad_wrt_xy(p, xy)
    Tx, Ty = grad_wrt_xy(T, xy)
    phix, phiy = grad_wrt_xy(phi, xy)

    uxx, _ = grad_wrt_xy(ux, xy)
    _, uyy = grad_wrt_xy(uy, xy)
    vxx, _ = grad_wrt_xy(vx, xy)
    _, vyy = grad_wrt_xy(vy, xy)
    Txx, _ = grad_wrt_xy(Tx, xy)
    _, Tyy = grad_wrt_xy(Ty, xy)
    phixx, _ = grad_wrt_xy(phix, xy)
    _, phiyy = grad_wrt_xy(phiy, xy)

    s_cont = ux + vy
    s_mx = u * ux + v * uy + px - NU * (uxx + uyy) - ETA_UV * phix + C_T * Tx
    s_my = u * vx + v * vy + py - NU * (vxx + vyy) - ETA_UV * phiy + C_T * Ty
    s_T = u * Tx + v * Ty - ALPHA_T * (Txx + Tyy) + JOULE * (phix ** 2 + phiy ** 2)
    s_phi = -DIFF_PHI * (phixx + phiyy) + PHI_COUP * (u * phix + v * phiy) - 0.05 * T * phi

    return s_cont.detach(), s_mx.detach(), s_my.detach(), s_T.detach(), s_phi.detach()


# ============================================================
# 损失
# ============================================================

def loss_pde(model, xy):
    xy = xy.clone().detach().requires_grad_(True)

    pred = model(xy)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]
    T = pred[:, 3:4]
    phi = pred[:, 4:5]

    ux, uy = grad_wrt_xy(u, xy)
    vx, vy = grad_wrt_xy(v, xy)
    px, py = grad_wrt_xy(p, xy)
    Tx, Ty = grad_wrt_xy(T, xy)
    phix, phiy = grad_wrt_xy(phi, xy)

    uxx, _ = grad_wrt_xy(ux, xy)
    _, uyy = grad_wrt_xy(uy, xy)
    vxx, _ = grad_wrt_xy(vx, xy)
    _, vyy = grad_wrt_xy(vy, xy)
    Txx, _ = grad_wrt_xy(Tx, xy)
    _, Tyy = grad_wrt_xy(Ty, xy)
    phixx, _ = grad_wrt_xy(phix, xy)
    _, phiyy = grad_wrt_xy(phiy, xy)

    s_cont, s_mx, s_my, s_T, s_phi = compute_sources_from_exact(xy)

    r_cont = ux + vy - s_cont
    r_mx = u * ux + v * uy + px - NU * (uxx + uyy) - ETA_UV * phix + C_T * Tx - s_mx
    r_my = u * vx + v * vy + py - NU * (vxx + vyy) - ETA_UV * phiy + C_T * Ty - s_my
    r_T = u * Tx + v * Ty - ALPHA_T * (Txx + Tyy) + JOULE * (phix ** 2 + phiy ** 2) - s_T
    r_phi = -DIFF_PHI * (phixx + phiyy) + PHI_COUP * (u * phix + v * phiy) - 0.05 * T * phi - s_phi

    l_cont = torch.mean(r_cont ** 2)
    l_mx = torch.mean(r_mx ** 2)
    l_my = torch.mean(r_my ** 2)
    l_T = torch.mean(r_T ** 2)
    l_phi = torch.mean(r_phi ** 2)

    total = (
        W_CONT * l_cont +
        W_MX * l_mx +
        W_MY * l_my +
        W_T * l_T +
        W_PHI * l_phi
    )
    return total


def loss_bc(model, xy_bc):
    pred = model(xy_bc)
    _, _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(xy_bc, need_grad=False)

    loss_u = torch.mean((pred[:, 0:1] - u_t) ** 2)
    loss_v = torch.mean((pred[:, 1:2] - v_t) ** 2)
    loss_p = torch.mean((pred[:, 2:3] - p_t) ** 2)
    loss_T = torch.mean((pred[:, 3:4] - T_t) ** 2)
    loss_phi = torch.mean((pred[:, 4:5] - phi_t) ** 2)
    return loss_u + loss_v + loss_p + loss_T + loss_phi


# ============================================================
# 训练
# ============================================================

def save_checkpoint(epoch, model, optimizer, scheduler, history):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }
    torch.save(ckpt, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pt"))


def train_model(model, data_pack):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5000, 15000, 30000, 42000],
        gamma=0.40
    )

    history = {
        "epochs": [],
        "train_total": [],
        "train_pde": [],
        "train_bc": [],
        "val_total_raw": [],
        "val_total": [],
        "val_epochs": [],
        "best_val": float("inf"),
        "best_epoch": -1,
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        train_pde = loss_pde(model, data_pack.interior_train)
        train_bc = loss_bc(model, data_pack.boundary_train)
        train_total = train_pde + W_BC * train_bc

        train_total.backward()
        optimizer.step()
        scheduler.step()

        history["epochs"].append(epoch)
        history["train_total"].append(float(train_total.detach().cpu()))
        history["train_pde"].append(float(train_pde.detach().cpu()))
        history["train_bc"].append(float(train_bc.detach().cpu()))

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            model.train()
            val_bc = loss_bc(model, data_pack.boundary_val)
            val_pde = loss_pde(model, data_pack.interior_val)
            val_total = val_pde + W_BC * val_bc

            raw_val = float(val_total.detach().cpu())
            history["val_total_raw"].append(raw_val)
            history["val_epochs"].append(epoch)

            smoothed = raw_val if len(history["val_total_raw"]) < 7 else float(np.mean(history["val_total_raw"][-7:]))
            history["val_total"].append(smoothed)

            if raw_val < history["best_val"]:
                history["best_val"] = raw_val
                history["best_epoch"] = epoch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:5d} | LR: {lr_now:.2e} | "
                f"Train Total: {float(train_total.detach().cpu()):.6e} | "
                f"Train PDE: {float(train_pde.detach().cpu()):.6e} | "
                f"Train BC: {float(train_bc.detach().cpu()):.6e}"
            )
            if history["val_total_raw"]:
                msg += f" | Val Total: {history['val_total_raw'][-1]:.6e}"
            print(msg)

        if epoch > 0 and epoch % SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, history)

    elapsed = time.time() - start_time
    return history, elapsed


# ============================================================
# 输出与评估
# ============================================================

def save_loss_plots(history, save_dir):
    epochs = np.array(history["epochs"], dtype=np.int32)
    val_epochs = np.array(history["val_epochs"], dtype=np.int32)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_total"], label="Train Total")
    plt.plot(epochs, history["train_pde"], label="Train PDE")
    plt.plot(epochs, history["train_bc"], label="Train BC")
    plt.plot(val_epochs, history["val_total_raw"], alpha=0.35, label="Val Total (raw)")
    plt.plot(val_epochs, history["val_total"], linewidth=3.0, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.semilogy(epochs, history["train_total"], label="Train Total")
    plt.semilogy(epochs, history["train_pde"], label="Train PDE")
    plt.semilogy(epochs, history["train_bc"], label="Train BC")
    plt.semilogy(val_epochs, history["val_total_raw"], alpha=0.35, label="Val Total (raw)")
    plt.semilogy(val_epochs, history["val_total"], linewidth=3.0, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Log-loss curves")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve_log.png"), dpi=300)
    plt.close()

    np.savez(
        os.path.join(save_dir, "history_raw.npz"),
        epochs=epochs,
        train_total=np.array(history["train_total"]),
        train_pde=np.array(history["train_pde"]),
        train_bc=np.array(history["train_bc"]),
        val_epochs=val_epochs,
        val_total_raw=np.array(history["val_total_raw"]),
        val_total=np.array(history["val_total"]),
    )


def build_plot_grid(nx=NX_PLOT):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return X, Y, torch.tensor(pts, dtype=torch.float32, device=DEVICE)


def metric_dict(pred, true):
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    l2 = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12))
    max_abs = float(np.max(np.abs(err)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2, "MAX_ABS": max_abs}


def save_field_txt(path, X, Y, Z):
    arr = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    np.savetxt(path, arr, fmt="%.8e", header="x y value", comments="")


def save_triplet(field_name, X, Y, pred, true, save_dir):
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    items = [
        (pred, f"{field_name} prediction"),
        (true, f"{field_name} exact"),
        (err, f"{field_name} abs error"),
    ]
    for ax, (Z, title) in zip(axes, items):
        im = ax.contourf(X, Y, Z, levels=120)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{field_name}_triplet.png"), dpi=300)
    plt.close()

    save_field_txt(os.path.join(save_dir, f"{field_name}_pred.txt"), X, Y, pred)
    save_field_txt(os.path.join(save_dir, f"{field_name}_true.txt"), X, Y, true)
    save_field_txt(os.path.join(save_dir, f"{field_name}_abs_err.txt"), X, Y, err)


def post_process_and_save(model, elapsed, history):
    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"[Info] 已恢复 best_model.pt (epoch={history['best_epoch']}).")

    model.eval()

    X, Y, pts = build_plot_grid()
    with torch.no_grad():
        pred = model(pts).cpu().numpy()

    _, _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(pts, need_grad=False)

    u_true = to_numpy(u_t).reshape(X.shape)
    v_true = to_numpy(v_t).reshape(X.shape)
    p_true = to_numpy(p_t).reshape(X.shape)
    T_true = to_numpy(T_t).reshape(X.shape)
    phi_true = to_numpy(phi_t).reshape(X.shape)

    u_pred = pred[:, 0].reshape(X.shape)
    v_pred = pred[:, 1].reshape(X.shape)
    p_pred = pred[:, 2].reshape(X.shape)
    T_pred = pred[:, 3].reshape(X.shape)
    phi_pred = pred[:, 4].reshape(X.shape)

    metrics_all = {
        "u": metric_dict(u_pred, u_true),
        "v": metric_dict(v_pred, v_true),
        "p": metric_dict(p_pred, p_true),
        "T": metric_dict(T_pred, T_true),
        "phi": metric_dict(phi_pred, phi_true),
    }

    save_triplet("u", X, Y, u_pred, u_true, OUTPUT_DIR)
    save_triplet("v", X, Y, v_pred, v_true, OUTPUT_DIR)
    save_triplet("p", X, Y, p_pred, p_true, OUTPUT_DIR)
    save_triplet("T", X, Y, T_pred, T_true, OUTPUT_DIR)
    save_triplet("phi", X, Y, phi_pred, phi_true, OUTPUT_DIR)

    save_loss_plots(history, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "statistics.txt"), "w", encoding="utf-8") as f:
        f.write("[meta]\n")
        f.write(f"CASE_NAME = {CASE_NAME}\n")
        f.write(f"MODEL_TYPE = {MODEL_TYPE}\n")
        f.write(f"SEED = {SEED}\n")
        f.write(f"DEVICE = {DEVICE}\n")
        f.write(f"EPOCHS = {EPOCHS}\n")
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"EMBED_DIM = {EMBED_DIM}\n")
        f.write(f"HIDDEN_DIM = {HIDDEN_DIM}\n")
        f.write(f"NUM_LAYERS = {NUM_LAYERS}\n")
        f.write(f"ELAPSED_SECONDS = {elapsed:.6f}\n")
        f.write(f"BEST_VAL_TOTAL = {history['best_val']:.8e}\n")
        f.write(f"BEST_EPOCH = {history['best_epoch']}\n\n")

        f.write("[error_analysis]\n")
        rmse_rank = sorted([(k, v["RMSE"]) for k, v in metrics_all.items()], key=lambda x: x[1], reverse=True)
        f.write("RMSE_rank = " + " > ".join([f"{k}:{v:.4e}" for k, v in rmse_rank]) + "\n\n")

        for name, md in metrics_all.items():
            f.write(f"[{name}]\n")
            for k, v in md.items():
                f.write(f"{k} = {v:.8e}\n")
            f.write("\n")

    print(f"\n训练完成，结果已保存到：{OUTPUT_DIR}")


def main():
    print("=" * 72)
    print("Case 8 | LSTM-PINN | 连续训练 50000 轮")
    print(f"DEVICE = {DEVICE}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"EPOCHS = {EPOCHS}")
    print("=" * 72)

    data_pack = build_dataset()
    model = LSTMPINN(
        out_dim=5,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    history, elapsed = train_model(model, data_pack)
    post_process_and_save(model, elapsed, history)


if __name__ == "__main__":
    main()
