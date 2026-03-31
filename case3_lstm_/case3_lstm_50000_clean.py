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
# Case 3 baseline: 强温度前沿主导型
# ------------------------------------------------------------
# 目标：基于已跑通的 MMS 多物理场框架，构造以温度前沿为主导的第 3 个 baseline 算例。
# 改动点：
# 1) 每个物理量做输出标准化，缓解不同量纲/幅值差异。
# 2) 引入 focused sampling，增加斜层/前沿附近采样密度。
# 3) 强化 T 方程与温度前沿采样，突出 Case 3 的温度主导特征。
# 4) 保留完整输出：triplet 图、txt、statistics、loss 曲线。
# ============================================================

# -----------------------------
# 0. 可改参数
# -----------------------------
MODEL_TYPE = "lstm"          # "lstm" 或 "mlp"
CASE_NAME = "case3_lstm_50000_tempfront"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50000
LEARNING_RATE = 1e-3

# 7:3 训练/验证
N_INTERIOR_TRAIN = 5200
N_INTERIOR_VAL = 2200
N_BOUNDARY_EACH_TRAIN = 420
N_BOUNDARY_EACH_VAL = 180

NX_PLOT = 121
NY_PLOT = 121

# 网络参数
HIDDEN = 96
DEPTH = 5
LSTM_LAYERS = 2

# PDE 系数（MMS 基准系统系数）
NU = 0.02
ALPHA_T = 0.015
DIFF_PHI = 0.02
C_PHI = 0.12
C_T = 0.08
JOULE = 0.03
ETA_UV = 0.05

# 损失权重
W_CONT = 1.0
W_MX = 1.8
W_MY = 1.8
W_T = 6.5
W_PHI = 3.0
W_BC = 7.0

# focused sampling 比例
FOCUS_RATIO_U = 0.15
FOCUS_RATIO_T = 0.25
FOCUS_RATIO_PHI = 0.10

# 统计归一化参数时所用网格
NORM_GRID_N = 200

from pathlib import Path
OUTPUT_DIR = Path(__file__).resolve().parent / f"outputs_{CASE_NAME}_{MODEL_TYPE}_e{EPOCHS}_seed{SEED}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRINT_EVERY = 500
VAL_EVERY = 200
SAVE_EVERY = 2000
GRAD_CLIP = 1.0

# -----------------------------
# 1. 基础设置
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
torch.set_default_dtype(torch.float32)

if MODEL_TYPE.lower() == "lstm" and DEVICE == "cuda":
    torch.backends.cudnn.enabled = False
    print("[Info] LSTM + CUDA + 二阶自动微分：已关闭 cuDNN。")


# -----------------------------
# 2. 工具函数
# -----------------------------
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    return gradients(gradients(u, x, order=1), x, order=order - 1)


def grad_wrt_xy(field, xy):
    g = torch.autograd.grad(
        field,
        xy,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return g[:, 0:1], g[:, 1:2]


def to_numpy(x):
    return x.detach().cpu().numpy()


# -----------------------------
# 3. 采样：方形区域 [0,1] x [0,1]
# -----------------------------
def sample_uniform_interior(n):
    x = torch.rand(n, 1, device=DEVICE)
    y = torch.rand(n, 1, device=DEVICE)
    return torch.cat([x, y], dim=1)



def sample_focus_u_front(n):
    # Case 3: 速度场保持中等复杂，单主斜层为主
    y = torch.rand(n, 1, device=DEVICE)
    eps = 0.025 * torch.randn(n, 1, device=DEVICE)
    x = 0.66 - 0.35 * y + eps                 # x + 0.35 y - 0.66 = 0
    x = torch.clamp(x, 0.0, 1.0)
    return torch.cat([x, y], dim=1)

def sample_focus_T_front(n):
    # Case 3: 温度主弯曲前沿，额外加密其附近点
    y = torch.rand(n, 1, device=DEVICE)
    eps = 0.022 * torch.randn(n, 1, device=DEVICE)
    x = 0.58 - 0.28 * torch.sin(2.0 * math.pi * y) + 0.06 * torch.cos(3.0 * math.pi * y) + eps
    x = torch.clamp(x, 0.0, 1.0)
    return torch.cat([x, y], dim=1)

def sample_focus_phi_front(n):
    # phi 主前沿：y - 0.50 + 0.18*cos(2πx) = 0
    x = torch.rand(n, 1, device=DEVICE)
    eps = 0.022 * torch.randn(n, 1, device=DEVICE)
    y = 0.50 - 0.18 * torch.cos(2.0 * math.pi * x) + eps
    y = torch.clamp(y, 0.0, 1.0)
    return torch.cat([x, y], dim=1)

def sample_interior_mixed(n):
    n_focus_u = int(n * FOCUS_RATIO_U)
    n_focus_t = int(n * FOCUS_RATIO_T)
    n_focus_phi = int(n * FOCUS_RATIO_PHI)
    n_uniform = n - n_focus_u - n_focus_t - n_focus_phi
    parts = []
    if n_uniform > 0:
        parts.append(sample_uniform_interior(n_uniform))
    if n_focus_u > 0:
        parts.append(sample_focus_u_front(n_focus_u))
    if n_focus_t > 0:
        parts.append(sample_focus_T_front(n_focus_t))
    if n_focus_phi > 0:
        parts.append(sample_focus_phi_front(n_focus_phi))
    xy = torch.cat(parts, dim=0)
    perm = torch.randperm(xy.shape[0], device=DEVICE)
    return xy[perm]


def sample_boundary_left(n):
    x = torch.zeros(n, 1, device=DEVICE)
    y = torch.rand(n, 1, device=DEVICE)
    return torch.cat([x, y], dim=1)


def sample_boundary_right(n):
    x = torch.ones(n, 1, device=DEVICE)
    y = torch.rand(n, 1, device=DEVICE)
    return torch.cat([x, y], dim=1)


def sample_boundary_bottom(n):
    x = torch.rand(n, 1, device=DEVICE)
    y = torch.zeros(n, 1, device=DEVICE)
    return torch.cat([x, y], dim=1)


def sample_boundary_top(n):
    x = torch.rand(n, 1, device=DEVICE)
    y = torch.ones(n, 1, device=DEVICE)
    return torch.cat([x, y], dim=1)


def sample_all_boundaries(n_each):
    return torch.cat([
        sample_boundary_left(n_each),
        sample_boundary_right(n_each),
        sample_boundary_bottom(n_each),
        sample_boundary_top(n_each),
    ], dim=0)


# -----------------------------
# 4. Case 3 真解（强温度前沿主导型）
# -----------------------------
def psi_true(x, y):
    return (
        0.070 * torch.sin(2.0 * math.pi * x + 0.20) * torch.sin(1.15 * math.pi * y)
        + 0.020 * torch.tanh(15.0 * (x + 0.35 * y - 0.66))
        - 0.015 * torch.tanh(14.0 * (x - 0.62 * y - 0.22))
        + 0.028 * torch.exp(-52.0 * ((x - 0.78) ** 2 + (y - 0.30) ** 2))
        - 0.020 * torch.exp(-60.0 * ((x - 0.25) ** 2 + (y - 0.68) ** 2))
    )

def p_true(x, y):
    return (
        0.16 * torch.sin(1.7 * math.pi * x + 0.25) * torch.cos(1.35 * math.pi * y)
        + 0.08 * torch.exp(-48.0 * ((x - 0.24) ** 2 + (y - 0.72) ** 2))
        - 0.05 * torch.exp(-60.0 * ((x - 0.82) ** 2 + (y - 0.26) ** 2))
        + 0.025 * torch.tanh(12.0 * (x - 0.30 * y - 0.46))
    )

def T_true(x, y):
    return (
        1.0
        + 0.42 * torch.tanh(28.0 * (x - 0.58 + 0.28 * torch.sin(2.0 * math.pi * y) - 0.06 * torch.cos(3.0 * math.pi * y)))
        + 0.16 * torch.exp(-62.0 * ((x - 0.74) ** 2 + (y - 0.46) ** 2))
        - 0.09 * torch.exp(-70.0 * ((x - 0.42) ** 2 + (y - 0.18) ** 2))
        + 0.05 * torch.sin(2.6 * math.pi * x + 0.30) * torch.sin(1.7 * math.pi * y)
    )

def phi_true(x, y):
    return (
        0.46 * torch.tanh(16.0 * (y - 0.50 + 0.18 * torch.cos(2.0 * math.pi * x)))
        + 0.12 * torch.tanh(12.0 * (0.72 - x - 0.22 * y))
        + 0.07 * torch.sin(2.2 * math.pi * x) * torch.cos(1.5 * math.pi * y)
        + 0.06 * torch.exp(-55.0 * ((x - 0.78) ** 2 + (y - 0.24) ** 2))
    )

def exact_fields_from_xy(xy, need_grad=False):

    if need_grad:
        x = xy[:, 0:1].clone().detach().requires_grad_(True)
        y = xy[:, 1:2].clone().detach().requires_grad_(True)
        psi = psi_true(x, y)
        u = gradients(psi, y, order=1)
        v = -gradients(psi, x, order=1)
        p = p_true(x, y)
        T = T_true(x, y)
        phi = phi_true(x, y)
        return x, y, u, v, p, T, phi

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    with torch.enable_grad():
        xg = x.clone().detach().requires_grad_(True)
        yg = y.clone().detach().requires_grad_(True)
        psig = psi_true(xg, yg)
        u = gradients(psig, yg, order=1).detach()
        v = (-gradients(psig, xg, order=1)).detach()
    p = p_true(x, y)
    T = T_true(x, y)
    phi = phi_true(x, y)
    return x, y, u, v, p, T, phi


# -----------------------------
# 5. 输出归一化统计量
# -----------------------------
@dataclass
class FieldStats:
    mu_u: torch.Tensor
    std_u: torch.Tensor
    mu_v: torch.Tensor
    std_v: torch.Tensor
    mu_p: torch.Tensor
    std_p: torch.Tensor
    mu_T: torch.Tensor
    std_T: torch.Tensor
    mu_phi: torch.Tensor
    std_phi: torch.Tensor


def compute_field_stats(grid_n=NORM_GRID_N):
    xs = torch.linspace(0.0, 1.0, grid_n, device=DEVICE)
    ys = torch.linspace(0.0, 1.0, grid_n, device=DEVICE)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    _, _, u, v, p, T, phi = exact_fields_from_xy(xy, need_grad=False)

    def ms(a):
        mu = torch.mean(a).detach()
        std = torch.std(a).detach()
        if float(std.cpu()) < 1e-6:
            std = torch.tensor(1.0, device=DEVICE)
        return mu, std

    mu_u, std_u = ms(u)
    mu_v, std_v = ms(v)
    mu_p, std_p = ms(p)
    mu_T, std_T = ms(T)
    mu_phi, std_phi = ms(phi)
    return FieldStats(mu_u, std_u, mu_v, std_v, mu_p, std_p, mu_T, std_T, mu_phi, std_phi)


FIELD_STATS = compute_field_stats()


def split_fields_denorm(out, stats: FieldStats):
    zu = out[:, 0:1]
    zv = out[:, 1:2]
    zp = out[:, 2:3]
    zT = out[:, 3:4]
    zphi = out[:, 4:5]

    u = stats.mu_u + stats.std_u * zu
    v = stats.mu_v + stats.std_v * zv
    p = stats.mu_p + stats.std_p * zp
    T = stats.mu_T + stats.std_T * zT
    phi = stats.mu_phi + stats.std_phi * zphi
    return u, v, p, T, phi


# -----------------------------
# 6. 由真解自动生成源项（MMS）
# -----------------------------
def manufactured_sources(xy):
    x, y, u, v, p, T, phi = exact_fields_from_xy(xy, need_grad=True)

    u_x = gradients(u, x, 1)
    u_y = gradients(u, y, 1)
    v_x = gradients(v, x, 1)
    v_y = gradients(v, y, 1)
    p_x = gradients(p, x, 1)
    p_y = gradients(p, y, 1)
    T_x = gradients(T, x, 1)
    T_y = gradients(T, y, 1)
    phi_x = gradients(phi, x, 1)
    phi_y = gradients(phi, y, 1)

    u_xx = gradients(u, x, 2)
    u_yy = gradients(u, y, 2)
    v_xx = gradients(v, x, 2)
    v_yy = gradients(v, y, 2)
    T_xx = gradients(T, x, 2)
    T_yy = gradients(T, y, 2)
    phi_xx = gradients(phi, x, 2)
    phi_yy = gradients(phi, y, 2)

    src_cont = u_x + v_y
    src_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) + C_PHI * phi_x + C_T * T
    src_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) + C_PHI * phi_y
    src_T = u * T_x + v * T_y - ALPHA_T * (T_xx + T_yy) + JOULE * (phi_x ** 2 + phi_y ** 2)
    src_phi = u * phi_x + v * phi_y - DIFF_PHI * (phi_xx + phi_yy) + ETA_UV * (u + 0.5 * v)

    return {
        "cont": src_cont.detach(),
        "mx": src_mx.detach(),
        "my": src_my.detach(),
        "T": src_T.detach(),
        "phi": src_phi.detach(),
    }


# -----------------------------
# 7. 网络
# -----------------------------
class LSTMPINN(nn.Module):
    def __init__(self, hidden=64, output_dim=5, lstm_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden, output_dim)

    def forward(self, xy):
        out = self.act(self.input_layer(xy))
        out = out.unsqueeze(1)
        out, _ = self.lstm(out)
        out = out.squeeze(1)
        out = self.output_layer(out)
        return out


class MLPPINN(nn.Module):
    def __init__(self, hidden=64, depth=4, output_dim=5):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)



def build_model(model_type="lstm"):
    if model_type.lower() == "lstm":
        return LSTMPINN(hidden=HIDDEN, output_dim=5, lstm_layers=LSTM_LAYERS).to(DEVICE)
    if model_type.lower() == "mlp":
        return MLPPINN(hidden=HIDDEN, depth=DEPTH, output_dim=5).to(DEVICE)
    raise ValueError("MODEL_TYPE must be 'lstm' or 'mlp'.")


# -----------------------------
# 8. 数据集（固定采样）
# -----------------------------
@dataclass
class DatasetPack:
    interior_train: torch.Tensor
    interior_val: torch.Tensor
    boundary_train: torch.Tensor
    boundary_val: torch.Tensor



def build_dataset():
    return DatasetPack(
        interior_train=sample_interior_mixed(N_INTERIOR_TRAIN),
        interior_val=sample_interior_mixed(N_INTERIOR_VAL),
        boundary_train=sample_all_boundaries(N_BOUNDARY_EACH_TRAIN),
        boundary_val=sample_all_boundaries(N_BOUNDARY_EACH_VAL),
    )


# -----------------------------
# 9. PDE / BC 损失
# -----------------------------
def compute_pde_residuals(model, xy):
    xy_req = xy.clone().detach().requires_grad_(True)
    out = model(xy_req)
    u, v, p, T, phi = split_fields_denorm(out, FIELD_STATS)

    u_x, u_y = grad_wrt_xy(u, xy_req)
    v_x, v_y = grad_wrt_xy(v, xy_req)
    p_x, p_y = grad_wrt_xy(p, xy_req)
    T_x, T_y = grad_wrt_xy(T, xy_req)
    phi_x, phi_y = grad_wrt_xy(phi, xy_req)

    u_xx, _ = grad_wrt_xy(u_x, xy_req)
    _, u_yy = grad_wrt_xy(u_y, xy_req)
    v_xx, _ = grad_wrt_xy(v_x, xy_req)
    _, v_yy = grad_wrt_xy(v_y, xy_req)
    T_xx, _ = grad_wrt_xy(T_x, xy_req)
    _, T_yy = grad_wrt_xy(T_y, xy_req)
    phi_xx, _ = grad_wrt_xy(phi_x, xy_req)
    _, phi_yy = grad_wrt_xy(phi_y, xy_req)

    src = manufactured_sources(xy_req)

    res_cont = u_x + v_y - src["cont"]
    res_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) + C_PHI * phi_x + C_T * T - src["mx"]
    res_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) + C_PHI * phi_y - src["my"]
    res_T = u * T_x + v * T_y - ALPHA_T * (T_xx + T_yy) + JOULE * (phi_x ** 2 + phi_y ** 2) - src["T"]
    res_phi = u * phi_x + v * phi_y - DIFF_PHI * (phi_xx + phi_yy) + ETA_UV * (u + 0.5 * v) - src["phi"]

    return {
        "cont": res_cont,
        "mx": res_mx,
        "my": res_my,
        "T": res_T,
        "phi": res_phi,
    }



def loss_pde(model, xy):
    res = compute_pde_residuals(model, xy)
    l_cont = torch.mean(res["cont"] ** 2)
    l_mx = torch.mean(res["mx"] ** 2)
    l_my = torch.mean(res["my"] ** 2)
    l_T = torch.mean(res["T"] ** 2)
    l_phi = torch.mean(res["phi"] ** 2)
    total = W_CONT * l_cont + W_MX * l_mx + W_MY * l_my + W_T * l_T + W_PHI * l_phi
    return total, {
        "cont": l_cont,
        "mx": l_mx,
        "my": l_my,
        "T": l_T,
        "phi": l_phi,
    }



def loss_bc(model, xy_bc):
    pred = model(xy_bc)
    u_p, v_p, p_p, T_p, phi_p = split_fields_denorm(pred, FIELD_STATS)
    _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(xy_bc, need_grad=False)
    pred_real = torch.cat([u_p, v_p, p_p, T_p, phi_p], dim=1)
    target = torch.cat([u_t, v_t, p_t, T_t, phi_t], dim=1).to(DEVICE)
    return torch.mean((pred_real - target) ** 2)


@torch.no_grad()
def loss_bc_eval(model, xy_bc):
    pred = model(xy_bc)
    u_p, v_p, p_p, T_p, phi_p = split_fields_denorm(pred, FIELD_STATS)
    _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(xy_bc, need_grad=False)
    pred_real = torch.cat([u_p, v_p, p_p, T_p, phi_p], dim=1)
    target = torch.cat([u_t, v_t, p_t, T_t, phi_t], dim=1).to(DEVICE)
    return torch.mean((pred_real - target) ** 2)


# -----------------------------
# 10. 训练
# -----------------------------
def moving_average(values, k=5):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size < k:
        return arr
    pad = k // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(arr_pad, kernel, mode="valid")


def train_model(model, data_pack):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5000, 15000, 30000, 40000],
        gamma=0.3,
    )

    history = {
        "train_epoch": [],
        "train_total": [],
        "train_pde": [],
        "train_bc": [],
        "val_epoch": [],
        "val_total": [],
        "val_pde": [],
        "val_bc": [],
        "lr": [],
    }

    best_val = float("inf")
    best_epoch = -1
    best_state = None

    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        xy_interior = sample_interior_mixed(N_INTERIOR_TRAIN)
        xy_boundary = sample_all_boundaries(N_BOUNDARY_EACH_TRAIN)

        lpde, pde_parts = loss_pde(model, xy_interior)
        lbc = loss_bc(model, xy_boundary)
        loss_total = lpde + W_BC * lbc

        loss_total.backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        history["train_epoch"].append(epoch)
        history["train_total"].append(float(loss_total.detach().cpu()))
        history["train_pde"].append(float(lpde.detach().cpu()))
        history["train_bc"].append(float(lbc.detach().cpu()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_pde, _ = loss_pde(model, data_pack.interior_val)
            val_bc = loss_bc_eval(model, data_pack.boundary_val)
            val_total = val_pde + W_BC * val_bc

            val_total_value = float(val_total.detach().cpu())
            history["val_epoch"].append(epoch)
            history["val_total"].append(val_total_value)
            history["val_pde"].append(float(val_pde.detach().cpu()))
            history["val_bc"].append(float(val_bc.detach().cpu()))

            if val_total_value < best_val:
                best_val = val_total_value
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, OUTPUT_DIR / "best_model.pt")

        if epoch % SAVE_EVERY == 0 and epoch > 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
            }, OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt")

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            last_val = history["val_total"][-1] if history["val_total"] else float("nan")
            print(
                f"Epoch {epoch:5d} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Train Total: {history['train_total'][-1]:.6e} | "
                f"Train PDE: {history['train_pde'][-1]:.6e} | "
                f"Train BC: {history['train_bc'][-1]:.6e} | "
                f"Val Total: {last_val:.6e}"
            )
            print(
                "          "
                f"cont={pde_parts['cont'].detach().item():.3e}, "
                f"mx={pde_parts['mx'].detach().item():.3e}, "
                f"my={pde_parts['my'].detach().item():.3e}, "
                f"T={pde_parts['T'].detach().item():.3e}, "
                f"phi={pde_parts['phi'].detach().item():.3e}"
            )

    elapsed = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    np.savez(OUTPUT_DIR / "history_raw.npz",
             train_epoch=np.array(history["train_epoch"], dtype=int),
             train_total=np.array(history["train_total"], dtype=float),
             train_pde=np.array(history["train_pde"], dtype=float),
             train_bc=np.array(history["train_bc"], dtype=float),
             val_epoch=np.array(history["val_epoch"], dtype=int),
             val_total=np.array(history["val_total"], dtype=float),
             val_pde=np.array(history["val_pde"], dtype=float),
             val_bc=np.array(history["val_bc"], dtype=float),
             lr=np.array(history["lr"], dtype=float))

    history["best_val"] = best_val
    history["best_epoch"] = best_epoch
    return history, elapsed


# -----------------------------
# 11. 后处理与指标
# -----------------------------
@torch.no_grad()
def predict_on_grid(model, nx=121, ny=121):
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)
    XY = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    xy_t = torch.tensor(XY, dtype=torch.float32, device=DEVICE)

    pred = model(xy_t)
    u_p, v_p, p_p, T_p, phi_p = split_fields_denorm(pred, FIELD_STATS)
    _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(xy_t, need_grad=False)

    fields_pred = {
        "u": to_numpy(u_p).reshape(ny, nx),
        "v": to_numpy(v_p).reshape(ny, nx),
        "p": to_numpy(p_p).reshape(ny, nx),
        "T": to_numpy(T_p).reshape(ny, nx),
        "phi": to_numpy(phi_p).reshape(ny, nx),
    }
    fields_true = {
        "u": to_numpy(u_t).reshape(ny, nx),
        "v": to_numpy(v_t).reshape(ny, nx),
        "p": to_numpy(p_t).reshape(ny, nx),
        "T": to_numpy(T_t).reshape(ny, nx),
        "phi": to_numpy(phi_t).reshape(ny, nx),
    }
    return X, Y, fields_pred, fields_true



def compute_metrics(pred, true):
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    l2 = float(np.linalg.norm(err.ravel()) / (np.linalg.norm(true.ravel()) + 1e-12))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2}



def save_triplet_plot(X, Y, pred, true, field_name, save_dir):
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    items = [(pred, f"{field_name} prediction"), (true, f"{field_name} exact"), (err, f"{field_name} abs error")]
    for ax, (data, title) in zip(axes, items):
        cf = ax.contourf(X, Y, data, levels=100)
        plt.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{field_name}_triplet.png"), dpi=300)
    plt.close(fig)



def save_field_txt(X, Y, data, file_path):
    out = np.column_stack([X.reshape(-1), Y.reshape(-1), data.reshape(-1)])
    np.savetxt(file_path, out, fmt="%.8e", header="x y value")



def save_loss_plots(history, save_dir):
    train_epoch = np.array(history["train_epoch"], dtype=int)
    train_total = np.array(history["train_total"], dtype=float)
    train_pde = np.array(history["train_pde"], dtype=float)
    train_bc = np.array(history["train_bc"], dtype=float)
    val_epoch = np.array(history["val_epoch"], dtype=int)
    val_total = np.array(history["val_total"], dtype=float)
    val_smooth = moving_average(val_total, k=5)
    eps = 1e-12

    plt.figure(figsize=(9, 5))
    plt.plot(train_epoch, train_total, label="Train Total")
    plt.plot(train_epoch, train_pde, label="Train PDE")
    plt.plot(train_epoch, train_bc, label="Train BC")
    if len(val_epoch) > 0:
        plt.plot(val_epoch, val_total, alpha=0.25, label="Val Total (raw)")
        plt.plot(val_epoch, val_smooth, linewidth=2.2, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.semilogy(train_epoch, np.maximum(train_total, eps), label="Train Total")
    plt.semilogy(train_epoch, np.maximum(train_pde, eps), label="Train PDE")
    plt.semilogy(train_epoch, np.maximum(train_bc, eps), label="Train BC")
    if len(val_epoch) > 0:
        plt.semilogy(val_epoch, np.maximum(val_total, eps), alpha=0.25, label="Val Total (raw)")
        plt.semilogy(val_epoch, np.maximum(val_smooth, eps), linewidth=2.2, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Log-loss curves")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve_log.png"), dpi=300)
    plt.close()


# -----------------------------
# 12. 主程序
# -----------------------------
def main():
    print("=" * 72)
    print(f"Case 3 high-accuracy 50000 | model = {MODEL_TYPE} | device = {DEVICE}")
    print(f"Output dir = {OUTPUT_DIR}")
    print("改动：50000轮长训练 + 固定验证集 + 最佳模型恢复 + checkpoint + clean loss曲线")
    print("建议：baseline主线先跑 MLP；若后续需要对比，再用同配置跑 LSTM。")
    print("=" * 72)

    data_pack = build_dataset()
    model = build_model(MODEL_TYPE)
    history, elapsed = train_model(model, data_pack)

    X, Y, fields_pred, fields_true = predict_on_grid(model, NX_PLOT, NY_PLOT)

    metrics_all = {}
    for name in ["u", "v", "p", "T", "phi"]:
        pred = fields_pred[name]
        true = fields_true[name]
        metrics_all[name] = compute_metrics(pred, true)
        save_triplet_plot(X, Y, pred, true, name, OUTPUT_DIR)
        save_field_txt(X, Y, pred, os.path.join(OUTPUT_DIR, f"{name}_pred.txt"))
        save_field_txt(X, Y, true, os.path.join(OUTPUT_DIR, f"{name}_true.txt"))
        save_field_txt(X, Y, np.abs(pred - true), os.path.join(OUTPUT_DIR, f"{name}_abs_err.txt"))

    save_loss_plots(history, OUTPUT_DIR)

    stats_path = os.path.join(OUTPUT_DIR, "statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"CASE_NAME = {CASE_NAME}\n")
        f.write(f"MODEL_TYPE = {MODEL_TYPE}\n")
        f.write(f"DEVICE = {DEVICE}\n")
        f.write(f"EPOCHS = {EPOCHS}\n")
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"N_INTERIOR_TRAIN = {N_INTERIOR_TRAIN}\n")
        f.write(f"N_INTERIOR_VAL = {N_INTERIOR_VAL}\n")
        f.write(f"N_BOUNDARY_EACH_TRAIN = {N_BOUNDARY_EACH_TRAIN}\n")
        f.write(f"N_BOUNDARY_EACH_VAL = {N_BOUNDARY_EACH_VAL}\n")
        f.write(f"W_CONT = {W_CONT}\n")
        f.write(f"W_MX = {W_MX}\n")
        f.write(f"W_MY = {W_MY}\n")
        f.write(f"W_T = {W_T}\n")
        f.write(f"W_PHI = {W_PHI}\n")
        f.write(f"W_BC = {W_BC}\n")
        f.write(f"FOCUS_RATIO_U = {FOCUS_RATIO_U}\n")
        f.write(f"FOCUS_RATIO_T = {FOCUS_RATIO_T}\n")
        f.write(f"FOCUS_RATIO_PHI = {FOCUS_RATIO_PHI}\n")
        f.write(f"ELAPSED_SECONDS = {elapsed:.6f}\n")
        f.write(f"BEST_VAL_TOTAL = {history['best_val']:.8e}\n")
        f.write(f"BEST_EPOCH = {history['best_epoch']}\n\n")
        f.write("[field_stats]\n")
        f.write(f"mu_u = {float(FIELD_STATS.mu_u.cpu()):.8e}\n")
        f.write(f"std_u = {float(FIELD_STATS.std_u.cpu()):.8e}\n")
        f.write(f"mu_v = {float(FIELD_STATS.mu_v.cpu()):.8e}\n")
        f.write(f"std_v = {float(FIELD_STATS.std_v.cpu()):.8e}\n")
        f.write(f"mu_p = {float(FIELD_STATS.mu_p.cpu()):.8e}\n")
        f.write(f"std_p = {float(FIELD_STATS.std_p.cpu()):.8e}\n")
        f.write(f"mu_T = {float(FIELD_STATS.mu_T.cpu()):.8e}\n")
        f.write(f"std_T = {float(FIELD_STATS.std_T.cpu()):.8e}\n")
        f.write(f"mu_phi = {float(FIELD_STATS.mu_phi.cpu()):.8e}\n")
        f.write(f"std_phi = {float(FIELD_STATS.std_phi.cpu()):.8e}\n\n")
        for field_name, metric_dict in metrics_all.items():
            f.write(f"[{field_name}]\n")
            for k, v in metric_dict.items():
                f.write(f"{k} = {v:.8e}\n")
            f.write("\n")

    print("\n训练完成，结果已保存到：", OUTPUT_DIR)
    print("下一步建议：先封存本次输出目录；若需要对比算法，再用同配置跑 LSTM。")


if __name__ == "__main__":
    main()
