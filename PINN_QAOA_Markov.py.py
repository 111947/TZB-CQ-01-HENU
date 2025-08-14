# strict_pinn_lindblad_qaoa_3qubit.py
# 严格 PINN 实现：
# - 一个网络 state_net(t) 严格作为 rho(t) 的近似（通过 Cholesky 参数化保证正定与 trace=1）
# - 一个网络 control_net(t) 严格输出控制量 gamma(t), beta(t)
# - 在 collocation 点上使用自动微分计算 time-derivative 并构造 Lindblad 残差作为 PINN 损失
# - 包含训练循环、collocation 采样与损失权重

import os
import math
import random
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------- config ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
np.random.seed(0); random.seed(0); torch.manual_seed(0)
os.makedirs('results', exist_ok=True)

# ---------- 问题定义（保持与原始 3 机组/3 qubit 一致） ----------
Unit_cost = {"1": 0.1, "2": 0.5, "3": 0.4}
Unit_power = {"1": 0.1, "2": 0.5, "3": 0.4}
P_demand = 0.5
items = list(Unit_cost.keys())
n = len(items)
N = n  # qubits
print(f"Using N={N} qubits (no slack bits)")

# ---------- 构造 Ising 能量用于目标态（保留原先逻辑） ----------
from sympy import symbols, Symbol
x_vars = symbols('x0:' + str(n))
lam = Symbol('lambda')
fx = sum(Unit_cost[items[i]] * x_vars[i] for i in range(n))
p_expr = sum(Unit_power[items[i]] * x_vars[i] for i in range(n)) - P_demand
penalty = lam * p_expr**2
QUBO = (fx + penalty).subs(lam, 15)
new_vars = {x_vars[i]: (1 - Symbol(f'z{i}'))/2 for i in range(n)}
ising_h = QUBO.subs(new_vars).expand()

z_symbols = [Symbol(f'z{i}') for i in range(N)]
all_cfg = list(product([0,1], repeat=N))
energies = []
for cfg in all_cfg:
    subs_map = {z_symbols[i]: 1 - 2*cfg[i] for i in range(N)}
    energies.append(float(ising_h.subs(subs_map)))
energies = np.array(energies, dtype=np.float64)
min_idx = int(np.argmin(energies))
print("Ising energies:", energies)
print("Min energy index:", min_idx, "bitstring:", np.binary_repr(min_idx, width=N))
target_index = min_idx

# Hamiltonians as torch complex matrices
dim = 2 ** N
Evec = torch.tensor(energies, dtype=torch.complex64, device=device)
H_cost = torch.diag(Evec).to(device)

# H_mix = sum X_i
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
I2 = torch.eye(2, dtype=torch.complex64, device=device)
H_mix = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
for i in range(N):
    ops = [I2]*N
    ops[i] = X
    M = ops[0]
    for op in ops[1:]:
        M = torch.kron(M, op)
    H_mix += M

# Lindblad operators
def get_lindblad_ops(n_qubits, device):
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
    I = torch.eye(2, dtype=torch.complex64, device=device)
    sm = (X - 1j*Y) / 2.0
    sp = (X + 1j*Y) / 2.0
    Ls = []
    for i in range(n_qubits):
        ops_m = [I]*n_qubits
        ops_p = [I]*n_qubits
        ops_m[i] = sm; ops_p[i] = sp
        Lm = ops_m[0]; Lp = ops_p[0]
        for op in ops_m[1:]: Lm = torch.kron(Lm, op)
        for op in ops_p[1:]: Lp = torch.kron(Lp, op)
        Ls += [Lm, Lp]
    return Ls

L_list = get_lindblad_ops(N, device)
gamma_noise = 0.05

# ---------------- complex<->real helper ----------------
def mat_to_vec_complex(rho):
    return rho.reshape(-1)

def vec_complex_to_mat(vec, D):
    return vec.reshape(D, D)

# lindblad RHS (complex)
def lindblad_rhs_complex(rho, H, Ls, gamma):
    comm = -1j * (H @ rho - rho @ H)
    diss = torch.zeros_like(rho)
    for L in Ls:
        Ld = L.conj().T
        diss = diss + gamma * (L @ rho @ Ld - 0.5*(Ld @ L @ rho + rho @ Ld @ L))
    return comm + diss

# ---------------- PINN 网络定义 ----------------
# ControlNet: 输入 t -> 输出 gamma(t), beta(t)（实数）
class ControlNet(nn.Module):
    def __init__(self, hidden=128, layers=3, gamma_range=(0.0, 2.0), beta_range=(0.0, math.pi)):
        super().__init__()
        dims = [1] + [hidden]*layers + [2]
        ops = []
        for i in range(len(dims)-1):
            ops.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                ops.append(nn.Tanh())
        self.net = nn.Sequential(*ops)
        self.gamma_min, self.gamma_max = gamma_range
        self.beta_min, self.beta_max = beta_range
        self.sigmoid = nn.Sigmoid()
    def forward(self, t):
        # t: (batch,1)
        raw = self.net(t)  # (batch,2)
        out = self.sigmoid(raw)
        gamma = self.gamma_min + out[:,0] * (self.gamma_max - self.gamma_min)
        beta  = self.beta_min  + out[:,1] * (self.beta_max - self.beta_min)
        return torch.stack([gamma, beta], dim=1)  # (batch,2)

# StateNet: 输入 t -> 输出 rho(t) 通过 Cholesky 参数化
class StateNet(nn.Module):
    def __init__(self, hidden=256, layers=3, dim=dim):
        super().__init__()
        self.dim = dim
        out_dim = 2 * dim * dim  # real and imag parts of lower-triangular full matrix (we'll use full reshape but enforce PD via A @ A^H)
        dims = [1] + [hidden]*layers + [out_dim]
        ops = []
        for i in range(len(dims)-1):
            ops.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                ops.append(nn.Tanh())
        self.net = nn.Sequential(*ops)
    def forward(self, t):
        # t: (batch,1)
        out = self.net(t)  # (batch, 2*D*D)
        batch = out.shape[0]
        re = out[:, :self.dim*self.dim].reshape(batch, self.dim, self.dim)
        im = out[:, self.dim*self.dim:].reshape(batch, self.dim, self.dim)
        # 构造复矩阵 A (不硬性只取下三角，这里更简单直接用全矩阵再做 A @ A^H 保证 PSD)
        A = re + 1j * im
        # 为了数值稳定：对角加上小的正数
        for b in range(batch):
            A[b].view(-1)[0] = A[b].view(-1)[0]  # no-op to keep device
        rho = torch.matmul(A, A.conj().transpose(-2,-1))
        tr = rho.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).view(batch,1,1)
        rho = rho / tr
        return rho  # shape (batch, D, D), complex64

# ---------------- derivative helper via autograd ----------------
# 由于 PyTorch 对 complex 的 jacobian 支持有限，我们对 real/imag 分开并用 jacobian

def flatten_rho_realimag(state_net, t):
    # t: (batch,1)
    rho = state_net(t)[0]  # (D,D) complex
    re = rho.real.reshape(-1)
    im = rho.imag.reshape(-1)
    return torch.cat([re, im])  # 2*D*D vector

# compute d/dt rho(t) at scalar t0 (t0 tensor requires_grad True)
def drho_dt_at_t(state_net, t_tensor):
    # t_tensor: tensor shape (1,1) with requires_grad=True
    # use autograd.functional.jacobian
    fun = lambda tt: flatten_rho_realimag(state_net, tt)
    J = torch.autograd.functional.jacobian(fun, t_tensor, create_graph=True)
    # J shape: (out_dim, 1, 1) -> take [:,0,0]
    dflat_dt = J[:,0,0]
    D2 = dim * dim
    dre_dt = dflat_dt[:D2].reshape(dim, dim)
    dim_dt = dflat_dt[D2:].reshape(dim, dim)
    drho = dre_dt + 1j * dim_dt
    return drho

# ---------------- initial/target ----------------
rho0 = torch.zeros((dim, dim), dtype=torch.complex64, device=device); rho0[0,0] = 1.0 + 0j
psi_target = torch.zeros((dim,), dtype=torch.complex64, device=device); psi_target[target_index] = 1.0 + 0j
rho_target = torch.outer(psi_target, psi_target.conj())

# ---------------- training hyperparameters ----------------
T = 1.0
collocation_N = 80  # collocation 点数量
epochs = 500
lr = 1e-3

# loss 权重
w_res = 1.0       # residual (PINN) 权重
w_ic = 100.0      # 初值项权重
w_term = 10.0     # 终端 fidelity 权重
w_ctrl = 1.0      # control 平滑/幅度正则（可选）

# nets
control_net = ControlNet(hidden=128, layers=3).to(device)
state_net = StateNet(hidden=256, layers=3, dim=dim).to(device)

params = list(control_net.parameters()) + list(state_net.parameters())
opt = optim.Adam(params, lr=lr)

# collocation 时间采样函数
def sample_collocation_points(Nc, T):
    ts = torch.rand(Nc, 1, device=device) * T
    return ts

# control smoothness loss (在 collocation 点上评估 gamma/beta 的二阶差分或梯度)
def control_smoothness_loss(control_net, ts):
    ts = ts.clone().detach().requires_grad_(False)
    vals = control_net(ts)  # (Nc,2)
    # 用一阶差分做近似平滑惩罚
    diffs = vals[1:] - vals[:-1]
    return torch.mean(diffs**2)

# 训练循环
print('Start training strict PINN...')
for ep in range(1, epochs+1):
    opt.zero_grad()
    # 采样 collocation
    ts_coll = sample_collocation_points(collocation_N, T)
    loss_res = 0.0
    for i in range(ts_coll.shape[0]):
        t_i = ts_coll[i].view(1,1)  # shape (1,1)
        t_i = t_i.to(device).requires_grad_(True)
        # 预测 rho and d/dt rho
        rho_pred = state_net(t_i)[0]  # (D,D) complex
        drho_dt = drho_dt_at_t(state_net, t_i)  # (D,D) complex
        # 控制量
        ctrl = control_net(t_i)[0]  # (2,)
        gamma_t = ctrl[0]; beta_t = ctrl[1]
        Ht = gamma_t * H_cost + beta_t * H_mix
        rhs = lindblad_rhs_complex(rho_pred, Ht, L_list, gamma_noise)
        res = drho_dt - rhs
        loss_res = loss_res + torch.mean(torch.abs(res)**2)
    loss_res = loss_res / ts_coll.shape[0]

    # initial condition loss
    rho_0_pred = state_net(torch.tensor([[0.0]], device=device))[0]
    loss_ic = torch.mean(torch.abs(rho_0_pred - rho0)**2)

    # terminal fidelity loss
    rho_T_pred = state_net(torch.tensor([[T]], device=device))[0]
    fid = torch.real(rho_T_pred[target_index, target_index])
    loss_term = (1.0 - fid)

    # control smoothness
    ts_for_ctrl = torch.linspace(0, T, steps=collocation_N, device=device).unsqueeze(-1)
    loss_ctrl_s = control_smoothness_loss(control_net, ts_for_ctrl)

    loss = w_res * loss_res + w_ic * loss_ic + w_term * loss_term + w_ctrl * loss_ctrl_s
    loss.backward()
    opt.step()

    if ep % 50 == 0 or ep == 1:
        with torch.no_grad():
            # 打印若干信息
            print(f"Epoch {ep}/{epochs}: total_loss={loss.item():.6e}, res={loss_res.item():.3e}, ic={loss_ic.item():.3e}, term_fid={fid.item():.6f}, ctrl_s={loss_ctrl_s.item():.3e}")
            # 打印 control 在若干层中点
            p_layers = 6
            for k in range(min(p_layers,6)):
                t_mid = torch.tensor([[(k + 0.5) / p_layers * T]], dtype=torch.float32, device=device)
                out = control_net(t_mid).cpu().numpy().flatten()
                print(f"  layer {k+1}: gamma={out[0]:.4f}, beta={out[1]:.4f}")

# 保存模型
torch.save({'state_net': state_net.state_dict(), 'control_net': control_net.state_dict()}, 'results/pinn_strict_models.pt')
print('Finished training. Models saved to results/pinn_strict_models.pt')

# ------------------ qaoa_circuit (保持与原来代码门序一致) ------------------
def qaoa_circuit(ising_Hamiltonian, p=1):
    betas = ParameterVector(r"$\beta$", p)
    gammas = ParameterVector(r"$\gamma$", p)
    ising_dict = ising_Hamiltonian.expand().as_coefficients_dict()
    num_qubits = len(ising_Hamiltonian.free_symbols)
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for layer in range(p):
        # 单体项 Rz
        for i in range(num_qubits):
            if Symbol(f"z{i}") in ising_dict:
                wi = float(ising_dict[Symbol(f"z{i}")])
                qc.rz(2 * gammas[layer] * wi, i)
        qc.barrier()
        # 交互项 Rzz
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if Symbol(f"z{i}") * Symbol(f"z{j}") in ising_dict:
                    wij = float(ising_dict[Symbol(f"z{i}") * Symbol(f"z{j}")])
                    qc.rzz(2 * gammas[layer] * wij, i, j)
        qc.barrier()
        # 混合项 Rx
        for i in range(num_qubits):
            qc.rx(-2 * betas[layer], i)
    qc = qc.reverse_bits()
    qc.measure_all()
    return qc, betas, gammas

# ------------------ 画未绑定参数的电路 ------------------
qc_param, betas_sym, gammas_sym = qaoa_circuit(ising_h, p=p_layers)
fig = qc_param.draw(output='mpl')
fig.savefig('results/qaoa_param_unbound.png', bbox_inches='tight', dpi=200)
print("Saved unbound circuit figure: results/qaoa_param_unbound.png")

# ------------------ 从训练好的 PINN 离散化得到 p 层参数并绑定 ------------------
betas_vals = []
gammas_vals = []
with torch.no_grad():
    for k in range(p_layers):
        t_mid = torch.tensor([[(k + 0.5)/p_layers * T]], dtype=torch.float32, device=device)
        out = control_net(t_mid).cpu().numpy().flatten()  # [gamma, beta]
        gammas_vals.append(float(out[0]))
        betas_vals.append(float(out[1]))

print("Discrete gammas:", np.array(gammas_vals))
print("Discrete betas:", np.array(betas_vals))

bind_map = {}
for i in range(p_layers):
    bind_map[betas_sym[i]] = betas_vals[i]
    bind_map[gammas_sym[i]] = gammas_vals[i]

qc_bound = qc_param.assign_parameters(bind_map)
fig2 = qc_bound.draw(output='mpl')
fig2.savefig('results/qaoa_param_bound.png', bbox_inches='tight', dpi=200)
print("Saved bound circuit figure: results/qaoa_param_bound.png")

# ------------------ 仿真和统计 ------------------
backend = AerSimulator()
job = backend.run(transpile(qc_bound, backend), shots=5000)
res = job.result()
counts = res.get_counts()
print("Counts (top 10):", sorted(counts.items(), key=lambda kv: -kv[1])[:10])

# 解析最高频bit串及对应机组开关状态
top = max(counts, key=counts.get)
top_rev = top[::-1]  # bit顺序翻转
units_on = [int(b) for b in top_rev[:n]]
cost_val = sum(Unit_cost[items[i]] * units_on[i] for i in range(n))
print("Most frequent bitstring:", top, "-> units_on (reversed first n):", units_on, "cost:", cost_val)

# 筛选满足功率需求的bit串
powers = [Unit_power[k] for k in items]
valid = {bs: v for bs, v in counts.items() if sum(int(b) * p for b, p in zip(bs[::-1][:n], powers)) >= P_demand}
print("Valid bitstrings (power >= P_demand):", valid)

# 统计出现次数超过阈值的bit串
threshold = 500
filtered_counts = {bs: cnt for bs, cnt in counts.items() if cnt > threshold}

print(f"Bitstrings appearing more than {threshold} times:")
for bs, cnt in sorted(filtered_counts.items(), key=lambda kv: -kv[1]):
    total_power = sum(int(bit) * p for bit, p in zip(bs[::-1][:n], powers))
    print(f"{bs}: {cnt} times, Total power: {total_power:.3f}")

# 进一步筛选功率 >= 0.5 的bit串
power_threshold = 0.5
power_filtered = {bs: cnt for bs, cnt in filtered_counts.items()
                  if sum(int(bit) * p for bit, p in zip(bs[::-1][:n], powers)) >= power_threshold}

print(f"\nBitstrings with total power >= {power_threshold}:")
for bs, cnt in sorted(power_filtered.items(), key=lambda kv: -kv[1]):
    total_power = sum(int(bit) * p for bit, p in zip(bs[::-1][:n], powers))
    print(f"{bs}: {cnt} times, Total power: {total_power:.3f}")

valid_bitstrings = list(power_filtered.keys())
print(f"\nValid bitstrings array: {valid_bitstrings}")

# 截断去重示例（简单展示）
truncated_unique = list(dict.fromkeys(valid_bitstrings))
print(f"\nTruncated and unique bitstrings array: {truncated_unique}")
