# pinn_lindblad_qaoa_3qubit_nonmarkov_fixed.py
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import Symbol, symbols
from itertools import product
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# ---------------- config ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results', exist_ok=True)
torch.set_default_dtype(torch.float32)

# ---------- 用户问题相关（原始 3 机组数据） ----------
Unit_cost = {"1": 0.1, "2": 0.5, "3": 0.4}
Unit_power = {"1": 0.1, "2": 0.5, "3": 0.4}
P_demand = 0.5

items = list(Unit_cost.keys())
n = len(items)            # =3
N = n   # 不加 slack bits
print(f"Using N={N} qubits (no slack bits)")

# ---------- QUBO -> Ising 映射（保持你原来的设计） ----------
x_vars = symbols('x0:' + str(n))
lam = Symbol('lambda')
fx = sum(Unit_cost[items[i]] * x_vars[i] for i in range(n))
p_expr = sum(Unit_power[items[i]] * x_vars[i] for i in range(n)) - P_demand
penalty = lam * p_expr**2
QUBO = (fx + penalty).subs(lam, 15)

new_vars = {x_vars[i]: (1 - Symbol(f'z{i}'))/2 for i in range(n)}
ising_h = QUBO.subs(new_vars).expand()
z_symbols = [Symbol(f'z{i}') for i in range(N)]

# 计算对角能量向量（长度 2^N）
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

# H_cost 对角矩阵（torch complex）
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

# ------------- Lindblad operators -------------
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

# ---------- Non-Markov memory 配置 ----------
memory_depth = 3   # <-- 过去 n 个状态（你可以改）
tau_mem = 0.1      # 记忆链时间常数（与 T 同量纲）
mem_alpha = 0.8    # 历史权重衰减（越小越强调近期）

# ---------- complex <-> real & augmented helpers ----------
def mat_to_vec_complex(rho):
    return rho.reshape(-1)

def vec_complex_to_mat(vec, D):
    return vec.reshape(D, D)

def complex_vec_to_real(vec):
    return torch.cat([vec.real.to(torch.float32), vec.imag.to(torch.float32)], dim=0)

def real_to_complex_vec(r):
    D2 = r.shape[0] // 2
    re = r[:D2].to(torch.float32)
    im = r[D2:].to(torch.float32)
    return re.type(torch.complex64) + 1j * im.type(torch.complex64)

def split_chunks_complex(vec, chunk_len):
    L = vec.shape[0]
    assert L % chunk_len == 0
    k = L // chunk_len
    return [vec[i*chunk_len:(i+1)*chunk_len] for i in range(k)]

def real_to_augmented_complex(r, D2, memory_depth):
    cvec = real_to_complex_vec(r)
    chunks = split_chunks_complex(cvec, D2)
    return chunks

def augmented_complex_to_real(chunks):
    cvec = torch.cat(chunks, dim=0)
    return complex_vec_to_real(cvec)

# ------------- 非马尔可夫 RHS（增强态） -------------
class LindbladRHS_NonMarkov(nn.Module):
    def __init__(self, ctrl_net, H_cost, H_mix, Ls, gamma_noise, device,
                 memory_depth=3, tau_mem=0.1, mem_alpha=0.8):
        super().__init__()
        self.ctrl = ctrl_net
        self.H_cost = H_cost
        self.H_mix = H_mix
        self.Ls = Ls
        self.gamma_noise = gamma_noise
        self.device = device
        self.memory_depth = memory_depth
        self.tau_mem = tau_mem
        self.mem_alpha = mem_alpha

    def forward(self, t, r):
        t_in = t.reshape(1,1).to(self.device)
        cb = self.ctrl(t_in)
        gamma_t = cb[0,0]; beta_t = cb[0,1]
        Ht = gamma_t * self.H_cost + beta_t * self.H_mix

        dim = self.H_cost.shape[0]
        D2 = dim * dim

        chunks = real_to_augmented_complex(r, D2, self.memory_depth)
        rho_vec = chunks[0]
        rho = vec_complex_to_mat(rho_vec, dim)
        hist_vecs = [vec_complex_to_mat(ch, dim) for ch in chunks[1:]]

        weights = [1.0] + [ (self.mem_alpha ** (j+1)) for j in range(self.memory_depth) ]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]

        rho_mem = weights[0] * rho
        for j, hj in enumerate(hist_vecs):
            rho_mem = rho_mem + weights[j+1] * hj

        comm = -1j * (Ht @ rho - rho @ Ht)
        diss = torch.zeros_like(rho)
        for L in self.Ls:
            Ld = L.conj().T
            diss = diss + self.gamma_noise * (L @ rho_mem @ Ld - 0.5*(Ld @ L @ rho + rho @ Ld @ L))

        drho = comm + diss

        dh_list = []
        if self.memory_depth > 0:
            h_prev = rho
            for j in range(self.memory_depth):
                hj = hist_vecs[j]
                dh = (h_prev - hj) / self.tau_mem
                dh_list.append(dh)
                h_prev = hj

        out_chunks = [mat_to_vec_complex(drho)]
        for dh in dh_list:
            out_chunks.append(mat_to_vec_complex(dh))
        dr_real = augmented_complex_to_real(out_chunks).to(self.device)
        return dr_real

# ------------------ QAOAPINN (生成 gamma, beta) ------------------
class QAOAPINN(nn.Module):
    def __init__(self, hidden=64, layers=2, gamma_range=(0.0, 2.0), beta_range=(0.0, math.pi)):
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
        raw_out = self.net(t)
        out = self.sigmoid(raw_out)
        gamma = self.gamma_min + out[:,0] * (self.gamma_max - self.gamma_min)
        beta = self.beta_min + out[:,1] * (self.beta_max - self.beta_min)
        return torch.stack([gamma, beta], dim=1)

# ------------------ initial & target states （augmented） ------------------
dim = 2 ** N
rho0 = torch.zeros((dim, dim), dtype=torch.complex64, device=device); rho0[0,0] = 1.0 + 0j
v0 = mat_to_vec_complex(rho0)
D2 = v0.shape[0]

aug_chunks = [v0] + [v0.clone() for _ in range(memory_depth)]
r0_aug_real = augmented_complex_to_real(aug_chunks).to(device)

psi_target = torch.zeros((dim,), dtype=torch.complex64, device=device); psi_target[target_index] = 1.0 + 0j
rho_target = torch.outer(psi_target, psi_target.conj())

# ------------------ training setup ------------------
T = 1.0
p_layers = 6
control_net = QAOAPINN(hidden=128, layers=2).to(device)
rhs = LindbladRHS_NonMarkov(control_net, H_cost, H_mix, L_list, gamma_noise, device,
                            memory_depth=memory_depth, tau_mem=tau_mem, mem_alpha=mem_alpha).to(device)

def integrate_rho(r0_real, rhs_module, T, method='rk4'):
    t_pts = torch.tensor([0.0, T], dtype=torch.float32, device=device)
    sol = odeint(rhs_module, r0_real, t_pts, method=method)
    return sol[-1]

opt = optim.Adam(control_net.parameters(), lr=1e-3)
epochs = 300
lambda_reg = 1e-2
print_every = 50

def control_smoothness_loss(ctrl_net, T, samples=20):
    ts = torch.linspace(0, T, steps=samples, device=device).unsqueeze(-1)
    cb = ctrl_net(ts)
    return torch.mean(cb[:,0]**2) + torch.mean(cb[:,1]**2)

# ------------------ training loop ------------------
for ep in range(1, epochs+1):
    opt.zero_grad()
    rT_aug = integrate_rho(r0_aug_real, rhs, T, method='rk4')
    chunks_T = real_to_augmented_complex(rT_aug, D2, memory_depth)
    rhoT = vec_complex_to_mat(chunks_T[0], dim)
    fid = torch.real(rhoT[target_index, target_index])
    loss = (1.0 - fid) + lambda_reg * control_smoothness_loss(control_net, T, samples=20)
    loss.backward()
    opt.step()

    if ep % print_every == 0 or ep == 1:
        print(f"Epoch {ep}/{epochs}: Loss={loss.item():.6f}")
        print("Params (gamma,beta):")
        with torch.no_grad():
            for k in range(p_layers):
                t_mid = torch.tensor([[(k + 0.5) / p_layers * T]], dtype=torch.float32, device=device)
                out = control_net(t_mid).cpu().numpy().flatten()
                print(f"  layer {k+1}: gamma={out[0]:.4f}, beta={out[1]:.4f}")

# ------------------ qaoa_circuit（修正：参数名不使用 $ 符号） ------------------
def qaoa_circuit(ising_Hamiltonian, p=1):
    betas = ParameterVector(r"$\beta$", p)
    gammas = ParameterVector(r"$\gamma$", p)
    ising_dict = ising_Hamiltonian.expand().as_coefficients_dict()
    num_qubits = len(ising_Hamiltonian.free_symbols)
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for layer in range(p):
        for i in range(num_qubits):
            if Symbol(f"z{i}") in ising_dict:
                wi = float(ising_dict[Symbol(f"z{i}")])
                qc.rz(2 * gammas[layer] * wi, i)
        qc.barrier()
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if Symbol(f"z{i}") * Symbol(f"z{j}") in ising_dict:
                    wij = float(ising_dict[Symbol(f"z{i}") * Symbol(f"z{j}")])
                    qc.rzz(2 * gammas[layer] * wij, i, j)
        qc.barrier()
        for i in range(num_qubits):
            qc.rx(-2 * betas[layer], i)
    qc = qc.reverse_bits()
    qc.measure_all()
    return qc, betas, gammas

# ------------------ draw parameterized circuit ------------------
qc_param, betas_sym, gammas_sym = qaoa_circuit(ising_h, p=p_layers)
fig = qc_param.draw(output='mpl')
fig.savefig('results/qaoa_param_unbound.png', bbox_inches='tight', dpi=200)
print("Saved unbound circuit figure: results/qaoa_param_unbound.png")

# ------------------ 从训练好的 PINN 离散化得到 p 层参数并绑定 ------------------
betas_vals = []; gammas_vals = []
with torch.no_grad():
    for k in range(p_layers):
        t_mid = torch.tensor([[(k + 0.5)/p_layers * T]], dtype=torch.float32, device=device)
        out = control_net(t_mid).cpu().numpy().flatten()  # [gamma, beta]
        gammas_vals.append(float(out[0])); betas_vals.append(float(out[1]))

print("Discrete gammas:", np.array(gammas_vals))
print("Discrete betas:", np.array(betas_vals))

bind_map = {}
for i in range(p_layers):
    bind_map[betas_sym[i]] = float(betas_vals[i])
    bind_map[gammas_sym[i]] = float(gammas_vals[i])

qc_bound = qc_param.assign_parameters(bind_map)
fig2 = qc_bound.draw(output='mpl'); fig2.savefig('results/qaoa_param_bound.png', bbox_inches='tight', dpi=200)
print("Saved bound circuit figure: results/qaoa_param_bound.png")

# ------------------ simulate and show counts ------------------
backend = AerSimulator()
job = backend.run(transpile(qc_bound, backend), shots=5000)
res = job.result()
counts = res.get_counts()
print("Counts (top 10):", sorted(counts.items(), key=lambda kv:-kv[1])[:10])

top = max(counts, key=counts.get)
top_rev = top[::-1]
units_on = [int(b) for b in top_rev[:n]]
cost_val = sum(Unit_cost[items[i]] * units_on[i] for i in range(n))
print("Most frequent bitstring:", top, "-> units_on (reversed first n):", units_on, "cost:", cost_val)

powers = [Unit_power[k] for k in items]
valid = {bs:v for bs,v in counts.items() if sum(int(b)*p for b,p in zip(bs[::-1][:n], powers)) >= P_demand}
print("Valid bitstrings (power>=P_demand):", valid)

threshold = 500
powers = [Unit_power[k] for k in items]
filtered_counts = {bs: cnt for bs, cnt in counts.items() if cnt > threshold}

print(f"Bitstrings appearing more than {threshold} times:")
for bs, cnt in sorted(filtered_counts.items(), key=lambda kv: -kv[1]):
    total_power = sum(int(bit) * p for bit, p in zip(bs[::-1][:n], powers))
    print(f"{bs}: {cnt} times, Total power: {total_power:.3f}")

power_threshold = 0.5
power_filtered = {bs: cnt for bs, cnt in filtered_counts.items()
                  if sum(int(bit)*p for bit, p in zip(bs[::-1][:n], powers)) >= power_threshold}

print(f"\nBitstrings with total power >= {power_threshold}:")
for bs, cnt in sorted(power_filtered.items(), key=lambda kv: -kv[1]):
    total_power = sum(int(bit)*p for bit, p in zip(bs[::-1][:n], powers))
    print(f"{bs}: {cnt} times, Total power: {total_power:.3f}")

valid_bitstrings = list(power_filtered.keys())
print(f"\nValid bitstrings array: {valid_bitstrings}")

truncated_unique = list(dict.fromkeys(valid_bitstrings))
print(f"\nTruncated and unique bitstrings array: {truncated_unique}")
