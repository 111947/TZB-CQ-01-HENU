import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator

# ---- 构建用户自定义电路 ----
def build_custom_circuit():
    qc = QuantumCircuit(3)
    qc.h(0); qc.rz(0.027318, 0)
    qc.h(1); qc.rz(0.81954, 1)
    qc.h(2); qc.rz(0.068295, 2)

    qc.cx(1,2); qc.rz(0.647, 2); qc.cx(1,2)
    qc.cx(0,2); qc.rz(0.021567, 2); qc.cx(0,2)
    qc.cx(0,1); qc.rz(0.2588, 1); qc.cx(0,1)

    qc.rx(-0.98987, 0)
    qc.rx(-0.98987, 1)
    qc.rx(-0.98987, 2)

    qc.save_density_matrix()
    return qc

# ---- 理想密度矩阵 ----
def get_ideal_density():
    sim = AerSimulator(method='density_matrix')
    qc = build_custom_circuit()
    result = sim.run(qc).result()
    return DensityMatrix(result.data(0)['density_matrix'])

# ---- NumPy 实现的 Depolarizing Noise ----
def single_qubit_depolarizing(rho, p, qubit_index):
    """
    对密度矩阵 rho 的 qubit_index 位施加 depolarizing 噪声
    """
    I = np.eye(2)
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])

    def apply(pauli):
        ops = [I, I, I]
        ops[qubit_index] = pauli
        U = np.kron(np.kron(ops[0], ops[1]), ops[2])
        return U @ rho @ U.conj().T

    return (1 - p) * rho + (p / 3) * (apply(X) + apply(Y) + apply(Z))

def apply_depolarizing_noise_to_each_qubit(rho, p):
    """
    对每个量子比特施加 depolarizing 通道
    """
    for qubit in range(3):
        rho = single_qubit_depolarizing(rho, p, qubit)
    return rho

# ---- 密度矩阵 ↔ 特征（实部/虚部）----
def density_to_features(rho):
    # rho: (8,8) complex
    real = np.real(rho).reshape(-1)
    imag = np.imag(rho).reshape(-1)
    feat = np.stack([real, imag], axis=-1)  # (64,2)
    return feat.astype(np.float32)

# def features_to_density(feat):
#     # feat: (64,2)
#     real = feat[:,0].reshape(8,8)
#     imag = feat[:,1].reshape(8,8)
#     return real + 1j*imag
def features_to_density(feat):
    real = feat[:, 0].reshape(8,8)
    imag = feat[:, 1].reshape(8,8)
    return real + 1j * imag

# ---- 数据集生成（含理想标签）----
def generate_dataset(n_samples=1000, noise_prob=0.1, seed=None):
    """
    生成 (noisy, ideal) 数据对，用于训练和验证。
    参数:
        n_samples: 样本数量
        noise_prob: 每个 qubit 的 depolarizing 概率
        seed: 随机种子
    返回:
        X: 输入 noisy 状态特征 (n, 64, 2)
        Y: 理想状态特征 (n, 64, 2)
        ideal_dm: 原始理想密度矩阵对象
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    ideal_dm = get_ideal_density()
    ideal = ideal_dm.data

    X_list = []
    Y_list = []

    for _ in range(n_samples):
        noisy = apply_depolarizing_noise_to_each_qubit(ideal, noise_prob)
        X_list.append(density_to_features(noisy))   # 输入 noisy 特征
        Y_list.append(density_to_features(ideal))   # 标签 ideal 特征

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, ideal_dm  # 返回 ideal_dm 方便主程序评估
