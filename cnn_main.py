# # main.py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
#
# from qiskit.quantum_info import DensityMatrix, state_fidelity
# from quantum_data_generator import generate_dataset, get_ideal_density, density_to_features, features_to_density,apply_depolarizing_noise_to_each_qubit
# # from quantum_data_generator import generate_dataset, get_ideal_density, density_to_features, features_to_density, sample_depolarizing_noise
# from rnn_denoiser import CNN_Denoiser
# from postprocess import project_to_physical, trace_distance
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def visualize_density(ax, title, rho):
#     im = ax.imshow(np.real(rho), cmap='viridis')
#     ax.set_title(title)
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#
# def train_epoch(model, optimizer, loss_fn, Xb, Yb):
#     model.train()
#     optimizer.zero_grad()
#     pred = model(Xb)
#     loss = loss_fn(pred, Yb)
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
# def eval_epoch(model, loss_fn, Xb, Yb):
#     model.eval()
#     with torch.no_grad():
#         pred = model(Xb)
#         loss = loss_fn(pred, Yb).item()
#     return loss
#
# def main(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     print("Generating dataset...")
#     X, Y, ideal_dm = generate_dataset(n_samples=10000, noise_prob=0.2, seed=seed)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
#
#     # 转tensor
#     X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
#     Y_train = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
#     X_test  = torch.tensor(X_test,  dtype=torch.float32, device=DEVICE)
#     Y_test  = torch.tensor(Y_test,  dtype=torch.float32, device=DEVICE)
#
#     # reshape到CNN输入格式 (B, 2, 8, 8)
#     X_train_cnn = X_train.permute(0, 2, 1).reshape(-1, 2, 8, 8)
#     Y_train_cnn = Y_train.permute(0, 2, 1).reshape(-1, 2, 8, 8)
#     X_test_cnn = X_test.permute(0, 2, 1).reshape(-1, 2, 8, 8)
#     Y_test_cnn = Y_test.permute(0, 2, 1).reshape(-1, 2, 8, 8)
#
#     model = CNN_Denoiser().to(DEVICE)
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     print("Training...")
#     epochs = 200
#     for ep in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         output = model(X_train_cnn)
#         loss = loss_fn(output, Y_train_cnn)
#         loss.backward()
#         optimizer.step()
#
#         if (ep+1) % 5 == 0:
#             model.eval()
#             with torch.no_grad():
#                 val_output = model(X_test_cnn)
#                 val_loss = loss_fn(val_output, Y_test_cnn)
#             print(f"Epoch {ep+1:02d} | Train Loss: {loss.item():.6f} | Test Loss: {val_loss.item():.6f}")
#
#     # 评估新样本去噪
#     print("\nEvaluating on fresh noisy sample...")
#     ideal_rho = ideal_dm.data
#     noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho,0.2)
#
#     x_feat = density_to_features(noisy_rho)   # (64, 2)
#     x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,64,2)
#     x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, 8, 8)  # 转为CNN输入格式
#
#     model.eval()
#     with torch.no_grad():
#         denoised_out = model(x_tensor)  # (1, 2, 8, 8)
#     denoised_out = denoised_out.reshape(1, 2, 64).permute(0, 2, 1).cpu().numpy()[0]  # (64, 2)
#     denoised_rho = features_to_density(denoised_out).astype(np.complex128)
#     denoised_rho = project_to_physical(denoised_rho)
#
#
#     # 下面加入调试打印（可选，帮助定位问题）
#     print("Hermitian diff:", np.max(np.abs(denoised_rho - denoised_rho.conj().T)))
#     print("Trace:", np.trace(denoised_rho))
#     eigvals = np.linalg.eigvalsh(denoised_rho)
#     print("Eigenvalues min/max:", eigvals.min(), eigvals.max())
#
#     fig, axs = plt.subplots(1, 3, figsize=(15,4))
#     visualize_density(axs[0], "Ideal", ideal_rho)
#     visualize_density(axs[1], "Noisy (p=0.2 sample)", noisy_rho)
#     visualize_density(axs[2], "Denoised (RNN)", denoised_rho)
#     plt.suptitle("Density Matrix Real Parts")
#     plt.tight_layout()
#     plt.show()
#
#     # ----- 指标 -----
#     # Qiskit fidelity
#     fid = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))
#     tdist = trace_distance(denoised_rho, ideal_rho)
#     print(f"Fidelity to ideal: {fid:.6f}")
#     print(f"Trace distance to ideal: {tdist:.6f}")
#     # 条形图可视化各比特串概率
#     plot_probabilities_bar(ideal_rho, noisy_rho, denoised_rho)
#
# def plot_probabilities_bar(ideal, noisy, denoised):
#     labels = [f'{i:03b}' for i in range(8)]
#     idxs = np.arange(len(labels))
#
#     def get_probs(rho):
#         return np.real(np.diag(rho))  # 对角线元素为概率
#
#     ideal_probs = get_probs(ideal)
#     noisy_probs = get_probs(noisy)
#     denoised_probs = get_probs(denoised)
#
#     width = 0.25
#
#     plt.figure(figsize=(10,6))
#     plt.bar(idxs - width, ideal_probs, width=width, label='Ideal State')
#     plt.bar(idxs, noisy_probs, width=width, label='Noisy State')
#     plt.bar(idxs + width, denoised_probs, width=width, label='Enhanced Mitigation')
#
#     plt.xticks(idxs, labels)
#     plt.ylabel("Probability")
#     plt.xlabel("Basis States")
#     plt.title("Basis State Distribution Comparison")
#     plt.legend()
#     plt.grid(True, axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     main()


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from qiskit.quantum_info import entropy
from qiskit.quantum_info import DensityMatrix, state_fidelity
from quantum_data_generator import generate_dataset, get_ideal_density, density_to_features, features_to_density, \
    apply_depolarizing_noise_to_each_qubit
from rnn_denoiser import CNN_Denoiser
from postprocess import project_to_physical, trace_distance
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, state_fidelity, entropy
from qiskit_aer import AerSimulator
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_density(ax, title, rho):
    im = ax.imshow(np.real(rho), cmap='viridis')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def train_epoch(model, optimizer, loss_fn, Xb, Yb):
    model.train()
    optimizer.zero_grad()
    pred = model(Xb)
    loss = loss_fn(pred, Yb)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_epoch(model, loss_fn, Xb, Yb):
    model.eval()
    with torch.no_grad():
        pred = model(Xb)
        loss = loss_fn(pred, Yb).item()
    return loss


def calculate_quantum_metrics(ideal_rho, noisy_rho, denoised_rho):
    """计算量子态的各项指标"""
    # 保真度
    fid_noisy = state_fidelity(DensityMatrix(noisy_rho), DensityMatrix(ideal_rho))
    fid_denoised = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))

    # 迹距离
    tdist_noisy = trace_distance(noisy_rho, ideal_rho)
    tdist_denoised = trace_distance(denoised_rho, ideal_rho)

    # 纯度
    purity_ideal = np.trace(ideal_rho @ ideal_rho).real
    purity_noisy = np.trace(noisy_rho @ noisy_rho).real
    purity_denoised = np.trace(denoised_rho @ denoised_rho).real

    # 冯诺依曼熵
    entropy_ideal = entropy(DensityMatrix(ideal_rho))
    entropy_noisy = entropy(DensityMatrix(noisy_rho))
    entropy_denoised = entropy(DensityMatrix(denoised_rho))

    return {
        'fidelity': {
            'noisy': fid_noisy,
            'denoised': fid_denoised,
            'improvement': fid_denoised - fid_noisy
        },
        'trace_distance': {
            'noisy': tdist_noisy,
            'denoised': tdist_denoised,
            'improvement': tdist_noisy - tdist_denoised
        },
        'purity': {
            'ideal': purity_ideal,
            'noisy': purity_noisy,
            'denoised': purity_denoised
        },
        'entropy': {
            'ideal': entropy_ideal,
            'noisy': entropy_noisy,
            'denoised': entropy_denoised
        }
    }


# def print_final_metrics(metrics, training_time, inference_time):
#     """打印最终汇总指标"""
#     print("\n=== 最终汇总指标 ===")
#
#     # 1. 基础指标
#     print("\n--- 基础指标 ---")
#     print(f"训练时间: {training_time:.2f}秒")
#     print(f"单个样本推理时间: {inference_time:.6f}秒")
#
#     # 2. 量子态质量指标
#     print("\n--- 量子态质量指标 (p=0.2噪声水平) ---")
#     print(f"保真度: 噪声态={metrics['fidelity']['noisy']:.6f} → 去噪后={metrics['fidelity']['denoised']:.6f} "
#           f"(提升: {metrics['fidelity']['improvement']:.6f})")
#     print(
#         f"迹距离: 噪声态={metrics['trace_distance']['noisy']:.6f} → 去噪后={metrics['trace_distance']['denoised']:.6f} "
#         f"(改善: {metrics['trace_distance']['improvement']:.6f})")
#
#     # 3. 量子态特性
#     print("\n--- 量子态特性 ---")
#     print(f"纯度: 理想态={metrics['purity']['ideal']:.6f} | 噪声态={metrics['purity']['noisy']:.6f} | "
#           f"去噪后={metrics['purity']['denoised']:.6f}")
#     print(f"冯诺依曼熵: 理想态={metrics['entropy']['ideal']:.6f} | 噪声态={metrics['entropy']['noisy']:.6f} | "
#           f"去噪后={metrics['entropy']['denoised']:.6f}")
#
#     # 4. 不同噪声水平下的性能
#     print("\n--- 不同噪声水平下的保真度 ---")
#     noise_levels = [0.05, 0.1, 0.2]
#     for p in noise_levels:
#         noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, p)
#
#         # 去噪处理
#         x_feat = density_to_features(noisy_rho)
#         x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#         x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, input_size, input_size)
#
#         with torch.no_grad():
#             denoised_out = model(x_tensor)
#         denoised_out = denoised_out.reshape(1, 2, input_size ** 2).permute(0, 2, 1).cpu().numpy()[0]
#         denoised_rho = features_to_density(denoised_out).astype(np.complex128)
#         denoised_rho = project_to_physical(denoised_rho)
#
#         fid_noisy = state_fidelity(DensityMatrix(noisy_rho), DensityMatrix(ideal_rho))
#         fid_denoised = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))
#
#         print(f"p={p:.2f}: 噪声态={fid_noisy:.6f} | 去噪后={fid_denoised:.6f} | "
#               f"提升={fid_denoised - fid_noisy:.6f}")
#
#     # 5. 鲁棒性评估 (p=0.1 ±20%变化)
#     print("\n--- 鲁棒性评估 (p=0.1 ±20%变化) ---")
#     p = 0.1
#     num_tests = 10
#     noisy_fids = []
#     denoised_fids = []
#
#     for _ in range(num_tests):
#         varied_p = p * np.random.uniform(0.8, 1.2)
#         noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, varied_p)
#
#         # 去噪处理
#         x_feat = density_to_features(noisy_rho)
#         x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#         x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, input_size, input_size)
#
#         with torch.no_grad():
#             denoised_out = model(x_tensor)
#         denoised_out = denoised_out.reshape(1, 2, input_size ** 2).permute(0, 2, 1).cpu().numpy()[0]
#         denoised_rho = features_to_density(denoised_out).astype(np.complex128)
#         denoised_rho = project_to_physical(denoised_rho)
#
#         fid_noisy = state_fidelity(DensityMatrix(noisy_rho), DensityMatrix(ideal_rho))
#         fid_denoised = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))
#
#         noisy_fids.append(fid_noisy)
#         denoised_fids.append(fid_denoised)
#
#     print(f"噪声态保真度: {np.mean(noisy_fids):.6f} ± {np.std(noisy_fids):.6f}")
#     print(f"去噪后保真度: {np.mean(denoised_fids):.6f} ± {np.std(denoised_fids):.6f}")
#     print(f"平均提升: {np.mean(np.array(denoised_fids) - np.array(noisy_fids)):.6f}")

def print_final_metrics(metrics, training_time, inference_time, ideal_rho, model, input_size,gate_count):
    """打印最终汇总指标"""
    print("\n=== 最终汇总指标 ===")

    # 1. 基础指标
    print("\n--- 基础指标 ---")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"单个样本推理时间: {inference_time:.6f}秒")
    # 新增门开销信息
    print("\n--- 量子电路门开销 ---")
    print(f"单量子位门数量: {gate_count['single_qubit']}")
    print(f"双量子位门数量: {gate_count['two_qubit']}")
    print(f"总门数量: {gate_count['total']}")
    # 2. 量子态质量指标
    print("\n--- 量子态质量指标 (p=0.2噪声水平) ---")
    print(f"保真度: 噪声态={metrics['fidelity']['noisy']:.6f} → 去噪后={metrics['fidelity']['denoised']:.6f} "
          f"(提升: {metrics['fidelity']['improvement']:.6f})")
    print(
        f"迹距离: 噪声态={metrics['trace_distance']['noisy']:.6f} → 去噪后={metrics['trace_distance']['denoised']:.6f} "
        f"(改善: {metrics['trace_distance']['improvement']:.6f})")

    # 3. 量子态特性
    print("\n--- 量子态特性 ---")
    print(f"纯度: 理想态={metrics['purity']['ideal']:.6f} | 噪声态={metrics['purity']['noisy']:.6f} | "
          f"去噪后={metrics['purity']['denoised']:.6f}")
    print(f"冯诺依曼熵: 理想态={metrics['entropy']['ideal']:.6f} | 噪声态={metrics['entropy']['noisy']:.6f} | "
          f"去噪后={metrics['entropy']['denoised']:.6f}")

    # 4. 不同噪声水平下的性能
    print("\n--- 不同噪声水平下的保真度 ---")
    noise_levels = [0.08, 0.15, 0.18]
    for p in noise_levels:
        noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, p)

        # 去噪处理
        x_feat = density_to_features(noisy_rho)
        x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, input_size, input_size)

        with torch.no_grad():
            denoised_out = model(x_tensor)
        denoised_out = denoised_out.reshape(1, 2, input_size ** 2).permute(0, 2, 1).cpu().numpy()[0]
        denoised_rho = features_to_density(denoised_out).astype(np.complex128)
        denoised_rho = project_to_physical(denoised_rho)

        fid_noisy = state_fidelity(DensityMatrix(noisy_rho), DensityMatrix(ideal_rho))
        fid_denoised = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))

        print(f"p={p:.2f}: 噪声态={fid_noisy:.6f} | 去噪后={fid_denoised:.6f} | "
              f"提升={fid_denoised - fid_noisy:.6f}")

    # 5. 鲁棒性评估 (p=0.1 ±20%变化)
    print("\n--- 鲁棒性评估 (p=0.1 ±20%变化) ---")
    p = 0.1
    num_tests = 10
    noisy_fids = []
    denoised_fids = []

    for _ in range(num_tests):
        varied_p = p * np.random.uniform(0.8, 1.2)
        noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, varied_p)

        # 去噪处理
        x_feat = density_to_features(noisy_rho)
        x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, input_size, input_size)

        with torch.no_grad():
            denoised_out = model(x_tensor)
        denoised_out = denoised_out.reshape(1, 2, input_size ** 2).permute(0, 2, 1).cpu().numpy()[0]
        denoised_rho = features_to_density(denoised_out).astype(np.complex128)
        denoised_rho = project_to_physical(denoised_rho)

        fid_noisy = state_fidelity(DensityMatrix(noisy_rho), DensityMatrix(ideal_rho))
        fid_denoised = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))

        noisy_fids.append(fid_noisy)
        denoised_fids.append(fid_denoised)

    print(f"噪声态保真度: {np.mean(noisy_fids):.6f} ± {np.std(noisy_fids):.6f}")
    print(f"去噪后保真度: {np.mean(denoised_fids):.6f} ± {np.std(denoised_fids):.6f}")
    print(f"平均提升: {np.mean(np.array(denoised_fids) - np.array(noisy_fids)):.6f}")
# 评估随机尺寸零样本场景（修改版）
def evaluate_random_size_zero_shots(model, noise_prob=0.2):
    """由于generate_dataset不支持n_qubits参数，改为测试相同尺寸下的不同噪声水平"""
    noise_levels = [0.1, 0.2, 0.3]  # 测试不同噪声水平作为替代方案
    results = []

    for noise in noise_levels:
        # 生成数据集（使用函数默认的量子比特数量）
        X, Y, ideal_dm = generate_dataset(n_samples=1000, noise_prob=noise)
        ideal_rho = ideal_dm.data
        n_qubits = int(np.log2(ideal_rho.shape[0]))  # 从密度矩阵维度推断量子比特数量

        # 计算指标
        total_incoherent = 0.0
        total_coherent = 0.0

        for i in range(100):  # 随机采样100个样本评估
            # 获取带噪声和去噪后的密度矩阵
            noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, noise)

            # 准备模型输入
            x_feat = density_to_features(noisy_rho)
            x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # 根据实际量子比特数量调整输入形状
            x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, 2 ** n_qubits, 2 ** n_qubits)

            # 模型去噪
            model.eval()
            with torch.no_grad():
                denoised_out = model(x_tensor)
            denoised_out = denoised_out.reshape(1, 2, 2 ** (2 * n_qubits)).permute(0, 2, 1).cpu().numpy()[0]
            denoised_rho = features_to_density(denoised_out).astype(np.complex128)
            denoised_rho = project_to_physical(denoised_rho)

            # 计算指标
            fid = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))
            incoherent = 1 - fid  # 不一致性：1 - 保真度
            coherent = np.exp(-incoherent)  # 一致性：指数衰减
            total_incoherent += incoherent
            total_coherent += coherent

        # 计算平均值
        avg_incoherent = total_incoherent / 100
        avg_coherent = total_coherent / 100
        provider = np.mean([avg_incoherent, avg_coherent])  # 综合指标

        results.append({
            'noise_level': noise,
            'incoherent': avg_incoherent,
            'coherent': avg_coherent,
            'provider': provider
        })

    # 计算总体平均
    avg_incoherent = np.mean([r['incoherent'] for r in results])
    avg_coherent = np.mean([r['coherent'] for r in results])
    avg_provider = np.mean([r['provider'] for r in results])

    return {
        'incoherent': avg_incoherent,
        'coherent': avg_coherent,
        'provider': avg_provider,
        'details': results
    }


# 评估Trotter步骤零样本场景（修改版）
def evaluate_trotter_step_zero_shots(model, noise_prob=0.2):
    steps = [1, 2, 3]  # 不同Trotter步骤
    results = []

    # 生成基础数据集（使用默认量子比特数量）
    X, Y, ideal_dm = generate_dataset(n_samples=1000, noise_prob=noise_prob)
    base_ideal_rho = ideal_dm.data
    n_qubits = int(np.log2(base_ideal_rho.shape[0]))  # 推断量子比特数量

    for step in steps:
        # 模拟多步Trotter演化（通过叠加噪声模拟）
        total_incoherent = 0.0
        total_coherent = 0.0

        for i in range(100):
            # 应用多步噪声
            noisy_rho = base_ideal_rho.copy()
            for _ in range(step):
                noisy_rho = apply_depolarizing_noise_to_each_qubit(noisy_rho, noise_prob / step)

            # 准备模型输入
            x_feat = density_to_features(noisy_rho)
            x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, 2 ** n_qubits, 2 ** n_qubits)

            # 模型去噪
            model.eval()
            with torch.no_grad():
                denoised_out = model(x_tensor)
            denoised_out = denoised_out.reshape(1, 2, 2 ** (2 * n_qubits)).permute(0, 2, 1).cpu().numpy()[0]
            denoised_rho = features_to_density(denoised_out).astype(np.complex128)
            denoised_rho = project_to_physical(denoised_rho)

            # 计算指标
            fid = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(base_ideal_rho))
            incoherent = 1 - fid
            coherent = np.exp(-incoherent * step)  # 随步骤数调整
            total_incoherent += incoherent
            total_coherent += coherent

        # 计算平均值
        avg_incoherent = total_incoherent / 100
        avg_coherent = total_coherent / 100
        provider = avg_coherent / (step + 1)  # 考虑步骤数的综合指标

        results.append({
            'step': step,
            'incoherent': avg_incoherent,
            'coherent': avg_coherent,
            'provider': provider
        })

    # 计算总体平均
    avg_incoherent = np.mean([r['incoherent'] for r in results])
    avg_coherent = np.mean([r['coherent'] for r in results])
    avg_provider = np.mean([r['provider'] for r in results])

    return {
        'incoherent': avg_incoherent,
        'coherent': avg_coherent,
        'provider': avg_provider,
        'details': results
    }


# 评估未见观测场景（修改版）
def evaluate_unseen_observables(model, noise_prob=0.2):
    # 定义不同的观测器（从密度矩阵提取不同特征）
    def obs1(rho, n_qubits):
        # X0观测
        return np.real(np.trace(rho @ np.kron(np.array([[0, 1], [1, 0]]), np.eye(2 ** (n_qubits - 1)))))

    def obs2(rho, n_qubits):
        # Yn-1观测
        return np.real(np.trace(rho @ np.kron(np.eye(2 ** (n_qubits - 1)), np.array([[0, -1j], [1j, 0]]))))

    def obs3(rho, n_qubits):
        # Z0⊗Zn-1观测
        z = np.array([[1, 0], [0, -1]])
        return np.real(np.trace(rho @ np.kron(z, np.kron(np.eye(2 ** (n_qubits - 2)), z))))

    observables = [obs1, obs2, obs3]
    results = []

    # 生成基础数据（使用默认量子比特数量）
    X, Y, ideal_dm = generate_dataset(n_samples=500, noise_prob=noise_prob)
    ideal_rho = ideal_dm.data
    n_qubits = int(np.log2(ideal_rho.shape[0]))  # 推断量子比特数量

    for i, obs in enumerate(observables):
        total_incoherent = 0.0
        total_coherent = 0.0

        for _ in range(100):
            # 获取带噪声状态
            noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, noise_prob)

            # 计算理想值和噪声值
            ideal_val = obs(ideal_rho, n_qubits)
            noisy_val = obs(noisy_rho, n_qubits)

            # 模型去噪
            x_feat = density_to_features(noisy_rho)
            x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, 2 ** n_qubits, 2 ** n_qubits)

            model.eval()
            with torch.no_grad():
                denoised_out = model(x_tensor)
            denoised_out = denoised_out.reshape(1, 2, 2 ** (2 * n_qubits)).permute(0, 2, 1).cpu().numpy()[0]
            denoised_rho = features_to_density(denoised_out).astype(np.complex128)
            denoised_rho = project_to_physical(denoised_rho)

            # 计算缓解后的值
            mitigated_val = obs(denoised_rho, n_qubits)

            # 计算指标
            incoherent = abs(noisy_val - ideal_val)
            coherent = 1 - abs(mitigated_val - ideal_val)
            total_incoherent += incoherent
            total_coherent += coherent

        # 计算平均值
        avg_incoherent = total_incoherent / 100
        avg_coherent = total_coherent / 100
        provider = (avg_coherent - avg_incoherent) / 2  # 改进度指标

        results.append({
            'observable': i + 1,
            'incoherent': avg_incoherent,
            'coherent': avg_coherent,
            'provider': provider
        })

    # 计算总体平均
    avg_incoherent = np.mean([r['incoherent'] for r in results])
    avg_coherent = np.mean([r['coherent'] for r in results])
    avg_provider = np.mean([r['provider'] for r in results])

    return {
        'incoherent': avg_incoherent,
        'coherent': avg_coherent,
        'provider': avg_provider,
        'details': results
    }


def calculate_gate_count(qc):
    """计算量子电路的门开销"""
    gate_count = {
        'single_qubit': 0,
        'two_qubit': 0,
        'total': 0
    }

    for instruction in qc.data:
        gate = instruction.operation
        if len(instruction.qubits) == 1:
            gate_count['single_qubit'] += 1
        elif len(instruction.qubits) == 2:
            gate_count['two_qubit'] += 1
        gate_count['total'] += 1

    return gate_count

def main(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Generating dataset...")
    X, Y, ideal_dm = generate_dataset(n_samples=10000, noise_prob=0.2, seed=seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    # 转tensor
    X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    Y_test = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE)

    # 从密度矩阵推断量子比特数量
    ideal_rho = ideal_dm.data
    n_qubits = int(np.log2(ideal_rho.shape[0]))
    print(f"推断的量子比特数量: {n_qubits}")

    # reshape到CNN输入格式 (B, 2, 2^n, 2^n)
    input_size = 2 ** n_qubits
    X_train_cnn = X_train.permute(0, 2, 1).reshape(-1, 2, input_size, input_size)
    Y_train_cnn = Y_train.permute(0, 2, 1).reshape(-1, 2, input_size, input_size)
    X_test_cnn = X_test.permute(0, 2, 1).reshape(-1, 2, input_size, input_size)
    Y_test_cnn = Y_test.permute(0, 2, 1).reshape(-1, 2, input_size, input_size)

    model = CNN_Denoiser().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training...")
    epochs = 100
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_cnn)
        loss = loss_fn(output, Y_train_cnn)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_cnn)
                val_loss = loss_fn(val_output, Y_test_cnn)
            print(f"Epoch {ep + 1:02d} | Train Loss: {loss.item():.6f} | Test Loss: {val_loss.item():.6f}")

    # 评估新样本去噪
    print("\nEvaluating on fresh noisy sample...")
    noisy_rho = apply_depolarizing_noise_to_each_qubit(ideal_rho, 0.2)

    x_feat = density_to_features(noisy_rho)  # (64, 2)
    x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,64,2)
    x_tensor = x_tensor.permute(0, 2, 1).reshape(1, 2, input_size, input_size)  # 转为CNN输入格式

    model.eval()
    with torch.no_grad():
        denoised_out = model(x_tensor)  # (1, 2, input_size, input_size)
    denoised_out = denoised_out.reshape(1, 2, input_size ** 2).permute(0, 2, 1).cpu().numpy()[0]  # (64, 2)
    denoised_rho = features_to_density(denoised_out).astype(np.complex128)
    denoised_rho = project_to_physical(denoised_rho)

    # 调试打印
    print("Hermitian diff:", np.max(np.abs(denoised_rho - denoised_rho.conj().T)))
    print("Trace:", np.trace(denoised_rho))
    eigvals = np.linalg.eigvalsh(denoised_rho)
    print("Eigenvalues min/max:", eigvals.min(), eigvals.max())

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    visualize_density(axs[0], "Ideal", ideal_rho)
    visualize_density(axs[1], "Noisy (p=0.2 sample)", noisy_rho)
    visualize_density(axs[2], "Denoised (CNN)", denoised_rho)
    plt.suptitle("Density Matrix Real Parts")
    plt.tight_layout()
    plt.show()

    # 基础指标
    fid = state_fidelity(DensityMatrix(denoised_rho), DensityMatrix(ideal_rho))
    tdist = trace_distance(denoised_rho, ideal_rho)
    print(f"Fidelity to ideal: {fid:.6f}")
    print(f"Trace distance to ideal: {tdist:.6f}")
    plot_probabilities_bar(ideal_rho, noisy_rho, denoised_rho)

    # 扩展性能指标评估
    print("\n=== 扩展性能指标评估 ===")

    # 1. 随机尺寸零样本评估（改为测试不同噪声水平）
    random_size_results = evaluate_random_size_zero_shots(model)
    print("\nRandom Size Zero Shots (不同噪声水平替代):")
    print(f"  Incoherent: {random_size_results['incoherent']:.4f}")
    print(f"  Coherent: {random_size_results['coherent']:.4f}")
    print(f"  Provider: {random_size_results['provider']:.4f}")

    # 2. Trotter步骤零样本评估
    trotter_results = evaluate_trotter_step_zero_shots(model)
    print("\nTrotter Step Zero Shots:")
    print(f"  Incoherent: {trotter_results['incoherent']:.4f}")
    print(f"  Coherent: {trotter_results['coherent']:.4f}")
    print(f"  Provider: {trotter_results['provider']:.4f}")

    # 3. 未见观测评估
    unseen_results = evaluate_unseen_observables(model)
    print("\nUnseen Observables:")
    print(f"  Incoherent: {unseen_results['incoherent']:.4f}")
    print(f"  Coherent: {unseen_results['coherent']:.4f}")
    print(f"  Provider: {unseen_results['provider']:.4f}")


    metrics = calculate_quantum_metrics(ideal_rho, noisy_rho, denoised_rho)

    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        _ = model(x_tensor)
    inference_time = time.time() - start_time
    # 训练时间测量
    print("Training...")
    epochs = 100
    start_time = time.time()  # 训练开始时间

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_cnn)
        loss = loss_fn(output, Y_train_cnn)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_cnn)
                val_loss = loss_fn(val_output, Y_test_cnn)
            print(f"Epoch {ep + 1:02d} | Train Loss: {loss.item():.6f} | Test Loss: {val_loss.item():.6f}")

    # ... 添加您的实际量子门操作 ...
    qc = QuantumCircuit(3)
    qc.h(0);
    qc.rz(0.027318, 0)
    qc.h(1);
    qc.rz(0.81954, 1)
    qc.h(2);
    qc.rz(0.068295, 2)

    qc.cx(1, 2);
    qc.rz(0.647, 2);
    qc.cx(1, 2)
    qc.cx(0, 2);
    qc.rz(0.021567, 2);
    qc.cx(0, 2)
    qc.cx(0, 1);
    qc.rz(0.2588, 1);
    qc.cx(0, 1)

    qc.rx(-0.98987, 0)
    qc.rx(-0.98987, 1)
    qc.rx(-0.98987, 2)

    qc.save_density_matrix()
    gate_count = calculate_gate_count(qc)
    total_train_time = time.time() - start_time  # 计算总训练时间
    # 打印最终汇总指标
    # print_final_metrics(metrics, epochs, inference_time)
    # 在main()函数末尾：
    print_final_metrics(metrics, total_train_time, inference_time, ideal_rho, model, input_size,gate_count)

def plot_probabilities_bar(ideal, noisy, denoised):
    n_qubits = int(np.log2(ideal.shape[0]))
    n_states = 2 ** n_qubits
    labels = [f'{i:0{n_qubits}b}' for i in range(n_states)]
    idxs = np.arange(len(labels))

    def get_probs(rho):
        return np.real(np.diag(rho))  # 对角线元素为概率

    ideal_probs = get_probs(ideal)
    noisy_probs = get_probs(noisy)
    denoised_probs = get_probs(denoised)

    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(idxs - width, ideal_probs, width=width, label='Ideal State')
    plt.bar(idxs, noisy_probs, width=width, label='Noisy State')
    plt.bar(idxs + width, denoised_probs, width=width, label='Enhanced Mitigation')

    plt.xticks(idxs, labels)
    plt.ylabel("Probability")
    plt.xlabel("Basis States")
    plt.title("Basis State Distribution Comparison")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
