# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
#
# # 量子相关库
# from qiskit import QuantumCircuit
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error
# from qiskit.quantum_info import DensityMatrix, state_fidelity
# from qiskit import transpile
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
# # 设置中文字体
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
#
#
# class QuantumDataset(Dataset):
#     """量子数据集类，用于生成和存储带噪声的量子态数据及对应的理想态数据"""
#
#     def __init__(self, n_qubits, num_samples=1000, noise_level=0.01, noise_variation=0.005):
#         self.n_qubits = n_qubits
#         self.num_samples = num_samples
#         self.noise_level = noise_level
#         self.noise_variation = noise_variation
#
#         # 生成目标密度矩阵
#         self.target_dm = self.build_target_density()
#         self.ideal_probs = np.real(np.diag(self.target_dm.data))
#
#         # 生成带噪声的样本
#         self.noisy_samples, self.ideal_samples = self.generate_noisy_samples()
#
#         # 数据标准化
#         self.scaler = MinMaxScaler()
#         self.noisy_samples_scaled = self.scaler.fit_transform(self.noisy_samples)
#
#     def build_target_density(self):
#         """构建目标量子电路并返回其密度矩阵"""
#         n_qubits = self.n_qubits
#         qc = QuantumCircuit(n_qubits)
#
#         # 目标电路实现
#         qc.h(0)
#         qc.rz(0.027318, 0)
#         qc.h(1)
#         qc.rz(0.81954, 1)
#         qc.h(2)
#         qc.rz(0.068295, 2)
#
#         qc.cx(1, 2)
#         qc.rz(0.647, 2)
#         qc.cx(1, 2)
#
#         qc.cx(0, 2)
#         qc.rz(0.021567, 2)
#         qc.cx(0, 2)
#
#         qc.cx(0, 1)
#         qc.rz(0.2588, 1)
#         qc.cx(0, 1)
#
#         qc.rx(-0.98987, 0)
#         qc.rx(-0.98987, 1)
#         qc.rx(-0.98987, 2)
#
#         qc.save_density_matrix()
#
#         sim = AerSimulator(method='density_matrix')
#         result = sim.run(qc).result()
#         return DensityMatrix(result.data(0)['density_matrix'])
#
#     def create_noise_model(self, noise_scale=1.0):
#         """创建带有去极化噪声的噪声模型"""
#         noise_model = NoiseModel()
#         effective_noise = self.noise_level * noise_scale
#
#         # 单量子位门噪声
#         single_qubit_gates = ['h', 'rz', 'rx']
#         for gate in single_qubit_gates:
#             for qubit in range(self.n_qubits):
#                 noise_model.add_quantum_error(
#                     depolarizing_error(effective_noise, 1),
#                     gate,
#                     [qubit]
#                 )
#
#         # 双量子位门噪声（通常双量子位门噪声更大）
#         two_qubit_error = depolarizing_error(effective_noise * 1.5, 2)
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])
#
#         return noise_model
#
#     def generate_noisy_samples(self):
#         """生成带噪声的量子态样本集"""
#         n_states = 2 ** self.n_qubits
#         noisy_samples = []
#         ideal_samples = []
#
#         # 理想状态的概率分布（用于所有样本的目标）
#         ideal_probs = np.real(np.diag(self.target_dm.data))
#
#         for _ in range(self.num_samples):
#             # 随机变化噪声水平，增加样本多样性
#             noise_scale = np.random.uniform(1 - self.noise_variation,
#                                             1 + self.noise_variation)
#
#             # 创建噪声模型
#             noise_model = self.create_noise_model(noise_scale)
#
#             # 构建带噪声的电路
#             qc = QuantumCircuit(self.n_qubits)
#             qc.h(0)
#             qc.rz(0.027318, 0)
#             qc.h(1)
#             qc.rz(0.81954, 1)
#             qc.h(2)
#             qc.rz(0.068295, 2)
#
#             qc.cx(1, 2)
#             qc.rz(0.647, 2)
#             qc.cx(1, 2)
#
#             qc.cx(0, 2)
#             qc.rz(0.021567, 2)
#             qc.cx(0, 2)
#
#             qc.cx(0, 1)
#             qc.rz(0.2588, 1)
#             qc.cx(0, 1)
#
#             qc.rx(-0.98987, 0)
#             qc.rx(-0.98987, 1)
#             qc.rx(-0.98987, 2)
#
#             qc.save_density_matrix()
#
#             # 运行带噪声的模拟
#             sim = AerSimulator(method='density_matrix')
#             result = sim.run(
#                 transpile(qc, sim),
#                 noise_model=noise_model
#             ).result()
#
#             noisy_dm = DensityMatrix(result.data(0)['density_matrix'])
#             noisy_probs = np.real(np.diag(noisy_dm.data))
#
#             noisy_samples.append(noisy_probs)
#             ideal_samples.append(ideal_probs)
#
#         return np.array(noisy_samples), np.array(ideal_samples)
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         return (torch.tensor(self.noisy_samples_scaled[idx], dtype=torch.float32),
#                 torch.tensor(self.ideal_samples[idx], dtype=torch.float32))
#
#
# class QuantumDenoisingMLP(nn.Module):
#     """用于量子态去噪的多层感知器模型"""
#
#     def __init__(self, input_size, hidden_sizes=[64, 128, 64], output_size=None):
#         super(QuantumDenoisingMLP, self).__init__()
#
#         # 如果未指定输出大小，则与输入大小相同
#         if output_size is None:
#             output_size = input_size
#
#         # 构建神经网络层
#         layers = []
#         prev_size = input_size
#
#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(prev_size, hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(hidden_size))
#             layers.append(nn.Dropout(0.2))
#             prev_size = hidden_size
#
#         # 输出层，使用sigmoid确保输出在0到1之间
#         layers.append(nn.Linear(prev_size, output_size))
#         layers.append(nn.Softmax(dim=1))  # 确保输出是有效的概率分布
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
#     """训练MLP模型"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     train_losses = []
#     val_losses = []
#
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#
#         for noisy, ideal in train_loader:
#             noisy, ideal = noisy.to(device), ideal.to(device)
#
#             # 前向传播
#             outputs = model(noisy)
#             loss = criterion(outputs, ideal)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * noisy.size(0)
#
#         # 计算平均训练损失
#         train_loss /= len(train_loader.dataset)
#         train_losses.append(train_loss)
#
#         # 在验证集上评估
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for noisy, ideal in val_loader:
#                 noisy, ideal = noisy.to(device), ideal.to(device)
#                 outputs = model(noisy)
#                 loss = criterion(outputs, ideal)
#                 val_loss += loss.item() * noisy.size(0)
#
#         val_loss /= len(val_loader.dataset)
#         val_losses.append(val_loss)
#
#         # 每10个epoch打印一次进度
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
#
#     # 绘制训练和验证损失
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='训练损失')
#     plt.plot(val_losses, label='验证损失')
#     plt.title('模型训练过程')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
#
#     return model
#
#
# def evaluate_performance(model, dataset, test_loader, scaler):
#     """评估模型性能并计算所需指标"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#
#     all_noisy = []
#     all_ideal = []
#     all_predicted = []
#
#     with torch.no_grad():
#         for noisy, ideal in test_loader:
#             noisy, ideal = noisy.to(device), ideal.to(device)
#             predicted = model(noisy)
#
#             all_noisy.extend(noisy.cpu().numpy())
#             all_ideal.extend(ideal.cpu().numpy())
#             all_predicted.extend(predicted.cpu().numpy())
#
#     all_noisy = np.array(all_noisy)
#     all_ideal = np.array(all_ideal)
#     all_predicted = np.array(all_predicted)
#
#     # 计算整体性能指标
#     mse_noisy = mean_squared_error(all_ideal, all_noisy)
#     mse_mlp = mean_squared_error(all_ideal, all_predicted)
#     r2_noisy = r2_score(all_ideal, all_noisy)
#     r2_mlp = r2_score(all_ideal, all_predicted)
#
#     print(f"\n=== 整体性能指标 ===")
#     print(f"含噪声状态与理想状态的MSE: {mse_noisy:.6f}")
#     print(f"MLP去噪后与理想状态的MSE: {mse_mlp:.6f}")
#     print(f"MSE改善比例: {(mse_noisy - mse_mlp) / mse_noisy:.2%}")
#     print(f"含噪声状态的R²分数: {r2_noisy:.6f}")
#     print(f"MLP去噪后的R²分数: {r2_mlp:.6f}")
#
#     # 生成特定测试样本的可视化结果
#     test_qc = QuantumCircuit(dataset.n_qubits)
#     test_qc.h(0)
#     test_qc.rz(0.027318, 0)
#     test_qc.h(1)
#     test_qc.rz(0.81954, 1)
#     test_qc.h(2)
#     test_qc.rz(0.068295, 2)
#
#     test_qc.cx(1, 2)
#     test_qc.rz(0.647, 2)
#     test_qc.cx(1, 2)
#
#     test_qc.cx(0, 2)
#     test_qc.rz(0.021567, 2)
#     test_qc.cx(0, 2)
#
#     test_qc.cx(0, 1)
#     test_qc.rz(0.2588, 1)
#     test_qc.cx(0, 1)
#
#     test_qc.rx(-0.98987, 0)
#     test_qc.rx(-0.98987, 1)
#     test_qc.rx(-0.98987, 2)
#
#     test_qc.save_density_matrix()
#
#     # 获取带噪声的状态
#     noise_model = dataset.create_noise_model(noise_scale=1.0)
#     sim = AerSimulator(method='density_matrix')
#     result_noisy = sim.run(
#         transpile(test_qc, sim),
#         noise_model=noise_model
#     ).result()
#     noisy_dm = DensityMatrix(result_noisy.data(0)['density_matrix'])
#     noisy_probs = np.real(np.diag(noisy_dm.data))
#
#     # 理想状态
#     ideal_probs = dataset.ideal_probs
#
#     # MLP去噪后的状态
#     noisy_scaled = scaler.transform([noisy_probs])
#     with torch.no_grad():
#         mlp_probs = model(torch.tensor(noisy_scaled, dtype=torch.float32).to(device))
#         mlp_probs = mlp_probs.cpu().numpy()[0]
#
#     # 计算保真度
#     ideal_dm = dataset.target_dm
#     mlp_dm = DensityMatrix(np.diag(mlp_probs))  # 简化处理，实际应考虑完整密度矩阵
#     noisy_fidelity = state_fidelity(ideal_dm, noisy_dm)
#     mlp_fidelity = state_fidelity(ideal_dm, mlp_dm)
#
#     # 计算所需的三个场景下的指标
#     metrics = {
#         "Random Size Zero Shots": {
#             "Incoherent": 1 - r2_noisy,  # 不一致性：用R²的补数表示
#             "Coherent": noisy_fidelity,  # 一致性：用保真度表示
#             "Provider": 1 - mse_noisy  # 提供者指标：用MSE的补数表示
#         },
#         "Trotter Step Zero Shots": {
#             "Incoherent": 0.8 * (1 - r2_noisy) + 0.2 * (1 - noisy_fidelity),
#             "Coherent": 0.8 * noisy_fidelity + 0.2 * r2_noisy,
#             "Provider": 0.5 * (1 - mse_noisy) + 0.5 * noisy_fidelity
#         },
#         "Unseen Obs": {
#             "Incoherent": 1 - r2_mlp,
#             "Coherent": mlp_fidelity,
#             "Provider": 1 - mse_mlp
#         }
#     }
#
#     # 打印指标
#     print("\n=== 场景性能指标 ===")
#     for scenario, scenario_metrics in metrics.items():
#         print(f"\n{scenario}:")
#         for metric_name, value in scenario_metrics.items():
#             print(f"  {metric_name}: {value:.6f}")
#
#     # 绘制概率分布对比图
#     n_states = 2 ** dataset.n_qubits
#     bitstrings = [format(i, f'0{dataset.n_qubits}b') for i in range(n_states)]
#
#     plt.figure(figsize=(12, 6))
#     x = np.arange(n_states)
#     width = 0.25
#
#     plt.bar(x - width, ideal_probs, width, label='理想状态 (Ideal State)', alpha=0.8, color='blue')
#     plt.bar(x, noisy_probs, width, label='含噪声状态 (Noisy State)', alpha=0.8, color='red')
#     plt.bar(x + width, mlp_probs, width, label='MLP去噪后 (Enhanced Mitigation)', alpha=0.8, color='green')
#
#     plt.xlabel('比特串状态 (Bit String States)')
#     plt.ylabel('概率 (Probability)')
#     plt.title('量子态概率分布对比')
#     plt.xticks(x, bitstrings)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#     # 打印保真度对比
#     print("\n=== 保真度对比 ===")
#     print(f"含噪声状态与理想状态的保真度: {noisy_fidelity:.6f}")
#     print(f"MLP去噪后与理想状态的保真度: {mlp_fidelity:.6f}")
#     print(f"保真度提升: {mlp_fidelity - noisy_fidelity:.6f}")
#
#     return metrics
#
#
# def main():
#     # 配置参数
#     n_qubits = 3
#     num_samples = 5000
#     noise_level = 0.05  # 基础噪声水平
#     noise_variation = 0.02  # 噪声变化范围
#
#     # 创建量子数据集
#     print("生成量子数据集...")
#     dataset = QuantumDataset(
#         n_qubits=n_qubits,
#         num_samples=num_samples,
#         noise_level=noise_level,
#         noise_variation=noise_variation
#     )
#
#     # 划分训练集和测试集
#     train_indices, test_indices = train_test_split(
#         np.arange(num_samples),
#         test_size=0.2,
#         random_state=42
#     )
#
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)
#
#     # 创建数据加载器
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # 创建MLP模型
#     input_size = 2 ** n_qubits  # 量子态空间维度
#     model = QuantumDenoisingMLP(
#         input_size=input_size,
#         hidden_sizes=[128, 256, 128]
#     )
#
#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # 训练模型
#     print("\n开始训练MLP去噪模型...")
#     trained_model = train_model(
#         model,
#         train_loader,
#         test_loader,
#         criterion,
#         optimizer,
#         num_epochs=50
#     )
#
#     # 评估模型性能
#     print("\n评估模型性能...")
#     metrics = evaluate_performance(trained_model, dataset, test_loader, dataset.scaler)
#
#     return trained_model, dataset, metrics
#
#
# if __name__ == "__main__":
#     main()
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
#
# # 量子相关库
# from qiskit import QuantumCircuit
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error
# from qiskit.quantum_info import DensityMatrix, state_fidelity, trace_distance
# from qiskit import transpile
# import warnings
#
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
# # 设置中文字体
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
#
#
# class QuantumDataset(Dataset):
#     """量子数据集类，用于生成和存储带噪声的量子态数据及对应的理想态数据"""
#
#     def __init__(self, n_qubits, num_samples=1000, noise_level=0.01, noise_variation=0.005):
#         self.n_qubits = n_qubits
#         self.num_samples = num_samples
#         self.noise_level = noise_level
#         self.noise_variation = noise_variation
#
#         # 生成目标密度矩阵
#         self.target_dm = self.build_target_density()
#         self.ideal_probs = np.real(np.diag(self.target_dm.data))
#
#         # 生成带噪声的样本
#         self.noisy_samples, self.ideal_samples = self.generate_noisy_samples()
#
#         # 移除MinMaxScaler，保留原始概率分布特性
#
#     def build_target_density(self):
#         """构建目标量子电路并返回其密度矩阵"""
#         n_qubits = self.n_qubits
#         qc = QuantumCircuit(n_qubits)
#
#         # 目标电路实现
#         qc.h(0)
#         qc.rz(0.027318, 0)
#         qc.h(1)
#         qc.rz(0.81954, 1)
#         qc.h(2)
#         qc.rz(0.068295, 2)
#
#         qc.cx(1, 2)
#         qc.rz(0.647, 2)
#         qc.cx(1, 2)
#
#         qc.cx(0, 2)
#         qc.rz(0.021567, 2)
#         qc.cx(0, 2)
#
#         qc.cx(0, 1)
#         qc.rz(0.2588, 1)
#         qc.cx(0, 1)
#
#         qc.rx(-0.98987, 0)
#         qc.rx(-0.98987, 1)
#         qc.rx(-0.98987, 2)
#
#         qc.save_density_matrix()
#
#         sim = AerSimulator(method='density_matrix')
#         result = sim.run(qc).result()
#         return DensityMatrix(result.data(0)['density_matrix'])
#
#     def create_noise_model(self, noise_scale=1.0):
#         """创建带有去极化噪声的噪声模型"""
#         noise_model = NoiseModel()
#         effective_noise = self.noise_level * noise_scale
#
#         # 单量子位门噪声
#         single_qubit_gates = ['h', 'rz', 'rx']
#         for gate in single_qubit_gates:
#             for qubit in range(self.n_qubits):
#                 noise_model.add_quantum_error(
#                     depolarizing_error(effective_noise, 1),
#                     gate,
#                     [qubit]
#                 )
#
#         # 双量子位门噪声（通常双量子位门噪声更大）
#         two_qubit_error = depolarizing_error(effective_noise * 1.5, 2)
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])
#
#         return noise_model
#
#     def generate_noisy_samples(self):
#         """生成带噪声的量子态样本集"""
#         n_states = 2 ** self.n_qubits
#         noisy_samples = []
#         ideal_samples = []
#
#         # 理想状态的概率分布（用于所有样本的目标）
#         ideal_probs = np.real(np.diag(self.target_dm.data))
#
#         for _ in range(self.num_samples):
#             # 随机变化噪声水平，增加样本多样性
#             noise_scale = np.random.uniform(1 - self.noise_variation,
#                                             1 + self.noise_variation)
#
#             # 创建噪声模型
#             noise_model = self.create_noise_model(noise_scale)
#
#             # 构建带噪声的电路
#             qc = QuantumCircuit(self.n_qubits)
#             qc.h(0)
#             qc.rz(0.027318, 0)
#             qc.h(1)
#             qc.rz(0.81954, 1)
#             qc.h(2)
#             qc.rz(0.068295, 2)
#
#             qc.cx(1, 2)
#             qc.rz(0.647, 2)
#             qc.cx(1, 2)
#
#             qc.cx(0, 2)
#             qc.rz(0.021567, 2)
#             qc.cx(0, 2)
#
#             qc.cx(0, 1)
#             qc.rz(0.2588, 1)
#             qc.cx(0, 1)
#
#             qc.rx(-0.98987, 0)
#             qc.rx(-0.98987, 1)
#             qc.rx(-0.98987, 2)
#
#             qc.save_density_matrix()
#
#             # 运行带噪声的模拟
#             sim = AerSimulator(method='density_matrix')
#             result = sim.run(
#                 transpile(qc, sim),
#                 noise_model=noise_model
#             ).result()
#
#             noisy_dm = DensityMatrix(result.data(0)['density_matrix'])
#             noisy_probs = np.real(np.diag(noisy_dm.data))
#
#             noisy_samples.append(noisy_probs)
#             ideal_samples.append(ideal_probs)
#
#         return np.array(noisy_samples), np.array(ideal_samples)
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         # 直接使用原始概率分布，不进行标准化
#         return (torch.tensor(self.noisy_samples[idx], dtype=torch.float32),
#                 torch.tensor(self.ideal_samples[idx], dtype=torch.float32))
#
#
# class QuantumDenoisingMLP(nn.Module):
#     """用于量子态去噪的多层感知器模型"""
#
#     def __init__(self, input_size, hidden_sizes=[64, 128, 64], output_size=None):
#         super(QuantumDenoisingMLP, self).__init__()
#
#         # 如果未指定输出大小，则与输入大小相同
#         if output_size is None:
#             output_size = input_size
#
#         # 构建神经网络层
#         layers = []
#         prev_size = input_size
#
#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(prev_size, hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(hidden_size))
#             layers.append(nn.Dropout(0.2))
#             prev_size = hidden_size
#
#         # 输出层，使用softmax确保输出是有效的概率分布
#         layers.append(nn.Linear(prev_size, output_size))
#         layers.append(nn.Softmax(dim=1))  # 确保输出和为1
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
#     """训练MLP模型"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     train_losses = []
#     val_losses = []
#
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#
#         for noisy, ideal in train_loader:
#             noisy, ideal = noisy.to(device), ideal.to(device)
#
#             # 前向传播
#             outputs = model(noisy)
#             loss = criterion(outputs, ideal)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * noisy.size(0)
#
#         # 计算平均训练损失
#         train_loss /= len(train_loader.dataset)
#         train_losses.append(train_loss)
#
#         # 在验证集上评估
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for noisy, ideal in val_loader:
#                 noisy, ideal = noisy.to(device), ideal.to(device)
#                 outputs = model(noisy)
#                 loss = criterion(outputs, ideal)
#                 val_loss += loss.item() * noisy.size(0)
#
#         val_loss /= len(val_loader.dataset)
#         val_losses.append(val_loss)
#
#         # 每10个epoch打印一次进度
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
#
#     # 绘制训练和验证损失
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='训练损失')
#     plt.plot(val_losses, label='验证损失')
#     plt.title('模型训练过程')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
#
#     return model
#
#
# def evaluate_performance(model, dataset, test_loader):
#     """评估模型性能并计算所需指标"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#
#     all_noisy = []
#     all_ideal = []
#     all_predicted = []
#
#     with torch.no_grad():
#         for noisy, ideal in test_loader:
#             noisy, ideal = noisy.to(device), ideal.to(device)
#             predicted = model(noisy)
#
#             all_noisy.extend(noisy.cpu().numpy())
#             all_ideal.extend(ideal.cpu().numpy())
#             all_predicted.extend(predicted.cpu().numpy())
#
#     all_noisy = np.array(all_noisy)
#     all_ideal = np.array(all_ideal)
#     all_predicted = np.array(all_predicted)
#
#     # 计算整体性能指标
#     mse_noisy = mean_squared_error(all_ideal, all_noisy)
#     mse_mlp = mean_squared_error(all_ideal, all_predicted)
#     r2_noisy = r2_score(all_ideal, all_noisy)
#     r2_mlp = r2_score(all_ideal, all_predicted)
#
#     print(f"\n=== 整体性能指标 ===")
#     print(f"含噪声状态与理想状态的MSE: {mse_noisy:.6f}")
#     print(f"MLP去噪后与理想状态的MSE: {mse_mlp:.6f}")
#     print(f"MSE改善比例: {(mse_noisy - mse_mlp) / mse_noisy:.2%}")
#     print(f"含噪声状态的R²分数: {r2_noisy:.6f}")
#     print(f"MLP去噪后的R²分数: {r2_mlp:.6f}")
#
#     # 生成特定测试样本的可视化结果
#     test_qc = QuantumCircuit(dataset.n_qubits)
#     test_qc.h(0)
#     test_qc.rz(0.027318, 0)
#     test_qc.h(1)
#     test_qc.rz(0.81954, 1)
#     test_qc.h(2)
#     test_qc.rz(0.068295, 2)
#
#     test_qc.cx(1, 2)
#     test_qc.rz(0.647, 2)
#     test_qc.cx(1, 2)
#
#     test_qc.cx(0, 2)
#     test_qc.rz(0.021567, 2)
#     test_qc.cx(0, 2)
#
#     test_qc.cx(0, 1)
#     test_qc.rz(0.2588, 1)
#     test_qc.cx(0, 1)
#
#     test_qc.rx(-0.98987, 0)
#     test_qc.rx(-0.98987, 1)
#     test_qc.rx(-0.98987, 2)
#
#     test_qc.save_density_matrix()
#
#     # 获取带噪声的状态
#     noise_model = dataset.create_noise_model(noise_scale=1.0)
#     sim = AerSimulator(method='density_matrix')
#     result_noisy = sim.run(
#         transpile(test_qc, sim),
#         noise_model=noise_model
#     ).result()
#     noisy_dm = DensityMatrix(result_noisy.data(0)['density_matrix'])
#     noisy_probs = np.real(np.diag(noisy_dm.data))
#
#     # 理想状态
#     ideal_probs = dataset.ideal_probs
#     ideal_dm = dataset.target_dm
#
#     # MLP去噪后的状态
#     with torch.no_grad():
#         mlp_probs = model(torch.tensor([noisy_probs], dtype=torch.float32).to(device))
#         mlp_probs = mlp_probs.cpu().numpy()[0]
#     mlp_dm = DensityMatrix(np.diag(mlp_probs))
#
#     # 计算量子领域标准指标
#     noisy_fidelity = state_fidelity(ideal_dm, noisy_dm)  # 保真度：0-1，越大越相似
#     mlp_fidelity = state_fidelity(ideal_dm, mlp_dm)
#     noisy_trace_dist = trace_distance(ideal_dm, noisy_dm)  # 迹距离：0-1，越大越不同
#     mlp_trace_dist = trace_distance(ideal_dm, mlp_dm)
#
#     # 重新定义合理的场景指标（确保在0-1范围内）
#     metrics = {
#         "Random Size Zero Shots": {
#             "Incoherent": noisy_trace_dist,  # 不一致性：用迹距离（0-1）
#             "Coherent": noisy_fidelity,  # 一致性：用保真度（0-1）
#             "Provider": 1 - mse_noisy if mse_noisy <= 1 else 0  # 提供者指标：0-1
#         },
#         "Trotter Step Zero Shots": {
#             "Incoherent": 0.8 * noisy_trace_dist + 0.2 * (1 - noisy_fidelity),  # 混合不一致性
#             "Coherent": 0.8 * noisy_fidelity + 0.2 * (1 - noisy_trace_dist),  # 混合一致性
#             "Provider": 0.5 * (1 - mse_noisy) + 0.5 * noisy_fidelity if mse_noisy <= 1 else 0.5 * noisy_fidelity
#         },
#         "Unseen Obs": {
#             "Incoherent": mlp_trace_dist,  # 去噪后的不一致性
#             "Coherent": mlp_fidelity,  # 去噪后的一致性
#             "Provider": 1 - mse_mlp if mse_mlp <= 1 else 0  # 去噪后的提供者指标
#         }
#     }
#
#     # 打印指标
#     print("\n=== 场景性能指标 ===")
#     for scenario, scenario_metrics in metrics.items():
#         print(f"\n{scenario}:")
#         for metric_name, value in scenario_metrics.items():
#             print(f"  {metric_name}: {value:.6f}")
#
#     # 绘制概率分布对比图
#     n_states = 2 ** dataset.n_qubits
#     bitstrings = [format(i, f'0{dataset.n_qubits}b') for i in range(n_states)]
#
#     plt.figure(figsize=(12, 6))
#     x = np.arange(n_states)
#     width = 0.25
#
#     plt.bar(x - width, ideal_probs, width, label='理想状态 (Ideal State)', alpha=0.8, color='blue')
#     plt.bar(x, noisy_probs, width, label='含噪声状态 (Noisy State)', alpha=0.8, color='red')
#     plt.bar(x + width, mlp_probs, width, label='MLP去噪后 (Enhanced Mitigation)', alpha=0.8, color='green')
#
#     plt.xlabel('比特串状态 (Bit String States)')
#     plt.ylabel('概率 (Probability)')
#     plt.title('量子态概率分布对比')
#     plt.xticks(x, bitstrings)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#     # 打印保真度对比
#     print("\n=== 保真度对比 ===")
#     print(f"含噪声状态与理想状态的保真度: {noisy_fidelity:.6f}")
#     print(f"MLP去噪后与理想状态的保真度: {mlp_fidelity:.6f}")
#     print(f"保真度提升: {mlp_fidelity - noisy_fidelity:.6f}")
#
#     return metrics
#
#
# def main():
#     # 配置参数
#     n_qubits = 3
#     num_samples = 5000
#     noise_level = 0.03  # 降低基础噪声水平，避免样本质量过差
#     noise_variation = 0.01  # 减小噪声变化范围
#
#     # 创建量子数据集
#     print("生成量子数据集...")
#     dataset = QuantumDataset(
#         n_qubits=n_qubits,
#         num_samples=num_samples,
#         noise_level=noise_level,
#         noise_variation=noise_variation
#     )
#
#     # 划分训练集和测试集
#     train_indices, test_indices = train_test_split(
#         np.arange(num_samples),
#         test_size=0.2,
#         random_state=42
#     )
#
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)
#
#     # 创建数据加载器
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # 创建MLP模型
#     input_size = 2 ** n_qubits  # 量子态空间维度
#     model = QuantumDenoisingMLP(
#         input_size=input_size,
#         hidden_sizes=[128, 256, 128]
#     )
#
#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # 训练模型
#     print("\n开始训练MLP去噪模型...")
#     trained_model = train_model(
#         model,
#         train_loader,
#         test_loader,
#         criterion,
#         optimizer,
#         num_epochs=80  # 增加训练轮次，提高收敛效果
#     )
#
#     # 评估模型性能
#     print("\n评估模型性能...")
#     metrics = evaluate_performance(trained_model, dataset, test_loader)
#
#     return trained_model, dataset, metrics
#
#
# if __name__ == "__main__":
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import time
# from scipy.linalg import sqrtm
# from qiskit.quantum_info import entropy
#
# # 量子相关库
# from qiskit import QuantumCircuit
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error
# # from qiskit.quantum_info import DensityMatrix, state_fidelity, trace_distance
# from qiskit.quantum_info import DensityMatrix, state_fidelity
# from qiskit.quantum_info.operators import Operator
# from qiskit.quantum_info import distance
# from qiskit import transpile
# import warnings
#
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
# # 设置中文字体
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
#
#
# class QuantumDataset(Dataset):
#     """量子数据集类，用于生成和存储带噪声的量子态数据及对应的理想态数据"""
#
#     def __init__(self, n_qubits, num_samples=1000, noise_level=0.01, noise_variation=0.005):
#         self.n_qubits = n_qubits
#         self.num_samples = num_samples
#         self.noise_level = noise_level
#         self.noise_variation = noise_variation
#
#         # 生成目标密度矩阵
#         self.target_dm = self.build_target_density()
#         self.ideal_probs = np.real(np.diag(self.target_dm.data))
#
#         # 生成带噪声的样本
#         self.noisy_samples, self.ideal_samples = self.generate_noisy_samples()
#
#         # 计算门开销
#         self.gate_count = self.calculate_gate_count()
#
#     def calculate_gate_count(self):
#         """计算目标电路的门开销"""
#         qc = QuantumCircuit(self.n_qubits)
#
#         # 目标电路实现
#         qc.h(0)
#         qc.rz(0.027318, 0)
#         qc.h(1)
#         qc.rz(0.81954, 1)
#         qc.h(2)
#         qc.rz(0.068295, 2)
#
#         qc.cx(1, 2)
#         qc.rz(0.647, 2)
#         qc.cx(1, 2)
#
#         qc.cx(0, 2)
#         qc.rz(0.021567, 2)
#         qc.cx(0, 2)
#
#         qc.cx(0, 1)
#         qc.rz(0.2588, 1)
#         qc.cx(0, 1)
#
#         qc.rx(-0.98987, 0)
#         qc.rx(-0.98987, 1)
#         qc.rx(-0.98987, 2)
#
#         # 统计门数量
#         gate_count = {
#             'single_qubit': 0,
#             'two_qubit': 0,
#             'total': 0
#         }
#
#         for instruction in qc.data:
#             gate = instruction.operation
#             if len(instruction.qubits) == 1:
#                 gate_count['single_qubit'] += 1
#             elif len(instruction.qubits) == 2:
#                 gate_count['two_qubit'] += 1
#             gate_count['total'] += 1
#
#         return gate_count
#
#     def build_target_density(self):
#         """构建目标量子电路并返回其密度矩阵"""
#         n_qubits = self.n_qubits
#         qc = QuantumCircuit(n_qubits)
#
#         # 目标电路实现
#         qc.h(0)
#         qc.rz(0.027318, 0)
#         qc.h(1)
#         qc.rz(0.81954, 1)
#         qc.h(2)
#         qc.rz(0.068295, 2)
#
#         qc.cx(1, 2)
#         qc.rz(0.647, 2)
#         qc.cx(1, 2)
#
#         qc.cx(0, 2)
#         qc.rz(0.021567, 2)
#         qc.cx(0, 2)
#
#         qc.cx(0, 1)
#         qc.rz(0.2588, 1)
#         qc.cx(0, 1)
#
#         qc.rx(-0.98987, 0)
#         qc.rx(-0.98987, 1)
#         qc.rx(-0.98987, 2)
#
#         qc.save_density_matrix()
#
#         sim = AerSimulator(method='density_matrix')
#         result = sim.run(qc).result()
#         return DensityMatrix(result.data(0)['density_matrix'])
#
#     def create_noise_model(self, noise_scale=1.0):
#         """创建带有去极化噪声的噪声模型"""
#         noise_model = NoiseModel()
#         effective_noise = self.noise_level * noise_scale
#
#         # 单量子位门噪声
#         single_qubit_gates = ['h', 'rz', 'rx']
#         for gate in single_qubit_gates:
#             for qubit in range(self.n_qubits):
#                 noise_model.add_quantum_error(
#                     depolarizing_error(effective_noise, 1),
#                     gate,
#                     [qubit]
#                 )
#
#         # 双量子位门噪声（通常双量子位门噪声更大）
#         two_qubit_error = depolarizing_error(effective_noise * 1.5, 2)
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])
#
#         return noise_model
#
#     def generate_noisy_samples(self):
#         """生成带噪声的量子态样本集"""
#         n_states = 2 ** self.n_qubits
#         noisy_samples = []
#         ideal_samples = []
#
#         # 理想状态的概率分布（用于所有样本的目标）
#         ideal_probs = np.real(np.diag(self.target_dm.data))
#
#         for _ in range(self.num_samples):
#             # 随机变化噪声水平，增加样本多样性
#             noise_scale = np.random.uniform(1 - self.noise_variation,
#                                             1 + self.noise_variation)
#
#             # 创建噪声模型
#             noise_model = self.create_noise_model(noise_scale)
#
#             # 构建带噪声的电路
#             qc = QuantumCircuit(self.n_qubits)
#             qc.h(0)
#             qc.rz(0.027318, 0)
#             qc.h(1)
#             qc.rz(0.81954, 1)
#             qc.h(2)
#             qc.rz(0.068295, 2)
#
#             qc.cx(1, 2)
#             qc.rz(0.647, 2)
#             qc.cx(1, 2)
#
#             qc.cx(0, 2)
#             qc.rz(0.021567, 2)
#             qc.cx(0, 2)
#
#             qc.cx(0, 1)
#             qc.rz(0.2588, 1)
#             qc.cx(0, 1)
#
#             qc.rx(-0.98987, 0)
#             qc.rx(-0.98987, 1)
#             qc.rx(-0.98987, 2)
#
#             qc.save_density_matrix()
#
#             # 运行带噪声的模拟
#             sim = AerSimulator(method='density_matrix')
#             result = sim.run(
#                 transpile(qc, sim),
#                 noise_model=noise_model
#             ).result()
#
#             noisy_dm = DensityMatrix(result.data(0)['density_matrix'])
#             noisy_probs = np.real(np.diag(noisy_dm.data))
#
#             noisy_samples.append(noisy_probs)
#             ideal_samples.append(ideal_probs)
#
#         return np.array(noisy_samples), np.array(ideal_samples)
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         # 直接使用原始概率分布，不进行标准化
#         return (torch.tensor(self.noisy_samples[idx], dtype=torch.float32),
#                 torch.tensor(self.ideal_samples[idx], dtype=torch.float32))
#
#
# class QuantumDenoisingMLP(nn.Module):
#     """用于量子态去噪的多层感知器模型"""
#
#     def __init__(self, input_size, hidden_sizes=[64, 128, 64], output_size=None):
#         super(QuantumDenoisingMLP, self).__init__()
#
#         # 如果未指定输出大小，则与输入大小相同
#         if output_size is None:
#             output_size = input_size
#
#         # 构建神经网络层
#         layers = []
#         prev_size = input_size
#
#         for hidden_size in hidden_sizes:
#             layers.append(nn.Linear(prev_size, hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.BatchNorm1d(hidden_size))
#             layers.append(nn.Dropout(0.2))
#             prev_size = hidden_size
#
#         # 输出层，使用softmax确保输出是有效的概率分布
#         layers.append(nn.Linear(prev_size, output_size))
#         layers.append(nn.Softmax(dim=1))  # 确保输出和为1
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
#     """训练MLP模型"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     train_losses = []
#     val_losses = []
#
#     # 记录开始时间
#     start_time = time.time()
#
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#
#         for noisy, ideal in train_loader:
#             noisy, ideal = noisy.to(device), ideal.to(device)
#
#             # 前向传播
#             outputs = model(noisy)
#             loss = criterion(outputs, ideal)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * noisy.size(0)
#
#         # 计算平均训练损失
#         train_loss /= len(train_loader.dataset)
#         train_losses.append(train_loss)
#
#         # 在验证集上评估
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for noisy, ideal in val_loader:
#                 noisy, ideal = noisy.to(device), ideal.to(device)
#                 outputs = model(noisy)
#                 loss = criterion(outputs, ideal)
#                 val_loss += loss.item() * noisy.size(0)
#
#         val_loss /= len(val_loader.dataset)
#         val_losses.append(val_loss)
#
#         # 每10个epoch打印一次进度
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
#
#     # 计算总训练时间
#     training_time = time.time() - start_time
#     print(f"总训练时间: {training_time:.2f}秒")
#
#     # 绘制训练和验证损失
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='训练损失')
#     plt.plot(val_losses, label='验证损失')
#     plt.title('模型训练过程')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
#
#     return model, training_time
#
#
# def evaluate_performance(model, dataset, test_loader, training_time):
#     """评估模型性能并计算所需指标"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()
#
#     all_noisy = []
#     all_ideal = []
#     all_predicted = []
#
#     with torch.no_grad():
#         for noisy, ideal in test_loader:
#             noisy, ideal = noisy.to(device), ideal.to(device)
#             predicted = model(noisy)
#
#             all_noisy.extend(noisy.cpu().numpy())
#             all_ideal.extend(ideal.cpu().numpy())
#             all_predicted.extend(predicted.cpu().numpy())
#
#     all_noisy = np.array(all_noisy)
#     all_ideal = np.array(all_ideal)
#     all_predicted = np.array(all_predicted)
#
#     # 计算整体性能指标
#     mse_noisy = mean_squared_error(all_ideal, all_noisy)
#     mse_mlp = mean_squared_error(all_ideal, all_predicted)
#     r2_noisy = r2_score(all_ideal, all_noisy)
#     r2_mlp = r2_score(all_ideal, all_predicted)
#
#     print(f"\n=== 整体性能指标 ===")
#     print(f"含噪声状态与理想状态的MSE: {mse_noisy:.6f}")
#     print(f"MLP去噪后与理想状态的MSE: {mse_mlp:.6f}")
#     print(f"MSE改善比例: {(mse_noisy - mse_mlp) / mse_noisy:.2%}")
#     print(f"含噪声状态的R²分数: {r2_noisy:.6f}")
#     print(f"MLP去噪后的R²分数: {r2_mlp:.6f}")
#
#     # 生成特定测试样本的可视化结果
#     test_qc = QuantumCircuit(dataset.n_qubits)
#     test_qc.h(0)
#     test_qc.rz(0.027318, 0)
#     test_qc.h(1)
#     test_qc.rz(0.81954, 1)
#     test_qc.h(2)
#     test_qc.rz(0.068295, 2)
#
#     test_qc.cx(1, 2)
#     test_qc.rz(0.647, 2)
#     test_qc.cx(1, 2)
#
#     test_qc.cx(0, 2)
#     test_qc.rz(0.021567, 2)
#     test_qc.cx(0, 2)
#
#     test_qc.cx(0, 1)
#     test_qc.rz(0.2588, 1)
#     test_qc.cx(0, 1)
#
#     test_qc.rx(-0.98987, 0)
#     test_qc.rx(-0.98987, 1)
#     test_qc.rx(-0.98987, 2)
#
#     test_qc.save_density_matrix()
#
#     # 获取带噪声的状态
#     noise_model = dataset.create_noise_model(noise_scale=1.0)
#     sim = AerSimulator(method='density_matrix')
#     result_noisy = sim.run(
#         transpile(test_qc, sim),
#         noise_model=noise_model
#     ).result()
#     noisy_dm = DensityMatrix(result_noisy.data(0)['density_matrix'])
#     noisy_probs = np.real(np.diag(noisy_dm.data))
#
#     # 理想状态
#     ideal_probs = dataset.ideal_probs
#     ideal_dm = dataset.target_dm
#
#     # MLP去噪后的状态
#     start_time = time.time()
#     with torch.no_grad():
#         mlp_probs = model(torch.tensor([noisy_probs], dtype=torch.float32).to(device))
#         mlp_probs = mlp_probs.cpu().numpy()[0]
#     inference_time = time.time() - start_time
#
#     mlp_dm = DensityMatrix(np.diag(mlp_probs))
#
#     # 计算量子领域标准指标
#     noisy_fidelity = state_fidelity(ideal_dm, noisy_dm)  # 保真度：0-1，越大越相似
#     mlp_fidelity = state_fidelity(ideal_dm, mlp_dm)
#     noisy_trace_dist = distance.trace_distance(ideal_dm, noisy_dm)  # 迹距离：0-1，越大越不同
#     mlp_trace_dist = distance.trace_distance(ideal_dm, mlp_dm)
#
#     # 重新定义合理的场景指标（确保在0-1范围内）
#     metrics = {
#         "Random Size Zero Shots": {
#             "Incoherent": noisy_trace_dist,  # 不一致性：用迹距离（0-1）
#             "Coherent": noisy_fidelity,  # 一致性：用保真度（0-1）
#             "Provider": 1 - mse_noisy if mse_noisy <= 1 else 0  # 提供者指标：0-1
#         },
#         "Trotter Step Zero Shots": {
#             "Incoherent": 0.8 * noisy_trace_dist + 0.2 * (1 - noisy_fidelity),  # 混合不一致性
#             "Coherent": 0.8 * noisy_fidelity + 0.2 * (1 - noisy_trace_dist),  # 混合一致性
#             "Provider": 0.5 * (1 - mse_noisy) + 0.5 * noisy_fidelity if mse_noisy <= 1 else 0.5 * noisy_fidelity
#         },
#         "Unseen Obs": {
#             "Incoherent": mlp_trace_dist,  # 去噪后的不一致性
#             "Coherent": mlp_fidelity,  # 去噪后的一致性
#             "Provider": 1 - mse_mlp if mse_mlp <= 1 else 0  # 去噪后的提供者指标
#         }
#     }
#
#     # 打印指标
#     print("\n=== 场景性能指标 ===")
#     for scenario, scenario_metrics in metrics.items():
#         print(f"\n{scenario}:")
#         for metric_name, value in scenario_metrics.items():
#             print(f"  {metric_name}: {value:.6f}")
#
#     # 绘制概率分布对比图
#     n_states = 2 ** dataset.n_qubits
#     bitstrings = [format(i, f'0{dataset.n_qubits}b') for i in range(n_states)]
#
#     plt.figure(figsize=(12, 6))
#     x = np.arange(n_states)
#     width = 0.25
#
#     plt.bar(x - width, ideal_probs, width, label='理想状态 (Ideal State)', alpha=0.8, color='blue')
#     plt.bar(x, noisy_probs, width, label='含噪声状态 (Noisy State)', alpha=0.8, color='red')
#     plt.bar(x + width, mlp_probs, width, label='MLP去噪后 (Enhanced Mitigation)', alpha=0.8, color='green')
#
#     plt.xlabel('比特串状态 (Bit String States)')
#     plt.ylabel('概率 (Probability)')
#     plt.title('量子态概率分布对比')
#     plt.xticks(x, bitstrings)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#     # 打印保真度对比
#     print("\n=== 保真度对比 ===")
#     print(f"含噪声状态与理想状态的保真度: {noisy_fidelity:.6f}")
#     print(f"MLP去噪后与理想状态的保真度: {mlp_fidelity:.6f}")
#     print(f"保真度提升: {mlp_fidelity - noisy_fidelity:.6f}")
#
#     # 计算量子态指标
#     def calculate_metrics(dm, ideal_dm):
#         fid = state_fidelity(ideal_dm, dm)
#         trace_dist = distance.trace_distance(ideal_dm, dm)
#
#         # 计算纯度: Tr(ρ^2)
#         purity = np.trace(dm.data @ dm.data).real
#
#         # 计算冯诺依曼熵: S(ρ) = -Tr(ρ ln ρ)
#         von_neumann_entropy = entropy(dm)
#
#         return {
#             'fidelity': fid,
#             'trace_distance': trace_dist,
#             'purity': purity,
#             'entropy': von_neumann_entropy
#         }
#
#     # 计算各状态的指标
#     noisy_metrics = calculate_metrics(noisy_dm, ideal_dm)
#     mlp_metrics = calculate_metrics(mlp_dm, ideal_dm)
#     ideal_metrics = calculate_metrics(ideal_dm, ideal_dm)
#
#     # 打印指标
#     print("\n=== 量子态指标 ===")
#     print(
#         f"理想状态: 保真度={ideal_metrics['fidelity']:.6f}, 纯度={ideal_metrics['purity']:.6f}, 熵={ideal_metrics['entropy']:.6f}")
#     print(
#         f"含噪声状态: 保真度={noisy_metrics['fidelity']:.6f}, 纯度={noisy_metrics['purity']:.6f}, 熵={noisy_metrics['entropy']:.6f}")
#     print(
#         f"MLP去噪后: 保真度={mlp_metrics['fidelity']:.6f}, 纯度={mlp_metrics['purity']:.6f}, 熵={mlp_metrics['entropy']:.6f}")
#
#     # 打印时间开销
#     print("\n=== 时间开销 ===")
#     print(f"训练时间: {training_time:.4f}秒")
#     print(f"单个样本推理时间: {inference_time:.6f}秒")
#
#     # 打印门开销
#     print("\n=== 门开销 ===")
#     gate_count = dataset.gate_count
#     print(f"单量子位门数量: {gate_count['single_qubit']}")
#     print(f"双量子位门数量: {gate_count['two_qubit']}")
#     print(f"总门数量: {gate_count['total']}")
#
#     # 计算不同噪声水平下的保真度
#     print("\n=== 不同噪声水平下的态保真度 ===")
#     noise_levels = [0.05, 0.1, 0.2]
#     fidelity_results = {'noisy': [], 'mlp': []}
#
#     for p in noise_levels:
#         # 创建当前噪声水平的噪声模型
#         noise_model = NoiseModel()
#         single_error = depolarizing_error(p, 1)
#         two_qubit_error = depolarizing_error(p * 1.5, 2)
#
#         for gate in ['h', 'rz', 'rx']:
#             for qubit in range(dataset.n_qubits):
#                 noise_model.add_quantum_error(single_error, gate, [qubit])
#
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])
#
#         # 运行带噪声的模拟
#         sim = AerSimulator(method='density_matrix')
#         result_noisy = sim.run(
#             transpile(test_qc, sim),
#             noise_model=noise_model
#         ).result()
#         noisy_dm_p = DensityMatrix(result_noisy.data(0)['density_matrix'])
#
#         # 计算MLP去噪后的状态
#         noisy_probs_p = np.real(np.diag(noisy_dm_p.data))
#         with torch.no_grad():
#             mlp_probs_p = model(torch.tensor([noisy_probs_p], dtype=torch.float32).to(device))
#             mlp_probs_p = mlp_probs_p.cpu().numpy()[0]
#         mlp_dm_p = DensityMatrix(np.diag(mlp_probs_p))
#
#         # 计算保真度
#         noisy_fidelity = state_fidelity(ideal_dm, noisy_dm_p)
#         mlp_fidelity = state_fidelity(ideal_dm, mlp_dm_p)
#
#         fidelity_results['noisy'].append(noisy_fidelity)
#         fidelity_results['mlp'].append(mlp_fidelity)
#
#         print(f"噪声水平 p={p}:")
#         print(f"  含噪声保真度 = {noisy_fidelity:.6f}")
#         print(f"  MLP去噪后保真度 = {mlp_fidelity:.6f}")
#         print(f"  保真度提升 = {mlp_fidelity - noisy_fidelity:.6f}")
#
#     # 鲁棒性评估 (p=0.1)
#     print("\n=== 鲁棒性评估 (p=0.1) ===")
#     p = 0.1
#     num_robustness_tests = 10
#     robustness_results = []
#
#     for i in range(num_robustness_tests):
#         # 创建带变化的噪声模型
#         varied_p = p * np.random.uniform(0.8, 1.2)  # ±20%变化
#         noise_model = NoiseModel()
#         single_error = depolarizing_error(varied_p, 1)
#         two_qubit_error = depolarizing_error(varied_p * 1.5, 2)
#
#         for gate in ['h', 'rz', 'rx']:
#             for qubit in range(dataset.n_qubits):
#                 noise_model.add_quantum_error(single_error, gate, [qubit])
#
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
#         noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])
#
#         # 运行带噪声的模拟
#         sim = AerSimulator(method='density_matrix')
#         result_noisy = sim.run(
#             transpile(test_qc, sim),
#             noise_model=noise_model
#         ).result()
#         noisy_dm_var = DensityMatrix(result_noisy.data(0)['density_matrix'])
#
#         # 计算MLP去噪后的状态
#         noisy_probs_var = np.real(np.diag(noisy_dm_var.data))
#         with torch.no_grad():
#             mlp_probs_var = model(torch.tensor([noisy_probs_var], dtype=torch.float32).to(device))
#             mlp_probs_var = mlp_probs_var.cpu().numpy()[0]
#         mlp_dm_var = DensityMatrix(np.diag(mlp_probs_var))
#
#         # 计算保真度
#         noisy_fidelity = state_fidelity(ideal_dm, noisy_dm_var)
#         mlp_fidelity = state_fidelity(ideal_dm, mlp_dm_var)
#
#         robustness_results.append({
#             'noise_level': varied_p,
#             'noisy_fidelity': noisy_fidelity,
#             'mlp_fidelity': mlp_fidelity,
#             'improvement': mlp_fidelity - noisy_fidelity
#         })
#
#     # 计算鲁棒性指标
#     noisy_fids = [r['noisy_fidelity'] for r in robustness_results]
#     mlp_fids = [r['mlp_fidelity'] for r in robustness_results]
#     improvements = [r['improvement'] for r in robustness_results]
#
#     print(f"平均含噪声保真度: {np.mean(noisy_fids):.6f} ± {np.std(noisy_fids):.6f}")
#     print(f"平均去噪后保真度: {np.mean(mlp_fids):.6f} ± {np.std(mlp_fids):.6f}")
#     print(f"平均保真度提升: {np.mean(improvements):.6f} ± {np.std(improvements):.6f}")
#
#     # 返回所有指标结果
#     metrics = {
#         # ... [原有指标] ...
#         'gate_count': gate_count,
#         'training_time': training_time,
#         'inference_time': inference_time,
#         'noise_level_fidelity': {
#             'levels': noise_levels,
#             'noisy': fidelity_results['noisy'],
#             'mlp': fidelity_results['mlp']
#         },
#         'robustness': robustness_results,
#         'purity': {
#             'ideal': ideal_metrics['purity'],
#             'noisy': noisy_metrics['purity'],
#             'mlp': mlp_metrics['purity']
#         },
#         'entropy': {
#             'ideal': ideal_metrics['entropy'],
#             'noisy': noisy_metrics['entropy'],
#             'mlp': mlp_metrics['entropy']
#         }
#     }
#
#     return metrics
#
#
# def main():
#     # 配置参数
#     n_qubits = 3
#     num_samples = 5000
#     noise_level = 0.03  # 降低基础噪声水平，避免样本质量过差
#     noise_variation = 0.01  # 减小噪声变化范围
#
#     # 创建量子数据集
#     print("生成量子数据集...")
#     dataset = QuantumDataset(
#         n_qubits=n_qubits,
#         num_samples=num_samples,
#         noise_level=noise_level,
#         noise_variation=noise_variation
#     )
#
#     # 划分训练集和测试集
#     train_indices, test_indices = train_test_split(
#         np.arange(num_samples),
#         test_size=0.2,
#         random_state=42
#     )
#
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)
#
#     # 创建数据加载器
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # 创建MLP模型
#     input_size = 2 ** n_qubits  # 量子态空间维度
#     model = QuantumDenoisingMLP(
#         input_size=input_size,
#         hidden_sizes=[128, 256, 128]
#     )
#
#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # 训练模型
#     print("\n开始训练MLP去噪模型...")
#     trained_model, training_time = train_model(
#         model,
#         train_loader,
#         test_loader,
#         criterion,
#         optimizer,
#         num_epochs=80  # 增加训练轮次，提高收敛效果
#     )
#
#     # 评估模型性能 (传入训练时间)
#     print("\n评估模型性能...")
#     metrics = evaluate_performance(trained_model, dataset, test_loader, training_time)
#
#     # 打印最终汇总
#     print("\n=== 最终汇总指标 ===")
#     print(
#         f"量子电路门开销: {metrics['gate_count']['total']} (单量子位: {metrics['gate_count']['single_qubit']}, 双量子位: {metrics['gate_count']['two_qubit']})")
#     print(f"训练时间: {metrics['training_time']:.2f}秒")
#     print(f"单个样本推理时间: {metrics['inference_time']:.6f}秒")
#
#     print("\n不同噪声水平下的态保真度:")
#     for i, p in enumerate(metrics['noise_level_fidelity']['levels']):
#         print(
#             f"p={p:.2f}: 噪声态={metrics['noise_level_fidelity']['noisy'][i]:.6f}, MLP去噪={metrics['noise_level_fidelity']['mlp'][i]:.6f}")
#
#     print("\n鲁棒性 (p=0.1 ±20%):")
#     noisy_fids = [r['noisy_fidelity'] for r in metrics['robustness']]
#     mlp_fids = [r['mlp_fidelity'] for r in metrics['robustness']]
#     print(f"噪声态保真度: {np.mean(noisy_fids):.6f} ± {np.std(noisy_fids):.6f}")
#     print(f"去噪后保真度: {np.mean(mlp_fids):.6f} ± {np.std(mlp_fids):.6f}")
#
#     print("\n量子态纯度:")
#     print(f"理想态: {metrics['purity']['ideal']:.6f}")
#     print(f"噪声态: {metrics['purity']['noisy']:.6f}")
#     print(f"去噪后: {metrics['purity']['mlp']:.6f}")
#
#     print("\n冯诺依曼熵:")
#     print(f"理想态: {metrics['entropy']['ideal']:.6f}")
#     print(f"噪声态: {metrics['entropy']['noisy']:.6f}")
#     print(f"去噪后: {metrics['entropy']['mlp']:.6f}")
#
#     return trained_model, dataset, metrics
#
#
# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from scipy.linalg import sqrtm

# 量子相关库
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, state_fidelity, entropy
from qiskit import transpile
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 手动实现迹距离计算
def trace_distance(rho1, rho2):
    """计算两个密度矩阵之间的迹距离"""
    return 0.5 * np.linalg.norm(rho1.data - rho2.data, 'nuc')


class QuantumDataset(Dataset):
    """量子数据集类，用于生成和存储带噪声的量子态数据及对应的理想态数据"""

    def __init__(self, n_qubits, num_samples=1000, noise_level=0.01, noise_variation=0.005):
        self.n_qubits = n_qubits
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.noise_variation = noise_variation

        # 生成目标密度矩阵
        self.target_dm = self.build_target_density()
        self.ideal_probs = np.real(np.diag(self.target_dm.data))

        # 生成带噪声的样本
        self.noisy_samples, self.ideal_samples = self.generate_noisy_samples()

        # 计算门开销
        self.gate_count = self.calculate_gate_count()

    def calculate_gate_count(self):
        """计算目标电路的门开销"""
        qc = QuantumCircuit(self.n_qubits)

        # 目标电路实现
        qc.h(0)
        qc.rz(0.027318, 0)
        qc.h(1)
        qc.rz(0.81954, 1)
        qc.h(2)
        qc.rz(0.068295, 2)

        qc.cx(1, 2)
        qc.rz(0.647, 2)
        qc.cx(1, 2)

        qc.cx(0, 2)
        qc.rz(0.021567, 2)
        qc.cx(0, 2)

        qc.cx(0, 1)
        qc.rz(0.2588, 1)
        qc.cx(0, 1)

        qc.rx(-0.98987, 0)
        qc.rx(-0.98987, 1)
        qc.rx(-0.98987, 2)

        # 统计门数量
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

    def build_target_density(self):
        """构建目标量子电路并返回其密度矩阵"""
        n_qubits = self.n_qubits
        qc = QuantumCircuit(n_qubits)

        # 目标电路实现
        qc.h(0)
        qc.rz(0.027318, 0)
        qc.h(1)
        qc.rz(0.81954, 1)
        qc.h(2)
        qc.rz(0.068295, 2)

        qc.cx(1, 2)
        qc.rz(0.647, 2)
        qc.cx(1, 2)

        qc.cx(0, 2)
        qc.rz(0.021567, 2)
        qc.cx(0, 2)

        qc.cx(0, 1)
        qc.rz(0.2588, 1)
        qc.cx(0, 1)

        qc.rx(-0.98987, 0)
        qc.rx(-0.98987, 1)
        qc.rx(-0.98987, 2)

        qc.save_density_matrix()

        sim = AerSimulator(method='density_matrix')
        result = sim.run(qc).result()
        return DensityMatrix(result.data(0)['density_matrix'])

    def create_noise_model(self, noise_scale=1.0):
        """创建带有去极化噪声的噪声模型"""
        noise_model = NoiseModel()
        effective_noise = self.noise_level * noise_scale

        # 单量子位门噪声
        single_qubit_gates = ['h', 'rz', 'rx']
        for gate in single_qubit_gates:
            for qubit in range(self.n_qubits):
                noise_model.add_quantum_error(
                    depolarizing_error(effective_noise, 1),
                    gate,
                    [qubit]
                )

        # 双量子位门噪声（通常双量子位门噪声更大）
        two_qubit_error = depolarizing_error(effective_noise * 1.5, 2)
        noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
        noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
        noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])

        return noise_model

    def generate_noisy_samples(self):
        """生成带噪声的量子态样本集"""
        n_states = 2 ** self.n_qubits
        noisy_samples = []
        ideal_samples = []

        # 理想状态的概率分布（用于所有样本的目标）
        ideal_probs = np.real(np.diag(self.target_dm.data))

        for _ in range(self.num_samples):
            # 随机变化噪声水平，增加样本多样性
            noise_scale = np.random.uniform(1 - self.noise_variation,
                                            1 + self.noise_variation)

            # 创建噪声模型
            noise_model = self.create_noise_model(noise_scale)

            # 构建带噪声的电路
            qc = QuantumCircuit(self.n_qubits)
            qc.h(0)
            qc.rz(0.027318, 0)
            qc.h(1)
            qc.rz(0.81954, 1)
            qc.h(2)
            qc.rz(0.068295, 2)

            qc.cx(1, 2)
            qc.rz(0.647, 2)
            qc.cx(1, 2)

            qc.cx(0, 2)
            qc.rz(0.021567, 2)
            qc.cx(0, 2)

            qc.cx(0, 1)
            qc.rz(0.2588, 1)
            qc.cx(0, 1)

            qc.rx(-0.98987, 0)
            qc.rx(-0.98987, 1)
            qc.rx(-0.98987, 2)

            qc.save_density_matrix()

            # 运行带噪声的模拟
            sim = AerSimulator(method='density_matrix')
            result = sim.run(
                transpile(qc, sim),
                noise_model=noise_model
            ).result()

            noisy_dm = DensityMatrix(result.data(0)['density_matrix'])
            noisy_probs = np.real(np.diag(noisy_dm.data))

            noisy_samples.append(noisy_probs)
            ideal_samples.append(ideal_probs)

        return np.array(noisy_samples), np.array(ideal_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 直接使用原始概率分布，不进行标准化
        return (torch.tensor(self.noisy_samples[idx], dtype=torch.float32),
                torch.tensor(self.ideal_samples[idx], dtype=torch.float32))


class QuantumDenoisingMLP(nn.Module):
    """用于量子态去噪的多层感知器模型"""

    def __init__(self, input_size, hidden_sizes=[64, 128, 64], output_size=None):
        super(QuantumDenoisingMLP, self).__init__()

        # 如果未指定输出大小，则与输入大小相同
        if output_size is None:
            output_size = input_size

        # 构建神经网络层
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # 输出层，使用softmax确保输出是有效的概率分布
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))  # 确保输出和为1

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    """训练MLP模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    val_losses = []

    # 记录开始时间
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for noisy, ideal in train_loader:
            noisy, ideal = noisy.to(device), ideal.to(device)

            # 前向传播
            outputs = model(noisy)
            loss = criterion(outputs, ideal)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 在验证集上评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, ideal in val_loader:
                noisy, ideal = noisy.to(device), ideal.to(device)
                outputs = model(noisy)
                loss = criterion(outputs, ideal)
                val_loss += loss.item() * noisy.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 计算总训练时间
    training_time = time.time() - start_time
    print(f"总训练时间: {training_time:.2f}秒")

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('模型训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model, training_time


def evaluate_performance(model, dataset, test_loader, training_time):
    """评估模型性能并计算所需指标"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_noisy = []
    all_ideal = []
    all_predicted = []

    with torch.no_grad():
        for noisy, ideal in test_loader:
            noisy, ideal = noisy.to(device), ideal.to(device)
            predicted = model(noisy)

            all_noisy.extend(noisy.cpu().numpy())
            all_ideal.extend(ideal.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    all_noisy = np.array(all_noisy)
    all_ideal = np.array(all_ideal)
    all_predicted = np.array(all_predicted)

    # 计算整体性能指标
    mse_noisy = mean_squared_error(all_ideal, all_noisy)
    mse_mlp = mean_squared_error(all_ideal, all_predicted)
    r2_noisy = r2_score(all_ideal, all_noisy)
    r2_mlp = r2_score(all_ideal, all_predicted)

    print(f"\n=== 整体性能指标 ===")
    print(f"含噪声状态与理想状态的MSE: {mse_noisy:.6f}")
    print(f"MLP去噪后与理想状态的MSE: {mse_mlp:.6f}")
    print(f"MSE改善比例: {(mse_noisy - mse_mlp) / mse_noisy:.2%}")
    print(f"含噪声状态的R²分数: {r2_noisy:.6f}")
    print(f"MLP去噪后的R²分数: {r2_mlp:.6f}")

    # 生成特定测试样本的可视化结果
    test_qc = QuantumCircuit(dataset.n_qubits)
    test_qc.h(0)
    test_qc.rz(0.027318, 0)
    test_qc.h(1)
    test_qc.rz(0.81954, 1)
    test_qc.h(2)
    test_qc.rz(0.068295, 2)

    test_qc.cx(1, 2)
    test_qc.rz(0.647, 2)
    test_qc.cx(1, 2)

    test_qc.cx(0, 2)
    test_qc.rz(0.021567, 2)
    test_qc.cx(0, 2)

    test_qc.cx(0, 1)
    test_qc.rz(0.2588, 1)
    test_qc.cx(0, 1)

    test_qc.rx(-0.98987, 0)
    test_qc.rx(-0.98987, 1)
    test_qc.rx(-0.98987, 2)

    test_qc.save_density_matrix()

    # 获取带噪声的状态
    noise_model = dataset.create_noise_model(noise_scale=1.0)
    sim = AerSimulator(method='density_matrix')
    result_noisy = sim.run(
        transpile(test_qc, sim),
        noise_model=noise_model
    ).result()
    noisy_dm = DensityMatrix(result_noisy.data(0)['density_matrix'])
    noisy_probs = np.real(np.diag(noisy_dm.data))

    # 理想状态
    ideal_probs = dataset.ideal_probs
    ideal_dm = dataset.target_dm

    # MLP去噪后的状态
    start_time = time.time()
    with torch.no_grad():
        mlp_probs = model(torch.tensor([noisy_probs], dtype=torch.float32).to(device))
        mlp_probs = mlp_probs.cpu().numpy()[0]
    inference_time = time.time() - start_time

    mlp_dm = DensityMatrix(np.diag(mlp_probs))

    # 计算量子领域标准指标
    noisy_fidelity = state_fidelity(ideal_dm, noisy_dm)  # 保真度：0-1，越大越相似
    mlp_fidelity = state_fidelity(ideal_dm, mlp_dm)
    noisy_trace_dist = trace_distance(ideal_dm, noisy_dm)  # 迹距离：0-1，越大越不同
    mlp_trace_dist = trace_distance(ideal_dm, mlp_dm)

    # 重新定义合理的场景指标（确保在0-1范围内）
    metrics = {
        "Random Size Zero Shots": {
            "Incoherent": noisy_trace_dist,  # 不一致性：用迹距离（0-1）
            "Coherent": noisy_fidelity,  # 一致性：用保真度（0-1）
            "Provider": 1 - mse_noisy if mse_noisy <= 1 else 0  # 提供者指标：0-1
        },
        "Trotter Step Zero Shots": {
            "Incoherent": 0.8 * noisy_trace_dist + 0.2 * (1 - noisy_fidelity),  # 混合不一致性
            "Coherent": 0.8 * noisy_fidelity + 0.2 * (1 - noisy_trace_dist),  # 混合一致性
            "Provider": 0.5 * (1 - mse_noisy) + 0.5 * noisy_fidelity if mse_noisy <= 1 else 0.5 * noisy_fidelity
        },
        "Unseen Obs": {
            "Incoherent": mlp_trace_dist,  # 去噪后的不一致性
            "Coherent": mlp_fidelity,  # 去噪后的一致性
            "Provider": 1 - mse_mlp if mse_mlp <= 1 else 0  # 去噪后的提供者指标
        }
    }

    # 打印指标
    print("\n=== 场景性能指标 ===")
    for scenario, scenario_metrics in metrics.items():
        print(f"\n{scenario}:")
        for metric_name, value in scenario_metrics.items():
            print(f"  {metric_name}: {value:.6f}")

    # 绘制概率分布对比图
    n_states = 2 ** dataset.n_qubits
    bitstrings = [format(i, f'0{dataset.n_qubits}b') for i in range(n_states)]

    plt.figure(figsize=(12, 6))
    x = np.arange(n_states)
    width = 0.25

    plt.bar(x - width, ideal_probs, width, label='理想状态 (Ideal State)', alpha=0.8, color='blue')
    plt.bar(x, noisy_probs, width, label='含噪声状态 (Noisy State)', alpha=0.8, color='red')
    plt.bar(x + width, mlp_probs, width, label='MLP去噪后 (Enhanced Mitigation)', alpha=0.8, color='green')

    plt.xlabel('比特串状态 (Bit String States)')
    plt.ylabel('概率 (Probability)')
    plt.title('量子态概率分布对比')
    plt.xticks(x, bitstrings)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印保真度对比
    print("\n=== 保真度对比 ===")
    print(f"含噪声状态与理想状态的保真度: {noisy_fidelity:.6f}")
    print(f"MLP去噪后与理想状态的保真度: {mlp_fidelity:.6f}")
    print(f"保真度提升: {mlp_fidelity - noisy_fidelity:.6f}")

    # 计算量子态指标
    def calculate_metrics(dm, ideal_dm):
        fid = state_fidelity(ideal_dm, dm)
        trace_dist = trace_distance(ideal_dm, dm)

        # 计算纯度: Tr(ρ^2)
        purity = np.trace(dm.data @ dm.data).real

        # 计算冯诺依曼熵: S(ρ) = -Tr(ρ ln ρ)
        von_neumann_entropy = entropy(dm)

        return {
            'fidelity': fid,
            'trace_distance': trace_dist,
            'purity': purity,
            'entropy': von_neumann_entropy
        }

    # 计算各状态的指标
    noisy_metrics = calculate_metrics(noisy_dm, ideal_dm)
    mlp_metrics = calculate_metrics(mlp_dm, ideal_dm)
    ideal_metrics = calculate_metrics(ideal_dm, ideal_dm)

    # 打印指标
    print("\n=== 量子态指标 ===")
    print(
        f"理想状态: 保真度={ideal_metrics['fidelity']:.6f}, 纯度={ideal_metrics['purity']:.6f}, 熵={ideal_metrics['entropy']:.6f}")
    print(
        f"含噪声状态: 保真度={noisy_metrics['fidelity']:.6f}, 纯度={noisy_metrics['purity']:.6f}, 熵={noisy_metrics['entropy']:.6f}")
    print(
        f"MLP去噪后: 保真度={mlp_metrics['fidelity']:.6f}, 纯度={mlp_metrics['purity']:.6f}, 熵={mlp_metrics['entropy']:.6f}")

    # 打印时间开销
    print("\n=== 时间开销 ===")
    print(f"训练时间: {training_time:.4f}秒")
    print(f"单个样本推理时间: {inference_time:.6f}秒")

    # 打印门开销
    print("\n=== 门开销 ===")
    gate_count = dataset.gate_count
    print(f"单量子位门数量: {gate_count['single_qubit']}")
    print(f"双量子位门数量: {gate_count['two_qubit']}")
    print(f"总门数量: {gate_count['total']}")

    # 计算不同噪声水平下的保真度
    print("\n=== 不同噪声水平下的态保真度 ===")
    noise_levels = [0.08, 0.15, 0.18]
    fidelity_results = {'noisy': [], 'mlp': []}

    for p in noise_levels:
        # 创建当前噪声水平的噪声模型
        noise_model = NoiseModel()
        single_error = depolarizing_error(p, 1)
        two_qubit_error = depolarizing_error(p * 1.5, 2)

        for gate in ['h', 'rz', 'rx']:
            for qubit in range(dataset.n_qubits):
                noise_model.add_quantum_error(single_error, gate, [qubit])

        noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
        noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
        noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])

        # 运行带噪声的模拟
        sim = AerSimulator(method='density_matrix')
        result_noisy = sim.run(
            transpile(test_qc, sim),
            noise_model=noise_model
        ).result()
        noisy_dm_p = DensityMatrix(result_noisy.data(0)['density_matrix'])

        # 计算MLP去噪后的状态
        noisy_probs_p = np.real(np.diag(noisy_dm_p.data))
        with torch.no_grad():
            mlp_probs_p = model(torch.tensor([noisy_probs_p], dtype=torch.float32).to(device))
            mlp_probs_p = mlp_probs_p.cpu().numpy()[0]
        mlp_dm_p = DensityMatrix(np.diag(mlp_probs_p))

        # 计算保真度
        noisy_fidelity = state_fidelity(ideal_dm, noisy_dm_p)
        mlp_fidelity = state_fidelity(ideal_dm, mlp_dm_p)

        fidelity_results['noisy'].append(noisy_fidelity)
        fidelity_results['mlp'].append(mlp_fidelity)

        print(f"噪声水平 p={p}:")
        print(f"  含噪声保真度 = {noisy_fidelity:.6f}")
        print(f"  MLP去噪后保真度 = {mlp_fidelity:.6f}")
        print(f"  保真度提升 = {mlp_fidelity - noisy_fidelity:.6f}")

    # 鲁棒性评估 (p=0.1)
    print("\n=== 鲁棒性评估 (p=0.1) ===")
    p = 0.1
    num_robustness_tests = 10
    robustness_results = []

    for i in range(num_robustness_tests):
        # 创建带变化的噪声模型
        varied_p = p * np.random.uniform(0.8, 1.2)  # ±20%变化
        noise_model = NoiseModel()
        single_error = depolarizing_error(varied_p, 1)
        two_qubit_error = depolarizing_error(varied_p * 1.5, 2)

        for gate in ['h', 'rz', 'rx']:
            for qubit in range(dataset.n_qubits):
                noise_model.add_quantum_error(single_error, gate, [qubit])

        noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 1])
        noise_model.add_quantum_error(two_qubit_error, 'cx', [1, 2])
        noise_model.add_quantum_error(two_qubit_error, 'cx', [0, 2])

        # 运行带噪声的模拟
        sim = AerSimulator(method='density_matrix')
        result_noisy = sim.run(
            transpile(test_qc, sim),
            noise_model=noise_model
        ).result()
        noisy_dm_var = DensityMatrix(result_noisy.data(0)['density_matrix'])

        # 计算MLP去噪后的状态
        noisy_probs_var = np.real(np.diag(noisy_dm_var.data))
        with torch.no_grad():
            mlp_probs_var = model(torch.tensor([noisy_probs_var], dtype=torch.float32).to(device))
            mlp_probs_var = mlp_probs_var.cpu().numpy()[0]
        mlp_dm_var = DensityMatrix(np.diag(mlp_probs_var))

        # 计算保真度
        noisy_fidelity = state_fidelity(ideal_dm, noisy_dm_var)
        mlp_fidelity = state_fidelity(ideal_dm, mlp_dm_var)

        robustness_results.append({
            'noise_level': varied_p,
            'noisy_fidelity': noisy_fidelity,
            'mlp_fidelity': mlp_fidelity,
            'improvement': mlp_fidelity - noisy_fidelity
        })

    # 计算鲁棒性指标
    noisy_fids = [r['noisy_fidelity'] for r in robustness_results]
    mlp_fids = [r['mlp_fidelity'] for r in robustness_results]
    improvements = [r['improvement'] for r in robustness_results]

    print(f"平均含噪声保真度: {np.mean(noisy_fids):.6f} ± {np.std(noisy_fids):.6f}")
    print(f"平均去噪后保真度: {np.mean(mlp_fids):.6f} ± {np.std(mlp_fids):.6f}")
    print(f"平均保真度提升: {np.mean(improvements):.6f} ± {np.std(improvements):.6f}")

    # 返回所有指标结果
    metrics = {
        'gate_count': gate_count,
        'training_time': training_time,
        'inference_time': inference_time,
        'noise_level_fidelity': {
            'levels': noise_levels,
            'noisy': fidelity_results['noisy'],
            'mlp': fidelity_results['mlp']
        },
        'robustness': robustness_results,
        'purity': {
            'ideal': ideal_metrics['purity'],
            'noisy': noisy_metrics['purity'],
            'mlp': mlp_metrics['purity']
        },
        'entropy': {
            'ideal': ideal_metrics['entropy'],
            'noisy': noisy_metrics['entropy'],
            'mlp': mlp_metrics['entropy']
        }
    }

    return metrics


def main():
    # 配置参数
    n_qubits = 3
    num_samples = 5000
    noise_level = 0.03  # 降低基础噪声水平，避免样本质量过差
    noise_variation = 0.01  # 减小噪声变化范围

    # 创建量子数据集
    print("生成量子数据集...")
    dataset = QuantumDataset(
        n_qubits=n_qubits,
        num_samples=num_samples,
        noise_level=noise_level,
        noise_variation=noise_variation
    )

    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        np.arange(num_samples),
        test_size=0.2,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建MLP模型
    input_size = 2 ** n_qubits  # 量子态空间维度
    model = QuantumDenoisingMLP(
        input_size=input_size,
        hidden_sizes=[128, 256, 128]
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("\n开始训练MLP去噪模型...")
    trained_model, training_time = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=80  # 增加训练轮次，提高收敛效果
    )

    # 评估模型性能 (传入训练时间)
    print("\n评估模型性能...")
    metrics = evaluate_performance(trained_model, dataset, test_loader, training_time)

    # 打印最终汇总
    print("\n=== 最终汇总指标 ===")
    print(
        f"量子电路门开销: {metrics['gate_count']['total']} (单量子位: {metrics['gate_count']['single_qubit']}, 双量子位: {metrics['gate_count']['two_qubit']})")
    print(f"训练时间: {metrics['training_time']:.2f}秒")
    print(f"单个样本推理时间: {metrics['inference_time']:.6f}秒")

    print("\n不同噪声水平下的态保真度:")
    for i, p in enumerate(metrics['noise_level_fidelity']['levels']):
        print(
            f"p={p:.2f}: 噪声态={metrics['noise_level_fidelity']['noisy'][i]:.6f}, MLP去噪={metrics['noise_level_fidelity']['mlp'][i]:.6f}")

    print("\n鲁棒性 (p=0.1 ±20%):")
    noisy_fids = [r['noisy_fidelity'] for r in metrics['robustness']]
    mlp_fids = [r['mlp_fidelity'] for r in metrics['robustness']]
    print(f"噪声态保真度: {np.mean(noisy_fids):.6f} ± {np.std(noisy_fids):.6f}")
    print(f"去噪后保真度: {np.mean(mlp_fids):.6f} ± {np.std(mlp_fids):.6f}")

    print("\n量子态纯度:")
    print(f"理想态: {metrics['purity']['ideal']:.6f}")
    print(f"噪声态: {metrics['purity']['noisy']:.6f}")
    print(f"去噪后: {metrics['purity']['mlp']:.6f}")

    print("\n冯诺依曼熵:")
    print(f"理想态: {metrics['entropy']['ideal']:.6f}")
    print(f"噪声态: {metrics['entropy']['noisy']:.6f}")
    print(f"去噪后: {metrics['entropy']['mlp']:.6f}")

    return trained_model, dataset, metrics


if __name__ == "__main__":
    main()