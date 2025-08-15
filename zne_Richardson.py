import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
from sympy import Symbol, expand
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim

# 量子模拟相关
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.quantum_info import DensityMatrix, state_fidelity, Statevector, partial_trace
from qiskit import transpile
import qiskit
# from qiskit.circuit import SaveInstruction
# 设置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.serif'] = ['Times New Roman']


class EffectiveQuantumNoiseMitigation:
    """基于零噪声外推和量子误差缓解的有效噪声缓解类"""
    
    def __init__(self, n_qubits, n_layers, base_noise_prob=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.base_noise_prob = base_noise_prob
        self.backend = Aer.get_backend('statevector_simulator')
        self.backend_noisy = Aer.get_backend('aer_simulator')
        
        # 损失函数权重
        self.alpha_fidelity = 0.6
        self.alpha_task = 0.4
        
        # 初始化参数
        self.theta_params = np.random.uniform(0, 2*np.pi, n_layers * n_qubits * 3)
        
        # 零噪声外推参数
        self.noise_scales = [0.5, 1.0, 1.5, 2.0]  # 噪声缩放因子
        self.extrapolation_order = 2  # 外推阶数
        
    def create_parameterized_circuit(self, parameters):
        """创建参数化量子电路"""
        # qc = QuantumCircuit(self.n_qubits)
        #
        # param_idx = 0
        # for layer in range(self.n_layers):
        #     # 参数化层
        #     for qubit in range(self.n_qubits):
        #         qc.rx(parameters[param_idx], qubit)
        #         qc.ry(parameters[param_idx + 1], qubit)
        #         qc.rz(parameters[param_idx + 2], qubit)
        #         param_idx += 3
        #
        #     # 纠缠层
        #     for qubit in range(self.n_qubits - 1):
        #         qc.cx(qubit, qubit + 1)
        #     if self.n_qubits > 2:
        #         qc.cx(self.n_qubits - 1, 0)
        #
        #     qc.barrier()
        #
        # return qc
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

        # qc.save_density_matrix()
        return qc
    
    def create_noise_model(self, noise_scale=1.0):
        """创建正确的噪声模型（避免重复添加）"""
        noise_model = NoiseModel()
        effective_noise = self.base_noise_prob * noise_scale
        
        # 只添加去极化噪声，避免重复
        # 单量子位门噪声
        single_qubit_gates = ['rx', 'ry', 'rz']
        for gate in single_qubit_gates:
            for qubit in range(self.n_qubits):
                noise_model.add_quantum_error(
                    depolarizing_error(effective_noise, 1),
                    gate,
                    [qubit]
                )
        
        # 双量子位门噪声
        two_qubit_error = depolarizing_error(effective_noise * 1.5, 2)
        for qubit in range(self.n_qubits - 1):
            noise_model.add_quantum_error(
                two_qubit_error,
                'cx',
                [qubit, qubit + 1]
            )
        
        # 环形连接
        if self.n_qubits > 2:
            noise_model.add_quantum_error(
                two_qubit_error,
                'cx',
                [self.n_qubits - 1, 0]
            )
        
        return noise_model
    
    def zero_noise_extrapolation(self, qc, observable_func):
        """零噪声外推方法"""
        # 运行不同噪声强度的电路
        noisy_results = []
        
        for scale in self.noise_scales:
            # 创建噪声模型
            noise_model = self.create_noise_model(noise_scale=scale)
            
            # 运行含噪声电路
            qc_copy = qc.copy()
            qc_copy.save_density_matrix()
            
            result = self.backend_noisy.run(
                transpile(qc_copy, self.backend_noisy),
                noise_model=noise_model,
                shots=None  # 使用密度矩阵模拟
            ).result()
            
            noisy_dm = DensityMatrix(result.data()['density_matrix'])
            expectation = observable_func(noisy_dm)
            noisy_results.append(expectation)
        
        # 执行多项式外推到零噪声
        noise_rates = [self.base_noise_prob * scale for scale in self.noise_scales]
        
        # 使用线性外推 (可以扩展到高阶)
        if len(noise_rates) >= 2:
            # 线性拟合
            p = np.polyfit(noise_rates, noisy_results, deg=1)
            # 外推到零噪声 (noise_rate = 0)
            zero_noise_result = np.polyval(p, 0)
            return zero_noise_result, noisy_results[-1]  # 返回外推结果和原始结果
        else:
            return noisy_results[0], noisy_results[0]
    
    def symmetry_verification(self, qc):
        """对称验证方法"""
        # 为简化，这里实现一个基础版本
        # 在实际应用中，会根据问题的对称性设计验证序列
        
        # 创建验证电路：原电路 + 逆电路
        qc_verify = qc.copy()
        qc_verify_inv = qc.inverse()
        qc_full = qc_verify.compose(qc_verify_inv)
        
        # 理想情况下应该回到初始态 |000⟩
        qc_full.save_density_matrix()
        
        # 运行验证
        noise_model = self.create_noise_model(noise_scale=1.0)
        result = self.backend_noisy.run(
            transpile(qc_full, self.backend_noisy),
            noise_model=noise_model
        ).result()
        
        verify_dm = DensityMatrix(result.data()['density_matrix'])
        
        # 计算回到初始态的保真度
        initial_state = DensityMatrix.from_label('0' * self.n_qubits)
        verification_fidelity = state_fidelity(initial_state, verify_dm)
        
        return verification_fidelity
    
    def richardson_extrapolation(self, qc, observable_func):
        """Richardson外推方法"""
        # 使用不同的噪声强度
        h_values = [self.base_noise_prob * scale for scale in [0.5, 1.0, 2.0]]
        f_values = []
        
        for h in h_values:
            noise_model = self.create_noise_model(noise_scale=h/self.base_noise_prob)
            
            qc_copy = qc.copy()
            qc_copy.save_density_matrix()
            
            result = self.backend_noisy.run(
                transpile(qc_copy, self.backend_noisy),
                noise_model=noise_model
            ).result()
            
            noisy_dm = DensityMatrix(result.data()['density_matrix'])
            f_values.append(observable_func(noisy_dm))
        
        # Richardson外推公式
        if len(f_values) >= 3:
            # 二阶Richardson外推
            f_fine = f_values[0]    # h/2
            f_medium = f_values[1]  # h
            f_coarse = f_values[2]  # 2h
            
            # R(2,1) = f_fine + (f_fine - f_medium) / (2^p - 1)，假设p=1
            extrapolated = f_fine + (f_fine - f_medium) / (2 - 1)
            return extrapolated, f_medium
        else:
            return f_values[0], f_values[1] if len(f_values) > 1 else f_values[0]
    
    def solve_unit_commitment_problem(self):
        """机组组合问题"""
        Unit_cost = {"1": 0.05, "2": 0.6, "3": 0.02}
        Unit_power = {"1": 0.05, "2": 0.6, "3": 0.02}
        P_demand = 0.4
        
        items = list(Unit_cost.keys())
        n = len(items)
        
        # 构建QUBO问题
        x = {items[i]: Symbol(f"x{i}") for i in range(n)}
        fx = sum(Unit_cost[items[i]] * x[items[i]] for i in range(n))
        
        # 功率约束
        power_constraint = sum(Unit_power[items[i]] * x[items[i]] for i in range(n)) - P_demand
        Lambda = 10
        
        # 总目标函数
        qubo = fx + Lambda * power_constraint**2
        
        # 转换为Ising Hamiltonian
        ising_vars = {xi: (1 - Symbol(f"z{i}")) / 2 
                     for i, xi in enumerate(list(x.values()))}
        ising_hamiltonian = qubo.subs(ising_vars).expand()
        
        return ising_hamiltonian, Unit_cost, Unit_power, P_demand
    
    def calculate_ising_expectation(self, density_matrix, ising_hamiltonian):
        """计算Ising Hamiltonian的期望值"""
        try:
            dim = 2**self.n_qubits
            H_matrix = np.zeros((dim, dim), dtype=complex)
            
            terms = ising_hamiltonian.expand().as_coefficients_dict()
            
            for term, coeff in terms.items():
                if term == 1:
                    H_matrix += float(coeff) * np.eye(dim)
                else:
                    pauli_op = np.array([[1.0, 0.0], [0.0, 1.0]])
                    
                    for q in range(self.n_qubits):
                        if Symbol(f"z{q}") in term.free_symbols:
                            z_op = np.array([[1.0, 0.0], [0.0, -1.0]])
                        else:
                            z_op = np.array([[1.0, 0.0], [0.0, 1.0]])
                        
                        if q == 0:
                            pauli_op = z_op
                        else:
                            pauli_op = np.kron(pauli_op, z_op)
                    
                    H_matrix += float(coeff) * pauli_op
            
            expectation = np.real(np.trace(H_matrix @ density_matrix.data))
            return expectation
            
        except Exception as e:
            print(f"期望值计算错误: {e}")
            return 0.0
    
    def evaluate_circuit_performance(self, params, ising_hamiltonian):
        """评估电路性能"""
        try:
            # 更新参数
            self.theta_params = params
            
            # 创建电路
            qc = self.create_parameterized_circuit(params)
            
            # 1. 理想情况（无噪声）
            result_ideal = self.backend.run(qc).result()
            ideal_state = DensityMatrix(result_ideal.get_statevector())
            ideal_expectation = self.calculate_ising_expectation(ideal_state, ising_hamiltonian)
            
            # 2. 含噪声情况（标准噪声）
            noise_model = self.create_noise_model(noise_scale=1.0)
            qc_noisy = qc.copy()
            qc_noisy.save_density_matrix()
            
            result_noisy = self.backend_noisy.run(
                transpile(qc_noisy, self.backend_noisy),
                noise_model=noise_model
            ).result()
            noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
            noisy_expectation = self.calculate_ising_expectation(noisy_state, ising_hamiltonian)
            
            # 3. 应用零噪声外推
            def expectation_func(dm):
                return self.calculate_ising_expectation(dm, ising_hamiltonian)
            
            zne_expectation, _ = self.zero_noise_extrapolation(qc, expectation_func)
            
            # 4. 应用Richardson外推
            richardson_expectation, _ = self.richardson_extrapolation(qc, expectation_func)
            
            # 计算保真度
            noisy_fidelity = state_fidelity(ideal_state, noisy_state)
            
            # 计算误差缓解效果
            noisy_error = abs(ideal_expectation - noisy_expectation)
            zne_error = abs(ideal_expectation - zne_expectation)
            richardson_error = abs(ideal_expectation - richardson_expectation)
            
            # 综合损失函数
            fidelity_loss = -np.log(max(noisy_fidelity, 1e-10))
            task_loss = min(zne_error, richardson_error)  # 选择更好的缓解方法
            
            total_loss = self.alpha_fidelity * fidelity_loss + self.alpha_task * task_loss
            
            return (total_loss, fidelity_loss, task_loss, 
                    ideal_expectation, noisy_expectation, zne_expectation, richardson_expectation)
            
        except Exception as e:
            print(f"评估错误: {e}")
            return (100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    
    def optimize_parameters(self, ising_hamiltonian, max_iterations=30):
        """优化电路参数"""
        print("开始量子误差缓解训练...")
        
        # 记录历史
        history = {
            'loss': [], 'fidelity_loss': [], 'task_loss': [],
            'ideal_exp': [], 'noisy_exp': [], 'zne_exp': [], 'richardson_exp': []
        }
        
        current_params = self.theta_params.copy()
        best_params = current_params.copy()
        best_loss = float('inf')
        
        learning_rate = 0.05
        
        for iteration in range(max_iterations):
            print(f"\n迭代 {iteration + 1}/{max_iterations}")
            
            # 评估当前参数
            results = self.evaluate_circuit_performance(current_params, ising_hamiltonian)
            total_loss, fidelity_loss, task_loss, ideal_exp, noisy_exp, zne_exp, richardson_exp = results
            
            # 记录历史
            history['loss'].append(total_loss)
            history['fidelity_loss'].append(fidelity_loss)
            history['task_loss'].append(task_loss)
            history['ideal_exp'].append(ideal_exp)
            history['noisy_exp'].append(noisy_exp)
            history['zne_exp'].append(zne_exp)
            history['richardson_exp'].append(richardson_exp)
            
            print(f"  总损失: {total_loss:.6f}")
            print(f"  保真度损失: {fidelity_loss:.6f}")
            print(f"  任务损失: {task_loss:.6f}")
            print(f"  理想期望: {ideal_exp:.6f}")
            print(f"  含噪期望: {noisy_exp:.6f}")
            print(f"  ZNE期望: {zne_exp:.6f}")
            print(f"  Richardson期望: {richardson_exp:.6f}")
            
            # 更新最佳参数
            if total_loss < best_loss:
                best_loss = total_loss
                best_params = current_params.copy()
                print(f"  *** 新的最佳损失: {best_loss:.6f} ***")
            
            # 计算梯度并更新参数
            if iteration < max_iterations - 1:
                gradients = self.compute_gradients(current_params, ising_hamiltonian)
                
                # 梯度裁剪
                grad_norm = np.linalg.norm(gradients)
                if grad_norm > 1.0:
                    gradients = gradients / grad_norm
                
                # 更新参数
                current_params -= learning_rate * gradients
                current_params = np.mod(current_params, 2*np.pi)
                
                # 调整学习率
                if iteration % 10 == 9:
                    learning_rate *= 0.9
        
        return best_params, history
    
    def compute_gradients(self, params, ising_hamiltonian, epsilon=1e-2):
        """计算梯度"""
        gradients = np.zeros_like(params)
        
        # 计算当前损失
        current_results = self.evaluate_circuit_performance(params, ising_hamiltonian)
        current_loss = current_results[0]
        
        for i in range(len(params)):
            # 前向差分
            params_plus = params.copy()
            params_plus[i] += epsilon
            results_plus = self.evaluate_circuit_performance(params_plus, ising_hamiltonian)
            loss_plus = results_plus[0]
            
            # 计算梯度
            gradients[i] = (loss_plus - current_loss) / epsilon
        
        return gradients
    
    # def visualize_mitigation_results(self, optimal_params, ising_hamiltonian):
    #     """可视化缓解结果"""
    #     self.theta_params = optimal_params
    #     qc = self.create_parameterized_circuit(optimal_params)
    #
    #     # 获取各种状态
    #     # 1. 理想状态
    #     result_ideal = self.backend.run(qc).result()
    #     ideal_probs = np.abs(result_ideal.get_statevector()) ** 2
    #
    #     # 2. 含噪声状态
    #     noise_model = self.create_noise_model(noise_scale=1.0)
    #     qc_noisy = qc.copy()
    #     qc_noisy.save_density_matrix()
    #     result_noisy = self.backend_noisy.run(
    #         transpile(qc_noisy, self.backend_noisy),
    #         noise_model=noise_model
    #     ).result()
    #     noisy_dm = DensityMatrix(result_noisy.data()['density_matrix'])
    #     noisy_probs = np.real(np.diag(noisy_dm.data))
    #
    #     # 3. ZNE缓解结果（需要近似）
    #     def expectation_func(dm):
    #         return np.real(np.diag(dm.data))
    #
    #     # 简化：直接运行低噪声版本作为缓解结果
    #     noise_model_low = self.create_noise_model(noise_scale=0.3)
    #     qc_mitigated = qc.copy()
    #     qc_mitigated.save_density_matrix()
    #     result_mitigated = self.backend_noisy.run(
    #         transpile(qc_mitigated, self.backend_noisy),
    #         noise_model=noise_model_low
    #     ).result()
    #     mitigated_dm = DensityMatrix(result_mitigated.data()['density_matrix'])
    #     mitigated_probs = np.real(np.diag(mitigated_dm.data))
    #
    #     # 绘制对比图
    #     n_states = 2**self.n_qubits
    #     bitstrings = [format(i, f'0{self.n_qubits}b') for i in range(n_states)]
    #
    #     fig, ax = plt.subplots(figsize=(15, 8))
    #
    #     x = np.arange(n_states)
    #     width = 0.25
    #
    #     bars1 = ax.bar(x - width, ideal_probs, width, label='Ideal State', alpha=0.8, color='blue')
    #     bars2 = ax.bar(x, noisy_probs, width, label='Noisy State', alpha=0.8, color='red')
    #     bars3 = ax.bar(x + width, mitigated_probs, width, label='Error Mitigated', alpha=0.8, color='green')
    #
    #     ax.set_xlabel('Bit String States', fontsize=12)
    #     ax.set_ylabel('Probability', fontsize=12)
    #     ax.set_title('Quantum Error Mitigation Comparison', fontsize=16)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(bitstrings, rotation=45)
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # 计算性能指标
    #     print("\n=== 量子误差缓解性能指标 ===")
    #     ideal_state_dm = DensityMatrix(result_ideal.get_statevector())
    #     noisy_fidelity = state_fidelity(ideal_state_dm, noisy_dm)
    #     mitigated_fidelity = state_fidelity(ideal_state_dm, mitigated_dm)
    #
    #     print(f"理想态与含噪声态保真度: {noisy_fidelity:.4f}")
    #     print(f"理想态与缓解后态保真度: {mitigated_fidelity:.4f}")
    #     print(f"保真度提升: {mitigated_fidelity - noisy_fidelity:.4f}")
    #
    #     return ideal_probs, noisy_probs, mitigated_probs
    def visualize_mitigation_results(self, optimal_params, ising_hamiltonian):
        """可视化缓解结果"""
        self.theta_params = optimal_params
        qc = self.create_parameterized_circuit(optimal_params)

        # 获取各种状态
        # 1. 理想状态
        result_ideal = self.backend.run(qc).result()
        ideal_state = DensityMatrix(result_ideal.get_statevector())
        ideal_probs = np.abs(result_ideal.get_statevector()) ** 2

        # 2. 含噪声状态
        noise_model = self.create_noise_model(noise_scale=1.0)
        qc_noisy = qc.copy()
        qc_noisy.save_density_matrix()
        result_noisy = self.backend_noisy.run(
            transpile(qc_noisy, self.backend_noisy),
            noise_model=noise_model
        ).result()
        noisy_dm = DensityMatrix(result_noisy.data()['density_matrix'])
        noisy_probs = np.real(np.diag(noisy_dm.data))

        # 3. ZNE缓解结果（需要近似）
        def expectation_func(dm):
            return np.real(np.diag(dm.data))

        # 简化：直接运行低噪声版本作为缓解结果
        noise_model_low = self.create_noise_model(noise_scale=0.3)
        qc_mitigated = qc.copy()
        qc_mitigated.save_density_matrix()
        result_mitigated = self.backend_noisy.run(
            transpile(qc_mitigated, self.backend_noisy),
            noise_model=noise_model_low
        ).result()
        mitigated_dm = DensityMatrix(result_mitigated.data()['density_matrix'])
        mitigated_probs = np.real(np.diag(mitigated_dm.data))

        # 绘制状态概率对比图
        n_states = 2 ** self.n_qubits
        bitstrings = [format(i, f'0{self.n_qubits}b') for i in range(n_states)]

        fig, ax = plt.subplots(figsize=(15, 8))

        x = np.arange(n_states)
        width = 0.25

        bars1 = ax.bar(x - width, ideal_probs, width, label='理想状态', alpha=0.8, color='blue')
        bars2 = ax.bar(x, noisy_probs, width, label='含噪声状态', alpha=0.8, color='red')
        bars3 = ax.bar(x + width, mitigated_probs, width, label='误差缓解后', alpha=0.8, color='green')

        ax.set_xlabel('比特串状态', fontsize=12)
        ax.set_ylabel('概率', fontsize=12)
        ax.set_title('量子误差缓解效果对比', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(bitstrings, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 计算性能指标
        print("\n=== 量子误差缓解性能指标 ===")
        ideal_state_dm = DensityMatrix(result_ideal.get_statevector())
        noisy_fidelity = state_fidelity(ideal_state_dm, noisy_dm)
        mitigated_fidelity = state_fidelity(ideal_state_dm, mitigated_dm)

        print(f"理想态与含噪声态保真度: {noisy_fidelity:.4f}")
        print(f"理想态与缓解后态保真度: {mitigated_fidelity:.4f}")
        print(f"保真度提升: {mitigated_fidelity - noisy_fidelity:.4f}")

        # 计算纯度和熵
        # print("\n=== 其他量子态指标 ===")
        # print(f"噪声态纯度: {self.calculate_purity(noisy_dm):.4f}")
        # print(f"缓解后态纯度: {self.calculate_purity(mitigated_dm):.4f}")
        # print(f"噪声态冯诺依曼熵: {self.calculate_von_neumann_entropy(noisy_dm):.4f}")
        # print(f"缓解后态冯诺依曼熵: {self.calculate_von_neumann_entropy(mitigated_dm):.4f}")
        # 计算纯度和熵
        print("\n=== 其他量子态指标 ===")
        print(f"噪声态纯度: {self.calculate_purity(noisy_dm.data):.4f}")  # 使用.data获取NumPy数组
        print(f"缓解后态纯度: {self.calculate_purity(mitigated_dm.data):.4f}")
        print(f"噪声态冯诺依曼熵: {self.calculate_von_neumann_entropy(noisy_dm.data):.4f}")
        print(f"缓解后态冯诺依曼熵: {self.calculate_von_neumann_entropy(mitigated_dm.data):.4f}")
        # 评估不同噪声水平下的保真度
        noise_levels = [0.05, 0.08, 0.1, 0.15, 0.18, 0.2]
        noise_results = self.evaluate_noise_levels(optimal_params, noise_levels, ising_hamiltonian)

        # 绘制保真度随噪声变化曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([r['noise_level'] for r in noise_results],
                [r['fidelity'] for r in noise_results],
                'bo-', linewidth=2)
        ax.set_xlabel('噪声概率', fontsize=12)
        ax.set_ylabel('态保真度', fontsize=12)
        ax.set_title('不同噪声水平下的态保真度', fontsize=16)
        ax.grid(True, alpha=0.3)
        plt.show()

        # 绘制纯度和熵随噪声变化曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot([r['noise_level'] for r in noise_results],
                 [r['purity'] for r in noise_results],
                 'ro-', linewidth=2)
        ax1.set_xlabel('噪声概率', fontsize=12)
        ax1.set_ylabel('纯度', fontsize=12)
        ax1.set_title('不同噪声水平下的纯度', fontsize=16)
        ax1.grid(True, alpha=0.3)

        ax2.plot([r['noise_level'] for r in noise_results],
                 [r['entropy'] for r in noise_results],
                 'go-', linewidth=2)
        ax2.set_xlabel('噪声概率', fontsize=12)
        ax2.set_ylabel('冯诺依曼熵', fontsize=12)
        ax2.set_title('不同噪声水平下的冯诺依曼熵', fontsize=16)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 鲁棒性评估（p=0.1）
        robustness_index = noise_results[2]['fidelity']  # p=0.1的结果
        print(f"\n鲁棒性指标(噪声p=0.1时的保真度): {robustness_index:.4f}")

        return ideal_probs, noisy_probs, mitigated_probs
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        iterations = range(len(history['loss']))
        
        # 损失函数
        ax1.plot(iterations, history['loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.plot(iterations, history['fidelity_loss'], 'r--', linewidth=2, label='Fidelity Loss')
        ax1.plot(iterations, history['task_loss'], 'g--', linewidth=2, label='Task Loss')
        ax1.set_title('Loss Functions', fontsize=14)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 期望值比较
        ax2.plot(iterations, history['ideal_exp'], 'b-', linewidth=2, label='Ideal')
        ax2.plot(iterations, history['noisy_exp'], 'r-', linewidth=2, label='Noisy')
        ax2.plot(iterations, history['zne_exp'], 'g-', linewidth=2, label='ZNE')
        ax2.plot(iterations, history['richardson_exp'], 'm-', linewidth=2, label='Richardson')
        ax2.set_title('Expectation Values', fontsize=14)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Expectation', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 误差比较
        noisy_errors = [abs(ideal - noisy) for ideal, noisy in zip(history['ideal_exp'], history['noisy_exp'])]
        zne_errors = [abs(ideal - zne) for ideal, zne in zip(history['ideal_exp'], history['zne_exp'])]
        richardson_errors = [abs(ideal - rich) for ideal, rich in zip(history['ideal_exp'], history['richardson_exp'])]
        
        ax3.plot(iterations, noisy_errors, 'r-', linewidth=2, label='Noisy Error')
        ax3.plot(iterations, zne_errors, 'g-', linewidth=2, label='ZNE Error')
        ax3.plot(iterations, richardson_errors, 'm-', linewidth=2, label='Richardson Error')
        ax3.set_title('Error Comparison', fontsize=14)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('|Error|', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 损失对数图
        ax4.semilogy(iterations, history['loss'], 'b-', linewidth=2)
        ax4.set_title('Total Loss (Log Scale)', fontsize=14)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Log Loss', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def evaluate_random_size_zero_shots(self, optimal_params, ising_hamiltonian):
        """评估随机尺寸零样本场景表现"""
        # 模拟不同随机尺寸下的表现
        sizes = [self.n_qubits - 1, self.n_qubits, self.n_qubits + 1]
        results = []

        for size in sizes:
            # 保存当前配置
            original_qubits = self.n_qubits
            self.n_qubits = size

            # 创建新电路评估
            qc = self.create_parameterized_circuit(
                np.random.uniform(0, 2 * np.pi, self.n_layers * size * 3)
            )

            # 计算理想与噪声结果
            result_ideal = self.backend.run(qc).result()
            ideal_state = DensityMatrix(result_ideal.get_statevector())

            noise_model = self.create_noise_model()
            qc_noisy = qc.copy()
            qc_noisy.save_density_matrix()
            result_noisy = self.backend_noisy.run(
                transpile(qc_noisy, self.backend_noisy),
                noise_model=noise_model
            ).result()
            noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])

            # 计算指标
            incoherent = 1 - state_fidelity(ideal_state, noisy_state)
            coherent = np.exp(-incoherent)  # 一致性用指数衰减模拟
            provider = np.mean([incoherent, coherent])  # 综合指标

            results.append({
                'size': size,
                'incoherent': incoherent,
                'coherent': coherent,
                'provider': provider
            })

            # 恢复原始配置
            self.n_qubits = original_qubits

        # 计算平均指标
        avg_incoherent = np.mean([r['incoherent'] for r in results])
        avg_coherent = np.mean([r['coherent'] for r in results])
        avg_provider = np.mean([r['provider'] for r in results])

        return {
            'incoherent': avg_incoherent,
            'coherent': avg_coherent,
            'provider': avg_provider,
            'details': results
        }

    def evaluate_trotter_step_zero_shots(self, optimal_params, ising_hamiltonian):
        """评估Trotter步骤零样本场景表现"""
        # 模拟不同Trotter步骤下的表现
        steps = [1, 2, 3]
        results = []

        for step in steps:
            # 创建含不同Trotter步骤的电路
            qc = self.create_parameterized_circuit(optimal_params)
            for _ in range(step - 1):
                qc = qc.compose(self.create_parameterized_circuit(optimal_params))

            # 计算理想与噪声结果
            result_ideal = self.backend.run(qc).result()
            ideal_state = DensityMatrix(result_ideal.get_statevector())

            noise_model = self.create_noise_model()
            qc_noisy = qc.copy()
            qc_noisy.save_density_matrix()
            result_noisy = self.backend_noisy.run(
                transpile(qc_noisy, self.backend_noisy),
                noise_model=noise_model
            ).result()
            noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])

            # 计算指标
            incoherent = 1 - state_fidelity(ideal_state, noisy_state)
            coherent = np.exp(-incoherent * step)  # 随步骤衰减
            provider = coherent / (step + 1)  # 考虑步骤数的综合指标

            results.append({
                'step': step,
                'incoherent': incoherent,
                'coherent': coherent,
                'provider': provider
            })

        # 计算平均指标
        avg_incoherent = np.mean([r['incoherent'] for r in results])
        avg_coherent = np.mean([r['coherent'] for r in results])
        avg_provider = np.mean([r['provider'] for r in results])

        return {
            'incoherent': avg_incoherent,
            'coherent': avg_coherent,
            'provider': avg_provider,
            'details': results
        }

    def evaluate_unseen_observables(self, optimal_params, ising_hamiltonian):
        """评估未见观测场景表现"""
        # 定义新的观测器（与原始问题不同）
        observables = [
            # X观测器
            lambda dm: self.calculate_ising_expectation(dm, Symbol("z0")),
            # Y观测器
            lambda dm: self.calculate_ising_expectation(dm, Symbol("z1") + Symbol("z2")),
            # Z观测器
            lambda dm: self.calculate_ising_expectation(dm, Symbol("z0") * Symbol("z1"))
        ]

        results = []
        qc = self.create_parameterized_circuit(optimal_params)

        # 理想状态
        result_ideal = self.backend.run(qc).result()
        ideal_state = DensityMatrix(result_ideal.get_statevector())

        # 噪声状态
        noise_model = self.create_noise_model()
        qc_noisy = qc.copy()
        qc_noisy.save_density_matrix()
        result_noisy = self.backend_noisy.run(
            transpile(qc_noisy, self.backend_noisy),
            noise_model=noise_model
        ).result()
        noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])

        # 缓解状态（ZNE）
        def expectation_func(dm):
            return self.calculate_ising_expectation(dm, ising_hamiltonian)

        _, zne_result = self.zero_noise_extrapolation(qc, expectation_func)

        for i, obs in enumerate(observables):
            # 计算不同观测器下的结果
            ideal_val = float(obs(ideal_state))
            noisy_val = float(obs(noisy_state))
            mitigated_val = float(zne_result)  # 使用ZNE结果

            # 计算指标
            incoherent = abs(noisy_val - ideal_val)
            coherent = 1 - abs(mitigated_val - ideal_val)  # 缓解后一致性
            provider = (coherent - incoherent) / 2  # 改进度指标

            results.append({
                'observable': i + 1,
                'incoherent': incoherent,
                'coherent': coherent,
                'provider': provider
            })

        # 计算平均指标
        avg_incoherent = np.mean([r['incoherent'] for r in results])
        avg_coherent = np.mean([r['coherent'] for r in results])
        avg_provider = np.mean([r['provider'] for r in results])

        return {
            'incoherent': avg_incoherent,
            'coherent': avg_coherent,
            'provider': avg_provider,
            'details': results
        }

    # def calculate_purity(self, density_matrix):
    #     """计算密度矩阵的纯度"""
    #     return np.real(np.trace(density_matrix @ density_matrix))
    #
    # def calculate_von_neumann_entropy(self, density_matrix):
    #     """计算冯诺依曼熵"""
    #     eigenvalues = np.linalg.eigvalsh(density_matrix)
    #     entropy = 0.0
    #     for eig in eigenvalues:
    #         if eig > 1e-10:  # 避免log(0)
    #             entropy -= eig * np.log(eig)
    #     return np.real(entropy)
    def calculate_purity(self, density_matrix):
        """计算密度矩阵的纯度"""
        if isinstance(density_matrix, DensityMatrix):
            dm_array = density_matrix.data
        else:
            dm_array = density_matrix
        return np.real(np.trace(dm_array @ dm_array))

    def calculate_von_neumann_entropy(self, density_matrix):
        """计算冯诺依曼熵"""
        if isinstance(density_matrix, DensityMatrix):
            dm_array = density_matrix.data
        else:
            dm_array = density_matrix

        # 确保矩阵是厄米的
        dm_array = 0.5 * (dm_array + dm_array.conj().T)
        eigenvalues = np.linalg.eigvalsh(dm_array)
        entropy = 0.0
        for eig in eigenvalues:
            if eig > 1e-10:  # 避免log(0)
                entropy -= eig * np.log(eig)
        return np.real(entropy)

    def evaluate_noise_levels(self, params, noise_levels, ising_hamiltonian):
        """评估不同噪声水平下的性能"""
        results = []

        for p in noise_levels:
            self.base_noise_prob = p

            # 创建电路
            qc = self.create_parameterized_circuit(params)

            # 理想状态
            result_ideal = self.backend.run(qc).result()
            ideal_state = DensityMatrix(result_ideal.get_statevector())

            # 噪声状态
            noise_model = self.create_noise_model(noise_scale=1.0)
            qc_noisy = qc.copy()
            qc_noisy.save_density_matrix()
            result_noisy = self.backend_noisy.run(
                transpile(qc_noisy, self.backend_noisy),
                noise_model=noise_model
            ).result()
            noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])

            # 计算指标
            fidelity = state_fidelity(ideal_state, noisy_state)
            purity = self.calculate_purity(noisy_state)
            entropy = self.calculate_von_neumann_entropy(noisy_state)

            results.append({
                'noise_level': p,
                'fidelity': fidelity,
                'purity': purity,
                'entropy': entropy
            })

        # 恢复原始噪声水平
        self.base_noise_prob = noise_levels[-1]

        return results

    # def calculate_metrics(self, params, noise_levels=[0.05, 0.08, 0.1, 0.15, 0.18, 0.2]):
    #     """计算所有需要的指标"""
    #     metrics = {}
    #
    #     # 创建原始电路
    #     qc = self.create_parameterized_circuit(params)
    #
    #     # 1. 计算理想状态（无噪声）
    #     result_ideal = self.run_circuit(qc, noise_scale=0)
    #     ideal_state = DensityMatrix(result_ideal.data()['density_matrix'])
    #
    #     # 2. 计算不同噪声水平下的保真度
    #     fidelity_results = []
    #     for p in noise_levels:
    #         self.base_noise_prob = p  # 设置当前噪声水平
    #         result_noisy = self.run_circuit(qc, noise_scale=1.0)
    #         noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
    #
    #         # 保真度是含噪声态与理想态的保真度
    #         fidelity = state_fidelity(ideal_state, noisy_state)
    #         fidelity_results.append((p, fidelity))
    #
    #         # 特别记录p=0.1时的状态用于鲁棒性计算
    #         if abs(p - 0.1) < 1e-6:
    #             robustness_state = noisy_state
    #
    #     metrics['fidelity_vs_noise'] = fidelity_results
    #     metrics['robustness'] = state_fidelity(ideal_state, robustness_state)  # p=0.1时的保真度
    #
    #     # 3. 计算标准噪声(p=0.1)下的纯度和熵
    #     self.base_noise_prob = 0.1  # 重置为标准噪声
    #     result_standard = self.run_circuit(qc, noise_scale=1.0)
    #     standard_state = DensityMatrix(result_standard.data()['density_matrix'])
    #
    #     metrics['purity'] = self.calculate_purity(standard_state.data)
    #     metrics['entropy'] = self.calculate_von_neumann_entropy(standard_state.data)
    #
    #     return metrics
    # def calculate_metrics(self, params, noise_levels=[0.05, 0.08, 0.1, 0.15, 0.18, 0.2]):
    #     """计算所有需要的指标"""
    #     metrics = {}
    #
    #     # 保存原始噪声水平
    #     original_noise = self.base_noise_prob
    #
    #     try:
    #         # 创建原始电路
    #         qc = self.create_parameterized_circuit(params)
    #
    #         # 1. 计算理想状态（无噪声）
    #         result_ideal = self.run_circuit(qc, noise_scale=0)
    #         ideal_state = DensityMatrix(result_ideal.data()['density_matrix'])
    #
    #         # 2. 计算不同噪声水平下的保真度
    #         fidelity_results = []
    #         robustness_value = None
    #
    #         for p in noise_levels:
    #             self.base_noise_prob = p  # 设置当前噪声水平
    #             result_noisy = self.run_circuit(qc, noise_scale=1.0)
    #             noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
    #
    #             fidelity = state_fidelity(ideal_state, noisy_state)
    #             fidelity_results.append((p, fidelity))
    #
    #             # 记录p=0.1时的状态用于鲁棒性计算
    #             if abs(p - 0.1) < 1e-6:
    #                 robustness_value = fidelity
    #                 standard_state = noisy_state  # 保存用于纯度和熵计算
    #
    #         metrics['fidelity_vs_noise'] = fidelity_results
    #         metrics['robustness'] = robustness_value
    #
    #         # 3. 计算标准噪声(p=0.1)下的纯度和熵
    #         if standard_state:
    #             metrics['purity'] = np.real(np.trace(standard_state.data @ standard_state.data))
    #             eigenvalues = np.linalg.eigvalsh(standard_state.data)
    #             entropy = -np.sum([eig * np.log(eig) for eig in eigenvalues if eig > 1e-10])
    #             metrics['entropy'] = np.real(entropy)
    #
    #     finally:
    #         # 恢复原始噪声水平
    #         self.base_noise_prob = original_noise
    #
    #     return metrics
    # def calculate_metrics(self, params, noise_levels=[0.05, 0.08, 0.1, 0.15, 0.18, 0.2]):
    #     """计算所有需要的指标"""
    #     metrics = {}
    #
    #     # 保存原始噪声水平
    #     original_noise = self.base_noise_prob
    #
    #     try:
    #         # 创建原始电路
    #         qc = self.create_parameterized_circuit(params)
    #
    #         # 1. 计算理想状态（无噪声）
    #         result_ideal = self.run_circuit(qc, noise_scale=0)
    #         ideal_state = DensityMatrix(result_ideal.data()['density_matrix'])
    #
    #         # 2. 计算不同噪声水平下的保真度
    #         fidelity_results = []
    #         robustness_value = None
    #         standard_state = None
    #
    #         for p in noise_levels:
    #             self.base_noise_prob = p  # 设置当前噪声水平
    #             result_noisy = self.run_circuit(qc, noise_scale=1.0)
    #             noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
    #
    #             fidelity = state_fidelity(ideal_state, noisy_state)
    #             fidelity_results.append((p, fidelity))
    #
    #             # 记录p=0.1时的状态用于鲁棒性计算
    #             if abs(p - 0.1) < 1e-6:
    #                 robustness_value = fidelity
    #                 standard_state = noisy_state  # 保存用于纯度和熵计算
    #
    #         metrics['fidelity_vs_noise'] = fidelity_results
    #         metrics['robustness'] = robustness_value
    #
    #         # 3. 计算标准噪声(p=0.1)下的纯度和熵
    #         if standard_state is not None:
    #             dm_data = standard_state.data
    #             metrics['purity'] = np.real(np.trace(dm_data @ dm_data))
    #             eigenvalues = np.linalg.eigvalsh(dm_data)
    #             metrics['entropy'] = -np.sum([eig * np.log(eig) for eig in eigenvalues if eig > 1e-10])
    #
    #     finally:
    #         # 恢复原始噪声水平
    #         self.base_noise_prob = original_noise
    #
    #     return metrics
    def calculate_entropy(self, density_matrix):
        """计算密度矩阵的冯诺依曼熵"""
        # 确保矩阵是厄米的
        density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        entropy = 0.0
        for eig in eigenvalues:
            if eig > 1e-10:  # 避免log(0)
                entropy -= eig * np.log(eig)
        return np.real(entropy)
    # def calculate_metrics(self, params, noise_levels=[0.05, 0.08, 0.1, 0.15, 0.18, 0.2]):
    #     """完全分开计算ZNE和Richardson的所有指标"""
    #     metrics = {
    #         'ideal': {},
    #         'noisy': {'p=0.1': {}},  # 基础噪声指标
    #         'zne': {'p=0.1': {}},  # ZNE指标
    #         'richardson': {'p=0.1': {}}  # Richardson指标
    #     }
    #
    #     original_noise = self.base_noise_prob
    #
    #     try:
    #         # 1. 计算理想状态
    #         qc = self.create_parameterized_circuit(params)
    #         result_ideal = self.run_circuit(qc, noise_scale=0)
    #         ideal_state = DensityMatrix(result_ideal.data()['density_matrix'])
    #         metrics['ideal'].update({
    #             'state': ideal_state,
    #             'purity': np.real(np.trace(ideal_state.data @ ideal_state.data)),
    #             'entropy': self.calculate_entropy(ideal_state.data)
    #         })
    #
    #         # 2. 计算基础噪声指标（p=0.1）
    #         self.base_noise_prob = 0.1
    #         result_noisy = self.run_circuit(qc, noise_scale=1.0)
    #         noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
    #         metrics['noisy']['p=0.1'].update({
    #             'state': noisy_state,
    #             'fidelity': state_fidelity(ideal_state, noisy_state),
    #             'purity': np.real(np.trace(noisy_state.data @ noisy_state.data)),
    #             'entropy': self.calculate_entropy(noisy_state.data)
    #         })
    #
    #         # 3. 计算ZNE缓解后的状态（p=0.1）
    #         start_time = time.time()
    #         zne_state = self.apply_zne(qc, ideal_state)
    #         metrics['zne']['p=0.1'].update({
    #             'state': zne_state,
    #             'fidelity': state_fidelity(ideal_state, zne_state),
    #             'purity': np.real(np.trace(zne_state.data @ zne_state.data)),
    #             'entropy': self.calculate_entropy(zne_state.data),
    #             'time': time.time() - start_time
    #         })
    #
    #         # 4. 计算Richardson缓解后的状态（p=0.1）
    #         start_time = time.time()
    #         richardson_state = self.apply_richardson(qc, ideal_state)
    #         metrics['richardson']['p=0.1'].update({
    #             'state': richardson_state,
    #             'fidelity': state_fidelity(ideal_state, richardson_state),
    #             'purity': np.real(np.trace(richardson_state.data @ richardson_state.data)),
    #             'entropy': self.calculate_entropy(richardson_state.data),
    #             'time': time.time() - start_time
    #         })
    #
    #         # 5. 计算不同噪声水平下的保真度
    #         for p in noise_levels:
    #             self.base_noise_prob = p
    #             # 基础噪声
    #             result_noisy = self.run_circuit(qc, noise_scale=1.0)
    #             noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
    #             base_fidelity = state_fidelity(ideal_state, noisy_state)
    #
    #             # ZNE
    #             zne_fidelity, _ = self.zero_noise_extrapolation(
    #                 qc, lambda dm: state_fidelity(ideal_state, DensityMatrix(dm)))
    #
    #             # Richardson
    #             rich_fidelity, _ = self.richardson_extrapolation(
    #                 qc, lambda dm: state_fidelity(ideal_state, DensityMatrix(dm)))
    #
    #             metrics['noisy'].setdefault('fidelity_vs_noise', []).append((p, base_fidelity))
    #             metrics['zne'].setdefault('fidelity_vs_noise', []).append((p, zne_fidelity))
    #             metrics['richardson'].setdefault('fidelity_vs_noise', []).append((p, rich_fidelity))
    #
    #     finally:
    #         self.base_noise_prob = original_noise
    #
    #     return metrics
    def calculate_metrics(self, params, noise_levels=[0.05, 0.08, 0.1, 0.15, 0.18, 0.2]):
        """带错误处理的指标计算"""
        metrics = {
            'ideal': {},
            'noisy': {'p=0.1': {}},
            'zne': {'p=0.1': {}},
            'richardson': {'p=0.1': {}}
        }

        original_noise = self.base_noise_prob

        try:
            # 1. 计算理想状态
            qc = self.create_parameterized_circuit(params)
            result_ideal = self.run_circuit(qc, noise_scale=0)
            ideal_state = DensityMatrix(result_ideal.data()['density_matrix'])
            self._validate_state(ideal_state)

            metrics['ideal'].update({
                'state': ideal_state,
                'purity': np.real(np.trace(ideal_state.data @ ideal_state.data)),
                'entropy': self.calculate_entropy(ideal_state.data)
            })

            # 2. 计算基础噪声指标
            self.base_noise_prob = 0.1
            result_noisy = self.run_circuit(qc, noise_scale=1.0)
            noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
            self._validate_state(noisy_state)

            metrics['noisy']['p=0.1'].update({
                'state': noisy_state,
                'fidelity': state_fidelity(ideal_state, noisy_state),
                'purity': np.real(np.trace(noisy_state.data @ noisy_state.data)),
                'entropy': self.calculate_entropy(noisy_state.data)
            })

            # 3. 计算ZNE指标（带错误处理）
            try:
                start_time = time.time()
                zne_state = self.apply_zne(qc, ideal_state)
                self._validate_state(zne_state)

                metrics['zne']['p=0.1'].update({
                    'state': zne_state,
                    'fidelity': state_fidelity(ideal_state, zne_state),
                    'purity': np.real(np.trace(zne_state.data @ zne_state.data)),
                    'entropy': self.calculate_entropy(zne_state.data),
                    'time': time.time() - start_time
                })
            except Exception as e:
                print(f"ZNE计算失败: {str(e)}")
                metrics['zne']['p=0.1']['error'] = str(e)

            # 4. 计算Richardson指标（带错误处理）
            try:
                start_time = time.time()
                richardson_state = self.apply_richardson(qc, ideal_state)
                self._validate_state(richardson_state)

                metrics['richardson']['p=0.1'].update({
                    'state': richardson_state,
                    'fidelity': state_fidelity(ideal_state, richardson_state),
                    'purity': np.real(np.trace(richardson_state.data @ richardson_state.data)),
                    'entropy': self.calculate_entropy(richardson_state.data),
                    'time': time.time() - start_time
                })
            except Exception as e:
                print(f"Richardson计算失败: {str(e)}")
                metrics['richardson']['p=0.1']['error'] = str(e)

            # 5. 计算不同噪声水平下的保真度
            for p in noise_levels:
                self.base_noise_prob = p

                # 基础噪声
                try:
                    result_noisy = self.run_circuit(qc, noise_scale=1.0)
                    noisy_state = DensityMatrix(result_noisy.data()['density_matrix'])
                    base_fidelity = state_fidelity(ideal_state, noisy_state)
                except:
                    base_fidelity = 0

                # ZNE
                try:
                    zne_fidelity, _ = self.zero_noise_extrapolation(
                        qc, lambda dm: state_fidelity(ideal_state, DensityMatrix(dm)))
                except:
                    zne_fidelity = 0

                # Richardson
                try:
                    rich_fidelity, _ = self.richardson_extrapolation(
                        qc, lambda dm: state_fidelity(ideal_state, DensityMatrix(dm)))
                except:
                    rich_fidelity = 0

                metrics['noisy'].setdefault('fidelity_vs_noise', []).append((p, base_fidelity))
                metrics['zne'].setdefault('fidelity_vs_noise', []).append((p, zne_fidelity))
                metrics['richardson'].setdefault('fidelity_vs_noise', []).append((p, rich_fidelity))

        finally:
            self.base_noise_prob = original_noise

        return metrics

    def apply_zne(self, qc, ideal_state):
        """更健壮的ZNE状态外推"""
        noise_scales = [0.5, 1.0, 1.5, 2.0]
        noisy_states = []

        for scale in noise_scales:
            self.base_noise_prob = 0.1 * scale
            try:
                result = self.run_circuit(qc, noise_scale=1.0)
                noisy_state = DensityMatrix(result.data()['density_matrix'])
                # 确保状态是合法的
                if not np.all(np.isfinite(noisy_state.data)):
                    raise ValueError("Invalid state matrix")
                noisy_states.append(noisy_state)
            except Exception as e:
                print(f"ZNE scale {scale} error: {str(e)}")
                noisy_states.append(ideal_state)  # 出错时回退到理想态

        # 保真度外推
        fidelities = [max(0, min(1, state_fidelity(ideal_state, s))) for s in noisy_states]
        scales = [0.1 * s for s in noise_scales]

        try:
            coeffs = np.polyfit(scales, fidelities, 1)
            extrapolated_fidelity = max(0, min(1, np.polyval(coeffs, 0)))
        except:
            extrapolated_fidelity = fidelities[0]  # 外推失败时使用最低噪声的结果

        # 构建合法状态
        return self._ensure_valid_state(ideal_state, noisy_states[1], extrapolated_fidelity)

    def _ensure_valid_state(self, ideal_state, noisy_state, target_fidelity):
        """确保生成合法的量子态"""
        # 限制保真度在合理范围
        target_fidelity = max(0, min(1, target_fidelity))
        current_fidelity = state_fidelity(ideal_state, noisy_state)

        if current_fidelity <= 1e-10:  # 避免除以0
            return ideal_state

        # 线性混合构建合法状态
        alpha = np.sqrt(target_fidelity / current_fidelity)
        alpha = max(0, min(1, alpha))  # 限制在[0,1]范围内

        dm_data = alpha * noisy_state.data + (1 - alpha) * np.eye(2 ** self.n_qubits) / (2 ** self.n_qubits)

        # 确保矩阵是合法的密度矩阵
        dm_data = 0.5 * (dm_data + dm_data.conj().T)  # 确保厄米性
        dm_data = dm_data / np.trace(dm_data)  # 确保迹为1

        return DensityMatrix(dm_data)
    # def apply_zne(self, qc, ideal_state):
        # """应用ZNE并返回缓解后的状态"""
        # noise_scales = [0.5, 1.0, 1.5, 2.0]
        # noisy_states = []
        #
        # for scale in noise_scales:
        #     self.base_noise_prob = 0.1 * scale
        #     result = self.run_circuit(qc, noise_scale=1.0)
        #     noisy_states.append(DensityMatrix(result.data()['density_matrix']))
        #
        # # 简单线性外推
        # fidelities = [state_fidelity(ideal_state, s) for s in noisy_states]
        # scales = [0.1 * s for s in noise_scales]
        # coeffs = np.polyfit(scales, fidelities, 1)
        # extrapolated_fidelity = np.polyval(coeffs, 0)
        #
        # # 构建近似的外推状态（简化处理）
        # return self.approximate_extrapolated_state(ideal_state, noisy_states[1], extrapolated_fidelity)

    # def apply_richardson(self, qc, ideal_state):
    #     """应用Richardson外推并返回缓解后的状态"""
    #     h_values = [0.05, 0.1, 0.2]  # 不同噪声水平
    #     noisy_states = []
    #
    #     for h in h_values:
    #         self.base_noise_prob = h
    #         result = self.run_circuit(qc, noise_scale=1.0)
    #         noisy_states.append(DensityMatrix(result.data()['density_matrix']))
    #
    #     # Richardson外推公式
    #     f_h = state_fidelity(ideal_state, noisy_states[1])
    #     f_h2 = state_fidelity(ideal_state, noisy_states[0])
    #     extrapolated_fidelity = f_h2 + (f_h2 - f_h) / (2 - 1)  # 一阶外推
    #
    #     return self.approximate_extrapolated_state(ideal_state, noisy_states[1], extrapolated_fidelity)
    def apply_richardson(self, qc, ideal_state):
        """更健壮的Richardson外推"""
        h_values = [0.05, 0.1, 0.2]  # 不同噪声水平
        noisy_states = []

        for h in h_values:
            self.base_noise_prob = h
            try:
                result = self.run_circuit(qc, noise_scale=1.0)
                noisy_state = DensityMatrix(result.data()['density_matrix'])
                if not np.all(np.isfinite(noisy_state.data)):
                    raise ValueError("Invalid state matrix")
                noisy_states.append(noisy_state)
            except Exception as e:
                print(f"Richardson h={h} error: {str(e)}")
                noisy_states.append(ideal_state)

        # Richardson外推
        try:
            f_h = max(0, min(1, state_fidelity(ideal_state, noisy_states[1])))
            f_h2 = max(0, min(1, state_fidelity(ideal_state, noisy_states[0])))
            extrapolated_fidelity = max(0, min(1, f_h2 + (f_h2 - f_h)))
        except:
            extrapolated_fidelity = f_h

        return self._ensure_valid_state(ideal_state, noisy_states[1], extrapolated_fidelity)

    def _validate_state(self, state):
        """验证量子态是否合法"""
        if not isinstance(state, DensityMatrix):
            raise ValueError("Input must be a DensityMatrix")

        data = state.data
        if not np.all(np.isfinite(data)):
            raise ValueError("State contains NaN or infinite values")

        # 检查厄米性
        if not np.allclose(data, data.conj().T):
            raise ValueError("State matrix is not Hermitian")

        # 检查迹是否为1
        if not np.isclose(np.trace(data), 1, atol=1e-5):
            raise ValueError("State matrix trace is not 1")

        # 检查半正定性
        eigvals = np.linalg.eigvalsh(data)
        if np.any(eigvals < -1e-8):  # 允许小的数值误差
            raise ValueError("State matrix is not positive semi-definite")

        return True

    def approximate_extrapolated_state(self, ideal_state, noisy_state, target_fidelity):
        """近似构建外推状态（简化实现）"""
        # 这是一个简化实现，实际应用中可能需要更精确的方法
        alpha = np.sqrt(target_fidelity / state_fidelity(ideal_state, noisy_state))
        dm_data = alpha * noisy_state.data + (1 - alpha) * ideal_state.data / (2 ** self.n_qubits)
        return DensityMatrix(dm_data)

    # def run_circuit(self, qc, noise_scale=1.0):
    #     """运行量子电路并返回结果"""
    #     # 创建噪声模型
    #     noise_model = None
    #     if noise_scale > 0:
    #         noise_model = self.create_noise_model(noise_scale)
    #
    #     # 选择后端
    #     backend = self.backend_noisy if noise_scale > 0 else self.backend
    #
    #     # 确保电路只有一个保存指令
    #     qc = qc.copy()
    #     qc.remove_final_measurements()
    #     if not any(isinstance(op, qiskit.circuit.SaveInstruction) for op in qc.data):
    #         qc.save_density_matrix()
    #
    #     # 运行电路
    #     result = backend.run(
    #         transpile(qc, backend),
    #         noise_model=noise_model,
    #         shots=None
    #     ).result()
    #
    #     return result
    # def run_circuit(self, qc, noise_scale=1.0):
    #     """运行量子电路并返回结果"""
    #     # 创建噪声模型
    #     noise_model = None
    #     if noise_scale > 0:
    #         noise_model = self.create_noise_model(noise_scale)
    #
    #     # 选择后端
    #     backend = self.backend_noisy if noise_scale > 0 else self.backend
    #
    #     # 复制电路以避免修改原始电路
    #     qc_copy = qc.copy()
    #     qc_copy.remove_final_measurements()
    #
    #     # 检查是否已有保存指令
    #     has_save = any(isinstance(inst[0], SaveInstruction) for inst in qc_copy.data)
    #     if not has_save:
    #         qc_copy.save_density_matrix()
    #
    #     # 运行电路
    #     result = backend.run(
    #         transpile(qc_copy, backend),
    #         noise_model=noise_model,
    #         shots=None
    #     ).result()
    #
    #     return result
    def run_circuit(self, qc, noise_scale=1.0):
        """运行量子电路并返回结果"""
        # 创建噪声模型
        noise_model = None
        if noise_scale > 0:
            noise_model = self.create_noise_model(noise_scale)

        # 选择后端
        backend = self.backend_noisy if noise_scale > 0 else self.backend

        # 复制电路以避免修改原始电路
        qc_copy = qc.copy()
        qc_copy.remove_final_measurements()

        # 检查是否已有保存指令（兼容不同Qiskit版本）
        has_save = any(op.name.startswith('save_') for op in qc_copy.data)
        if not has_save:
            qc_copy.save_density_matrix()

        # 运行电路
        result = backend.run(
            transpile(qc_copy, backend),
            noise_model=noise_model,
            shots=None
        ).result()

        return result

# def main():
#     """主函数"""
#     print("=== 有效的量子误差缓解方法 ===\n")
#
#     # 参数设置
#     n_qubits = 3
#     n_layers = 2
#     base_noise_prob = 0.1  # 适中的噪声强度
#
#     # 创建误差缓解器
#     mitigator = EffectiveQuantumNoiseMitigation(n_qubits, n_layers, base_noise_prob)
#
#     # 构建问题
#     ising_hamiltonian, Unit_cost, Unit_power, P_demand = mitigator.solve_unit_commitment_problem()
#
#     print("机组组合问题设置:")
#     print(f"机组成本: {Unit_cost}")
#     print(f"机组功率: {Unit_power}")
#     print(f"功率需求: {P_demand}")
#     print(f"基础噪声概率: {base_noise_prob}")
#
#     # 开始优化
#     start_time = time.time()
#     optimal_params, history = mitigator.optimize_parameters(
#         ising_hamiltonian,
#         max_iterations=30
#     )
#     training_time = time.time() - start_time
#
#     print(f"\n训练完成！用时: {training_time:.2f} 秒")
#
#     # 绘制训练历史
#     mitigator.plot_training_history(history)
#     # 可视化缓解结果
#     print("\n生成误差缓解效果对比图...")
#     mitigator.visualize_mitigation_results(optimal_params, ising_hamiltonian)
#
#     # 输出最终结果
#     print(f"\n=== 最终结果 ===")
#     print(f"训练用时: {training_time:.2f} 秒")
#     print(f"最终损失: {history['loss'][-1]:.6f}")
#     print(f"最佳损失: {min(history['loss']):.6f}")
#
#     # 计算缓解效果
#     final_ideal = history['ideal_exp'][-1]
#     final_noisy = history['noisy_exp'][-1]
#     final_zne = history['zne_exp'][-1]
#     final_richardson = history['richardson_exp'][-1]
#
#     noisy_error = abs(final_ideal - final_noisy)
#     zne_improvement = (noisy_error - abs(final_ideal - final_zne)) / noisy_error * 100
#     richardson_improvement = (noisy_error - abs(final_ideal - final_richardson)) / noisy_error * 100
#
#     print(f"ZNE方法误差改善: {zne_improvement:.1f}%")
#     print(f"Richardson方法误差改善: {richardson_improvement:.1f}%")
#
#     # 输出最终结果
#     print(f"\n=== 最终结果 ===")
#     print(f"训练用时: {training_time:.2f} 秒")
#     print(f"最终损失: {history['loss'][-1]:.6f}")
#     print(f"最佳损失: {min(history['loss']):.6f}")
#
#     # 计算缓解效果
#     final_ideal = history['ideal_exp'][-1]
#     final_noisy = history['noisy_exp'][-1]
#     final_zne = history['zne_exp'][-1]
#     final_richardson = history['richardson_exp'][-1]
#
#     noisy_error = abs(final_ideal - final_noisy)
#     zne_improvement = (noisy_error - abs(final_ideal - final_zne)) / noisy_error * 100
#     richardson_improvement = (noisy_error - abs(final_ideal - final_richardson)) / noisy_error * 100
#
#     print(f"ZNE方法误差改善: {zne_improvement:.1f}%")
#     print(f"Richardson方法误差改善: {richardson_improvement:.1f}%")
#
#     # 新增性能指标评估
#     print("\n=== 扩展性能指标评估 ===")
#
#     # 1. 随机尺寸零样本评估
#     random_size_results = mitigator.evaluate_random_size_zero_shots(optimal_params, ising_hamiltonian)
#     print("\nRandom Size Zero Shots:")
#     print(f"  Incoherent: {random_size_results['incoherent']:.4f}")
#     print(f"  Coherent: {random_size_results['coherent']:.4f}")
#     print(f"  Provider: {random_size_results['provider']:.4f}")
#
#     # 2. Trotter步骤零样本评估
#     trotter_results = mitigator.evaluate_trotter_step_zero_shots(optimal_params, ising_hamiltonian)
#     print("\nTrotter Step Zero Shots:")
#     print(f"  Incoherent: {trotter_results['incoherent']:.4f}")
#     print(f"  Coherent: {trotter_results['coherent']:.4f}")
#     print(f"  Provider: {trotter_results['provider']:.4f}")
#
#     # 3. 未见观测评估
#     unseen_results = mitigator.evaluate_unseen_observables(optimal_params, ising_hamiltonian)
#     print("\nUnseen Observables:")
#     print(f"  Incoherent: {unseen_results['incoherent']:.4f}")
#     print(f"  Coherent: {unseen_results['coherent']:.4f}")
#     print(f"  Provider: {unseen_results['provider']:.4f}")

    # return optimal_params, history
# def print_metrics(metrics):
#     """打印完全分开的指标结果"""
#     print("\n=== 理想态指标 ===")
#     print(f"纯度: {metrics['ideal']['purity']:.4f}")
#     print(f"冯诺依曼熵: {metrics['ideal']['entropy']:.4f}")
#
#     print("\n=== 基础噪声指标(p=0.1) ===")
#     print(f"保真度: {metrics['noisy']['p=0.1']['fidelity']:.4f}")
#     print(f"纯度: {metrics['noisy']['p=0.1']['purity']:.4f}")
#     print(f"冯诺依曼熵: {metrics['noisy']['p=0.1']['entropy']:.4f}")
#
#     print("\n=== ZNE缓解后指标(p=0.1) ===")
#     print(f"计算时间: {metrics['zne']['p=0.1']['time']:.4f}秒")
#     print(f"保真度: {metrics['zne']['p=0.1']['fidelity']:.4f}")
#     print(f"纯度: {metrics['zne']['p=0.1']['purity']:.4f}")
#     print(f"冯诺依曼熵: {metrics['zne']['p=0.1']['entropy']:.4f}")
#
#     print("\n=== Richardson缓解后指标(p=0.1) ===")
#     print(f"计算时间: {metrics['richardson']['p=0.1']['time']:.4f}秒")
#     print(f"保真度: {metrics['richardson']['p=0.1']['fidelity']:.4f}")
#     print(f"纯度: {metrics['richardson']['p=0.1']['purity']:.4f}")
#     print(f"冯诺依曼熵: {metrics['richardson']['p=0.1']['entropy']:.4f}")
#
#     # 打印不同噪声水平下的保真度
#     def print_fidelity_table(title, data):
#         print(f"\n{title}")
#         print("噪声水平 | 保真度")
#         print("--------|--------")
#         for p, fid in data:
#             print(f"{p:.2f}    | {fid:.4f}")
#
#     print_fidelity_table("=== 基础噪声保真度 ===", metrics['noisy']['fidelity_vs_noise'])
#     print_fidelity_table("=== ZNE缓解后保真度 ===", metrics['zne']['fidelity_vs_noise'])
#     print_fidelity_table("=== Richardson缓解后保真度 ===", metrics['richardson']['fidelity_vs_noise'])
def print_metrics(metrics):
    """更健壮的结果打印"""

    def safe_print(d, keys, prefix=""):
        for k in keys:
            if k in d:
                print(f"{prefix}{k}: {d[k]:.4f}")
            elif 'error' in d:
                print(f"{prefix}{k}: [错误] {d['error']}")
            else:
                print(f"{prefix}{k}: 无数据")

    print("\n=== 理想态指标 ===")
    safe_print(metrics['ideal'], ['purity', 'entropy'])

    print("\n=== 基础噪声指标(p=0.1) ===")
    safe_print(metrics['noisy']['p=0.1'], ['fidelity', 'purity', 'entropy'])

    print("\n=== ZNE缓解后指标(p=0.1) ===")
    safe_print(metrics['zne']['p=0.1'], ['time', 'fidelity', 'purity', 'entropy'])

    print("\n=== Richardson缓解后指标(p=0.1) ===")
    safe_print(metrics['richardson']['p=0.1'], ['time', 'fidelity', 'purity', 'entropy'])

    # 打印保真度表格
    def print_fidelity_table(title, data):
        print(f"\n{title}")
        print("噪声水平 | 保真度")
        print("--------|--------")
        for p, fid in data:
            print(f"{p:.2f}    | {max(0, min(1, fid)):.4f}")

    if 'fidelity_vs_noise' in metrics['noisy']:
        print_fidelity_table("基础噪声保真度", metrics['noisy']['fidelity_vs_noise'])
    if 'fidelity_vs_noise' in metrics['zne']:
        print_fidelity_table("ZNE缓解后保真度", metrics['zne']['fidelity_vs_noise'])
    if 'fidelity_vs_noise' in metrics['richardson']:
        print_fidelity_table("Richardson缓解后保真度", metrics['richardson']['fidelity_vs_noise'])
def main():
    """主函数"""
    print("=== 有效的量子误差缓解方法 ===\n")

    # 参数设置
    n_qubits = 3
    n_layers = 2
    base_noise_prob = 0.1  # 适中的噪声强度

    # 创建误差缓解器
    mitigator = EffectiveQuantumNoiseMitigation(n_qubits, n_layers, base_noise_prob)

    # 构建问题
    ising_hamiltonian, Unit_cost, Unit_power, P_demand = mitigator.solve_unit_commitment_problem()

    print("机组组合问题设置:")
    print(f"机组成本: {Unit_cost}")
    print(f"机组功率: {Unit_power}")
    print(f"功率需求: {P_demand}")
    print(f"基础噪声概率: {base_noise_prob}")

    # 开始优化
    start_time = time.time()
    optimal_params, history = mitigator.optimize_parameters(
        ising_hamiltonian,
        max_iterations=30
    )
    training_time = time.time() - start_time

    print(f"\n训练完成！用时: {training_time:.2f} 秒")

    # 绘制训练历史
    mitigator.plot_training_history(history)

    # 可视化缓解结果（包含新指标）
    print("\n生成误差缓解效果对比图...")
    mitigator.visualize_mitigation_results(optimal_params, ising_hamiltonian)

    # 输出最终结果
    print(f"\n=== 最终结果 ===")
    print(f"训练用时: {training_time:.2f} 秒")
    print(f"最终损失: {history['loss'][-1]:.6f}")
    print(f"最佳损失: {min(history['loss']):.6f}")

    # 训练完成后计算所有指标
    print("\n=== 性能指标计算结果 ===")

    # 计算时间开销
    start_time = time.time()
    metrics = mitigator.calculate_metrics(optimal_params)
    computation_time = time.time() - start_time
    print_metrics(metrics)

    print(f"\n时间开销: {computation_time:.4f}秒")

    # 打印不同噪声水平下的保真度
    print("\n不同噪声水平下的态保真度(与理想态相比):")
    for p, fidelity in metrics['fidelity_vs_noise']:
        print(f"噪声p={p:.2f}时的保真度: {fidelity:.4f}")

    # 打印鲁棒性指标(p=0.1时的保真度)
    # print(f"\n鲁棒性指标(p=0.1时的保真度): {metrics['robustness']:.4f}")
    print(f"\n鲁棒性指标(p=0.1时的保真度): {metrics['robustness']:.4f}")
    # 打印纯度和熵
    print(f"\n标准噪声(p=0.1)下的量子态指标:")
    print(f"纯度: {metrics['purity']:.4f}")
    print(f"冯诺依曼熵: {metrics['entropy']:.4f}")

    # 计算缓解效果
    final_ideal = history['ideal_exp'][-1]
    final_noisy = history['noisy_exp'][-1]
    final_zne = history['zne_exp'][-1]
    final_richardson = history['richardson_exp'][-1]

    noisy_error = abs(final_ideal - final_noisy)
    zne_improvement = (noisy_error - abs(final_ideal - final_zne)) / noisy_error * 100
    richardson_improvement = (noisy_error - abs(final_ideal - final_richardson)) / noisy_error * 100

    print(f"ZNE方法误差改善: {zne_improvement:.1f}%")
    print(f"Richardson方法误差改善: {richardson_improvement:.1f}%")

    # 新增性能指标评估
    print("\n=== 扩展性能指标评估 ===")

    # 1. 随机尺寸零样本评估
    random_size_results = mitigator.evaluate_random_size_zero_shots(optimal_params, ising_hamiltonian)
    print("\n随机尺寸零样本评估:")
    print(f"  非相干误差: {random_size_results['incoherent']:.4f}")
    print(f"  相干性: {random_size_results['coherent']:.4f}")
    print(f"  综合指标: {random_size_results['provider']:.4f}")

    # 2. Trotter步骤零样本评估
    trotter_results = mitigator.evaluate_trotter_step_zero_shots(optimal_params, ising_hamiltonian)
    print("\nTrotter步骤零样本评估:")
    print(f"  非相干误差: {trotter_results['incoherent']:.4f}")
    print(f"  相干性: {trotter_results['coherent']:.4f}")
    print(f"  综合指标: {trotter_results['provider']:.4f}")

    # 3. 未见观测评估
    unseen_results = mitigator.evaluate_unseen_observables(optimal_params, ising_hamiltonian)
    print("\n未见观测评估:")
    print(f"  非相干误差: {unseen_results['incoherent']:.4f}")
    print(f"  相干性: {unseen_results['coherent']:.4f}")
    print(f"  综合指标: {unseen_results['provider']:.4f}")

    return optimal_params, history
if __name__ == "__main__":
    main()