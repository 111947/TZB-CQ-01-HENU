# postprocess.py
import numpy as np

# def project_to_physical(rho):
#     """Hermitian + PSD + Trace=1 投影"""
#     rho = (rho + rho.conj().T) / 2
#     evals, vecs = np.linalg.eigh(rho)
#     evals = np.clip(evals, 0, None)
#     s = evals.sum()
#     if s <= 0:
#         # 极端情况全部裁零，回退到 |000> 状态
#         evals[0] = 1.0
#         s = 1.0
#     evals /= s
#     rho_p = (vecs * evals) @ vecs.conj().T
#     return rho_p
# def project_to_physical(rho):
#     rho = (rho + rho.conj().T) / 2
#     evals, vecs = np.linalg.eigh(rho)
#     # evals = np.clip(evals, 0, None)
#     evals = np.where(evals < 1e-6, 0, evals)
#     s = evals.sum()
#     if s <= 1e-10:
#         evals[0] = 1.0
#         s = 1.0
#     evals /= s
#     rho_p = (vecs * evals) @ vecs.conj().T
#     return rho_p
def project_to_physical(rho):
    rho = (rho + rho.conj().T) / 2
    evals, vecs = np.linalg.eigh(rho)
    # 更严格截断负特征值和极小特征值
    evals = np.where(evals < 1e-6, 0, evals)
    s = evals.sum()
    if s < 1e-10:
        evals[0] = 1.0
        s = 1.0
    evals /= s
    rho_p = (vecs * evals) @ vecs.conj().T
    # 对结果再做Hermitian化保证数值对称
    rho_p = (rho_p + rho_p.conj().T) / 2
    return rho_p

def trace_distance(rho1, rho2):
    delta = rho1 - rho2
    delta = (delta + delta.conj().T)/2
    evals, _ = np.linalg.eigh(delta)
    return 0.5 * np.sum(np.abs(evals))
