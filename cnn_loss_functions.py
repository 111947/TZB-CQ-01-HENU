# 文件名：loss_functions.py
import numpy as np
import torch

from quantum_data_generator import features_to_density
from postprocess import project_to_physical

def trace_distance(rho1, rho2):
    delta = rho1 - rho2
    eigvals = np.linalg.eigvalsh(delta @ delta.conj().T)
    return 0.5 * np.sum(np.sqrt(np.maximum(eigvals, 0)))

def trace_distance_loss(y_pred, y_true):
    batch_loss = 0
    for i in range(y_pred.shape[0]):
        pred_rho = features_to_density(y_pred[i].detach().cpu().numpy())
        true_rho = features_to_density(y_true[i].detach().cpu().numpy())
        pred_rho = project_to_physical(pred_rho)
        batch_loss += trace_distance(pred_rho, true_rho)
    return torch.tensor(batch_loss / y_pred.shape[0], requires_grad=True)
