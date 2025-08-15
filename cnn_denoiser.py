# import torch
# import torch.nn as nn
#
# class RNN_Denoiser(nn.Module):
#     def __init__(self, input_size=64, hidden_size=128, num_layers=2):
#         super().__init__()
#         self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
#                           num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         x = x.unsqueeze(-1)  # shape: (B, L) → (B, L, 1)
#         out, _ = self.rnn(x)
#         out = self.fc(out).squeeze(-1)  # (B, L, 1) → (B, L)
#         return out
# rnn_denoiser.py
# import torch
# import torch.nn as nn
#
# class RNN_Denoiser(nn.Module):
#     """
#     输入: (B, 64, 2)  # 64个矩阵元素，每个元素实/虚两个特征
#     输出: (B, 64, 2)  # 预测对应理想密度矩阵的实/虚
#     """
#     def __init__(self, hidden_size=128, num_layers=2, dropout=0.0):
#         super().__init__()
#         self.rnn = nn.GRU(
#             input_size=2,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#             bidirectional=True
#         )
#         self.fc = nn.Linear(hidden_size*2, 2)  # 双向
#     def forward(self, x):
#         # x: (B, 64, 2)
#         out, _ = self.rnn(x)
#         out = self.fc(out)  # (B, 64, 2)
#         return out
# import torch
# import torch.nn as nn
#
# class CNN_Denoiser(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(2, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 2, kernel_size=3, padding=1),
#         )
#
#     # def forward(self, x):
#     #     # 输入x形状 (B, 64, 2) 需reshape为 (B, 2, 8, 8)
#     #     x = x.permute(0, 2, 1).reshape(-1, 2, 8, 8)
#     #     x = self.encoder(x)
#     #     x = self.decoder(x)
#     #     x = x.reshape(-1, 2, 64).permute(0, 2, 1)  # (B,64,2)
#     #     return x
#     def forward(self, x):
#         # x shape: (B, 2, 8, 8)
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x




import torch
import torch.nn as nn

class CNN_Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x shape: (B, 2, 8, 8)
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # 输出也是 (B, 2, 8, 8)
