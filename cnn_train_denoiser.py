import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from quantum_data_generator import generate_dataset
from rnn_denoiser import RNN_Denoiser

# 1. 数据准备
X, Y = generate_dataset(n_samples=2000, noise_prob=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
# 例：在训练前调整形状，注意这里是张量，使用permute和reshape
X_train_cnn = X_train.permute(0, 2, 1).reshape(-1, 2, 8, 8)
Y_train_cnn = Y_train.permute(0, 2, 1).reshape(-1, 2, 8, 8)
X_test_cnn = X_test.permute(0, 2, 1).reshape(-1, 2, 8, 8)
Y_test_cnn = Y_test.permute(0, 2, 1).reshape(-1, 2, 8, 8)

# 2. 模型与训练
model = RNN_Denoiser(input_size=64)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
for epoch in range(epochs):
    model.train()
    # output = model(X_train)
    # loss = loss_fn(output, Y_train)
    # 然后调用模型时直接传CNN输入即可，比如
    output = model(X_train_cnn)
    loss = loss_fn(output, Y_train_cnn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test), Y_test)
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.5f}, Test Loss: {val_loss.item():.5f}")
