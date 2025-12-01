import torch
import torch.nn as nn

class StudentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, n_classes)
        )

    def forward(self, x):
        return self.net(x)
