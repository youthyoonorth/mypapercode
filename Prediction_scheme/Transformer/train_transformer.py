import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# allow importing from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Transformer.model_transformer import TransformerModel
from ODE.model_ODE import ODELSTMCell


@dataclass
class Config:
    seq_len: int = 20
    num_classes: int = 64
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-3


def make_dataset(n_samples: int, cfg: Config):
    """Generate a synthetic beam tracking dataset."""
    angles = torch.rand(n_samples) * 2 * math.pi
    labels = (angles / (2 * math.pi / cfg.num_classes)).long()
    seq = []
    for theta in angles:
        t = torch.linspace(0, 1, cfg.seq_len)
        # two features: cos(theta) and sin(theta) modulated over time
        feat = torch.stack([torch.cos(theta) * torch.ones_like(t),
                            torch.sin(theta) * torch.ones_like(t)], dim=0)
        feat = feat.unsqueeze(-1)  # shape (2, seq_len, 1)
        seq.append(feat)
    x = torch.stack(seq, dim=0)
    y = labels.unsqueeze(1).repeat(1, cfg.seq_len)
    return x, y


class SimpleODELSTM(nn.Module):
    """Minimal ODE-LSTM baseline for comparison."""

    def __init__(self, input_size=2, hidden_size=64, num_classes=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.ode_cell = ODELSTMCell(hidden_size, hidden_size, solver_type="fixed_euler")
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 2, seq_len, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, seq_len, 2)
        b, t, _ = x.shape
        h = torch.zeros(b, self.hidden_size, device=x.device)
        c = torch.zeros(b, self.hidden_size, device=x.device)
        outputs = []
        ts = torch.ones(b, device=x.device)
        for i in range(t):
            h, c = self.lstm(x[:, i, :], (h, c))
            h = self.ode_cell(h, ts)
            outputs.append(self.fc(h))
        return torch.stack(outputs, dim=1)


def evaluate(model, loader, cfg: Config):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    ang_err = 0.0
    loss_sum = 0.0
    start = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss_sum += criterion(logits.view(-1, cfg.num_classes), yb.view(-1)).item()
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            angle_step = 2 * math.pi / cfg.num_classes
            ang_pred = (preds + 0.5) * angle_step
            ang_true = (yb + 0.5) * angle_step
            ang_err += torch.abs(ang_pred - ang_true).sum().item()
    elapsed = time.time() - start
    return (
        correct / total,
        ang_err / total,
        loss_sum / len(loader),
        elapsed,
    )


def train_model(model, loader, cfg: Config):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits.view(-1, cfg.num_classes), yb.view(-1))
            loss.backward()
            optim.step()


def main():
    cfg = Config()
    train_x, train_y = make_dataset(128, cfg)
    test_x, test_y = make_dataset(32, cfg)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=cfg.batch_size)

    transformer = TransformerModel(
        x_size=cfg.seq_len, y_size=1, z_size=1, num_classes=cfg.num_classes
    )
    ode_lstm = SimpleODELSTM(num_classes=cfg.num_classes)

    print("Training Transformer model")
    train_model(transformer, train_loader, cfg)
    t_acc, t_ang, t_loss, t_time = evaluate(transformer, test_loader, cfg)
    print(f"Transformer accuracy: {t_acc:.3f}, angular error: {t_ang:.3f}, inference time: {t_time:.3f}s")

    print("Training ODE-LSTM model")
    train_model(ode_lstm, train_loader, cfg)
    o_acc, o_ang, o_loss, o_time = evaluate(ode_lstm, test_loader, cfg)
    print(f"ODE-LSTM accuracy: {o_acc:.3f}, angular error: {o_ang:.3f}, inference time: {o_time:.3f}s")


if __name__ == "__main__":
    main()
