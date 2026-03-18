from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poison_tester_ui.data.preprocessing import Preprocess

Mode = Literal["Quick", "Normal"]

@dataclass
class TrainConfig:
    mode: Mode
    max_train_samples: str  # "5k","10k","20k","all"
    epochs: int
    batch_size: int
    optimizer: str  # "SGD"|"Adam"
    lr: float
    weight_decay: float
    seed: int
    num_classes: int

class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

def _cap_n(max_train_samples: str, n: int) -> int:
    if max_train_samples == "all":
        return n
    if max_train_samples.endswith("k"):
        cap = int(max_train_samples[:-1]) * 1000
        return min(n, cap)
    return n

def train_model_simple(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    preprocess: Preprocess,
    cfg: TrainConfig,
    device: str = "cpu",
):
    torch.manual_seed(cfg.seed)
    model = TinyCNN(cfg.num_classes).to(device)

    n_train = x_train.shape[0]
    if cfg.mode == "Quick":
        n_use = _cap_n(cfg.max_train_samples, n_train)
        if n_use < n_train:
            idx = np.random.choice(n_train, size=n_use, replace=False)
            x_train = x_train[idx]
            y_train = y_train[idx]

    loss_fn = nn.CrossEntropyLoss()

    if cfg.optimizer == "SGD":
        opt = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist: Dict[str, Any] = {"config": asdict(cfg), "epochs": []}

    best_val = -1.0
    best_state = None
    patience = 2
    bad = 0

    for ep in range(cfg.epochs):
        model.train()
        # shuffle
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        total_loss = 0.0
        total = 0
        correct = 0

        for i in range(0, x_train.shape[0], cfg.batch_size):
            xb = preprocess.to_tensor_batch(x_train[i:i+cfg.batch_size]).to(device)
            yb = torch.from_numpy(y_train[i:i+cfg.batch_size]).long().to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * yb.shape[0]
            total += yb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().item())

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # val
        model.eval()
        vtotal = 0
        vcorrect = 0
        with torch.no_grad():
            for i in range(0, x_val.shape[0], cfg.batch_size):
                xb = preprocess.to_tensor_batch(x_val[i:i+cfg.batch_size]).to(device)
                yb = torch.from_numpy(y_val[i:i+cfg.batch_size]).long().to(device)
                logits = model(xb)
                vcorrect += int((logits.argmax(dim=1) == yb).sum().item())
                vtotal += yb.shape[0]
        val_acc = vcorrect / max(1, vtotal)

        hist["epochs"].append({"epoch": ep+1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})

        # early stop
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist