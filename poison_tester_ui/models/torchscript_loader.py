from dataclasses import dataclass
import torch

@dataclass
class TorchscriptModelInfo:
    device: str = "cpu"

def load_torchscript_model(path: str):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model, TorchscriptModelInfo(device="cpu")