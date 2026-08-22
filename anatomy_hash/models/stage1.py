from __future__ import annotations
import torch
from torch import nn
import timm


class Stage1AnatomyClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name="convnext_tiny.fb_in1k", pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits, labels, max_iter=100):
    scaler = TemperatureScaler().to(logits.device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(scaler.temperature.detach().cpu())
