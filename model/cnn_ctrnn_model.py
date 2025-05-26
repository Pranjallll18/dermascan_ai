import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CTRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, tau=1.0):
        super(CTRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        dh = (-h + torch.tanh(self.input2h(x) + self.h2h(h))) / self.tau
        h = h + dh
        return h

class CNN_CTRNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(CNN_CTRNN, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC layer
        self.flatten = nn.Flatten()
        self.ctrnn = CTRNNCell(512, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 classes: benign and malignant

    def forward(self, x_seq):
        batch_size, seq_len, C, H, W = x_seq.size()
        h = torch.zeros(batch_size, self.ctrnn.hidden_size).to(x_seq.device)
        for t in range(seq_len):
            x_t = x_seq[:, t]
            features = self.cnn(x_t)
            features = self.flatten(features)
            h = self.ctrnn(features, h)
        return self.classifier(h)
