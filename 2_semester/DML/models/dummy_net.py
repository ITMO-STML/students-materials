from torch import Tensor, nn


class DummyNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_classes, bias=False)

        self.act = nn.SiLU()

    def forward(self, features: Tensor) -> Tensor:
        out = self.act(self.ln1(self.fc1(features)))
        out = self.act(self.ln2(self.fc2(out)))
        out = self.fc3(out)
        return out
