import torch
from torch import Tensor

class Diffusion:

    def __init__(self, T: int = 1000, device: torch.device = torch.device("cpu")) -> None:
        self.T = T
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

    def p_sample(self, model_out: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)

        mean = (1.0 / alpha_t.sqrt()) * (x_t - ((1.0 - alpha_t) / (1.0 - a_bar).sqrt()) * model_out)

        if t[0].item() == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + beta_t.sqrt() * noise

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, n: int, labels: Tensor, image_size: int = 28, channels: int = 1) -> Tensor:
        """Генерирует n изображений заданных классов (labels)."""
        model.eval()
        x = torch.randn(n, channels, image_size, image_size, device=self.device)

        for step in reversed(range(self.T)):
            t = torch.full((n,), step, dtype=torch.long, device=self.device)
            model_out = model(x, t, labels)
            x = self.p_sample(model_out, x, t)

        return x.clamp(-1.0, 1.0)