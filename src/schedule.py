from abc import ABC, abstractmethod

import torch

class AlphaSchedule(ABC):
    def __init__(self, max_t):
        self.max_t = max_t

    @abstractmethod
    def __call__(self, t: torch.Tensor):
        pass
    
    @abstractmethod
    def derivative(self, t: torch.Tensor, continuous=True):
        pass

class LinearAlphaSchedule(AlphaSchedule):
    def __call__(self, t: torch.Tensor):
        return 1.0 - t / self.max_t
    def derivative(self, t: torch.Tensor, _continuous=True):
        return torch.tensor(-1.0, device=t.device) / self.max_t
    
class CosineAlphaSchedule(AlphaSchedule):
    def __call__(self, t: torch.Tensor):
        return torch.cos((t / self.max_t) * (torch.pi / 2))
    def derivative(self, t: torch.Tensor, continuous=True):
        if continuous:
            return - torch.pi / 2 * torch.sin((t / self.max_t) * (torch.pi / 2))

        return - (torch.pi / (2 * self.max_t)) * torch.sin((t / self.max_t) * (torch.pi / 2))


class SigmaSchedule(ABC):
    def __init__(self, max_t):
        self.max_t = max_t

    @abstractmethod
    def __call__(self, t: torch.Tensor):
        pass
    
    @abstractmethod
    def derivative(self, t: torch.Tensor):
        pass

class LinearSigmaSchedule(SigmaSchedule):
    def __call__(self, t: torch.Tensor):
        return t / self.max_t
    def derivative(self, t: torch.Tensor, _continuous=True):
        return torch.tensor(1.0, device=t.device) / self.max_t
    
class CosineSigmaSchedule(SigmaSchedule):
    def __call__(self, t: torch.Tensor):
        return torch.sin((t / self.max_t) * (torch.pi / 2))
    def derivative(self, t: torch.Tensor, continuous=True):
        if continuous:
            return torch.pi / 2 * torch.cos((t / self.max_t) * (torch.pi / 2))

        return (torch.pi / (2 * self.max_t)) * torch.cos((t / self.max_t) * (torch.pi / 2))


class LambdaSchedule:
    alpha: AlphaSchedule
    sigma: SigmaSchedule

    def __init__(self, alpha_schedule, sigma_schedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule

    def __call__(self, t: torch.Tensor):
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        return torch.log((alpha / sigma) ** 2)

    def derivative(self, t: torch.Tensor, continuous=True):
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        d_alpha = self.alpha.derivative(t, continuous=continuous)
        d_sigma = self.sigma.derivative(t, continuous=continuous)
        return 2 * (d_alpha / alpha - d_sigma / sigma)

class SchedulePack:
    alpha: AlphaSchedule
    sigma: SigmaSchedule
    lambda_: LambdaSchedule

    def __init__(self, alpha_schedule, sigma_schedule):
        self.alpha = alpha_schedule
        self.sigma = sigma_schedule
        self.lambda_ = LambdaSchedule(alpha_schedule, sigma_schedule)