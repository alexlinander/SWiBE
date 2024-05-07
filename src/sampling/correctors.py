import abc
import torch

from src import sdes

# class Corrector(abc.ABC):
#     """The abstract class for a corrector algorithm."""

#     def __init__(self, sde, score_fn, snr, n_steps):
#         super().__init__()
#         self.rsde = sde.reverse(score_fn)
#         self.score_fn = score_fn
#         self.snr = snr
#         self.n_steps = n_steps

#     @abc.abstractmethod
#     def update_fn(self, x, t, *args):
#         """One update of the corrector.

#         Args:
#             x: A PyTorch tensor representing the current state
#             t: A PyTorch tensor representing the current time step.
#             *args: Possibly additional arguments, in particular `y` for OU processes

#         Returns:
#             x: A PyTorch tensor of the next state.
#             x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
#         """
#         pass

class AnnealedLangevinDynamics():
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        if not isinstance(sde, (sdes.OUVESDE,)):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, t, *args):
        n_steps = self.n_steps
        target_snr = self.snr
        std = self.sde.marginal_prob(x, t, *args)[1]

        for _ in range(n_steps):
            grad = self.score_fn(x, t, *args)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2
            x_mean = x + step_size[:, None, None, None] * grad 
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        # return x, x_mean
        return x, step_size[:, None, None, None] * grad