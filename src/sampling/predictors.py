import abc

import torch


# class Predictor(abc.ABC):
#     """The abstract class for a predictor algorithm."""

#     def __init__(self, sde, score_fn, probability_flow=False):
#         super().__init__()
#         self.sde = sde
#         self.rsde = sde.reverse(score_fn)
#         self.score_fn = score_fn
#         self.probability_flow = probability_flow

#     @abc.abstractmethod
#     def update_fn(self, x, t, *args):
#         """One update of the predictor.

#         Args:
#             x: A PyTorch tensor representing the current state
#             t: A Pytorch tensor representing the current time step.
#             *args: Possibly additional arguments, in particular `y` for OU processes

#         Returns:
#             x: A PyTorch tensor of the next state.
#             x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
#         """
#         pass

#     def debug_update_fn(self, x, t, *args):
#         raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


class ReverseDiffusionPredictor():
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    def update_fn(self, x, t, *args):
        # print("=====predictor=====")
        f, g = self.rsde.discretize(x, t, *args)
        z = torch.randn_like(x)

        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean
    