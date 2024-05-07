import torch

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
    