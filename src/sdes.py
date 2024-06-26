"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import warnings

import numpy as np
import torch

class OUVESDE():
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=1000, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--gamma", type=float, default=1.5, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        parser.add_argument("--Alpha", type=float, default=0.03, help="param control the expansion saturation time, 0 < alpha <= 1")
        parser.add_argument("--Lambda", type=float, default=0, help="param control the expansion start point, 0 <= lambda")
        return parser

    def __init__(self, t_eps, gamma, sigma_min, sigma_max, alpha=0.03, lambda_=10, N=1000, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -gamma (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            gamma: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__()
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.alpha = alpha
        self.lambda_ = lambda_
        self.t_eps = t_eps
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N

    def copy(self):
        return OUVESDE(self.t_eps, self.gamma, self.sigma_min, self.sigma_max, self.alpha, self.lambda_, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, t, y):
        drift = self.gamma * (y - x)

        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig )
        # * exp
        return drift, diffusion

    def _mean(self, x0, t, y):
        gamma = self.gamma
        exp_interp = torch.exp(-gamma * t)[:, None, None, None]

        return self.mask((exp_interp * x0 + (1 - exp_interp) * y), t)

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, gamma, logsig = self.sigma_min, self.gamma, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * gamma * t)
                * (torch.exp(2 * (gamma + logsig) * t) - 1)
                * logsig
            )
            /
            (gamma + logsig)
        )
    
    def band_step(self, timestep, bias, t_eps):
        return torch.log(torch.tensor(10+bias-((timestep-t_eps)*(1/(1-t_eps))*9)))
        # return torch.ones_like(torch.tensor(timestep))
    
    def mask(self, spec, t, bias=None):
        if bias is None:
            bias = torch.zeros_like(t).to(torch.int32)
        mask = torch.ones_like(spec)
        _,_,f,_ = mask.size()
        
        band_step = self.band_step(t, self.lambda_, self.t_eps)*(1/self.band_step(self.alpha, self.lambda_, self.t_eps))
        band_step = torch.asarray([1 if b > 1 else b for b in band_step ])

        for i, (bia, b) in enumerate(zip(bias, band_step)):
            freq_ceil = int(torch.ceil(f*(1+b)/2)) + bia
            mask[i, :,freq_ceil:,:]=0

        return spec*mask
    
    def get_step(self, t):
        band_step = self.band_step(t, self.lambda_, self.t_eps)*(1/self.band_step(self.alpha, self.lambda_, self.t_eps))
        return torch.asarray([1 if b > 1 else b for b in band_step ])
    
    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))

        y = self.mask(y, torch.tensor([1]))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]

        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")

    
    def discretize(self, x, t, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        # print("======================")
        # print(dt)
        # print("======================")
        # mmm
        drift, diffusion = self.sde(x, t, *args)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, *args):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, *args)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, *args) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()