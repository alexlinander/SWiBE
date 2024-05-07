# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch
import pdb

from .predictors import ReverseDiffusionPredictor
from .correctors import AnnealedLangevinDynamics


__all__ = ['get_sampler']


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    sde, score_fn, y, guide,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=probability_flow)
    corrector = AnnealedLangevinDynamics(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                
                _,_,f,_ = xt.size()
                if guide:
                    current_band = int(torch.ceil(f*(1+sde.get_step(vec_t))/2))
                    next_band = int(torch.ceil(f*(1+sde.get_step(vec_t-1/sde.N))/2))
                    difference = next_band-current_band
                    if difference != 0:
                        xt[:,:,current_band:next_band,:] = xt[:,:,current_band-difference:current_band,:]

                xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                xt, xt_mean = predictor.update_fn(xt, vec_t, y)
            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns
    
    return pc_sampler


def get_pc_sampler_step(
    sde, score_fn, y, guide,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=probability_flow)
    corrector = AnnealedLangevinDynamics(sde, score_fn, snr=snr, n_steps=corrector_steps, guide=guide)

    def pc_sampler():
        """The PC sampler function."""
        x_results = []
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            for i in range(sde.N):
                t = timesteps[i]
                print(t)
                vec_t = torch.ones(y.shape[0], device=y.device) * t

                _,_,f,_ = xt.size()
                if guide:
                    current_band = int(torch.ceil(f*(1+sde.get_step(vec_t))/2))
                    next_band = int(torch.ceil(f*(1+sde.get_step(vec_t-1/sde.N))/2))
                    difference = next_band-current_band
                    if difference != 0:
                        xt[:,:,current_band:next_band,:] = xt[:,:,current_band-difference:current_band,:]                        

                xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                if i+1 != sde.N:
                    x_results.append(xt)
                    x_results.append(y.clone())
                xt, xt_mean = predictor.update_fn(xt, vec_t, y)
                if i+1 != sde.N:
                    x_results.append(xt_mean)
                # _,_,f,l = 

            x_final = xt_mean if denoise else xt
            x_results.append(x_final)
            ns = sde.N * (corrector.n_steps + 1)
            return x_results, ns
    
    return pc_sampler
