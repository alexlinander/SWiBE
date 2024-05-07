import time
from math import ceil
import warnings
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import numpy as np

from src import sampling
from src.util.inference import evaluate_model
from src.util.other import pad_spec
import pdb


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--guide", "-g", action='store_false', help="Append the high freq part with low freq component")
        return parser

    def __init__(
        self, backbone, sde, guide=True, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
        num_eval_files=20, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        print(kwargs)
        self.dnn = backbone(**kwargs)
        # Initialize SDE
        self.sde = sde(t_eps, **kwargs)
        # Store hyperparams and save them
        self.guide = guide
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, y = batch
        _,_,f,_ = x.size()
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        _, diffusion = self.sde.sde(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
            
        #======Guided expansion=====#
        if self.guide:
            t_ = torch.rand(x.shape[0])
            current_band = self.sde.get_step(t)
            current_band = (torch.ceil(f*(1+current_band)/2)).to(torch.int32)
            band_step = ((f-current_band)*t_).to(torch.int32)
            
            for i,(c,b) in enumerate(zip(current_band, band_step)):
                mean[i,:,c:c+b,:] = mean[i,:,c-b:c,:]
        #===========================#

        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y)
        err = (diffusion[:, None, None, None]**2) * (score + z / sigmas)
        loss = self._loss(err)
        return loss

    def _downsample(self, wav, target_sr):
        
        transform_down = torchaudio.transforms.Resample(16000, target_sr)
        transform_up = torchaudio.transforms.Resample(target_sr, 16000)
        wav_ = transform_down(wav)
        wav_ = transform_up(wav_)

        return wav_

    def training_step(self, batch, batch_idx):
        # now = datetime.now()
        # print("training start =", now)
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # now = datetime.now()
        # print("training end =", now)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi, lsd = evaluate_model(self, self.num_eval_files)

            # print("evaluate model done, Time =", datetime.now().strftime("%H:%M:%S"))

            self.log('pesq', pesq, on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True, sync_dist=True)
            self.log('lsd', lsd, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, y, N=None, minibatch=None, mode=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            if mode is None:
                return sampling.get_pc_sampler(sde=sde, score_fn=self, y=y, guide=self.guide, **kwargs)
            else:
                return sampling.get_pc_sampler_step(sde=sde, score_fn=self, guide=self.guide, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        sampler = self.get_pc_sampler(Y.cuda(), N=N, 
            corrector_steps=corrector_steps, snr=snr, intermediate=False,
            **kwargs)
            
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
        
    def plot(self, model_name, alpha=0.03, gamma=10):
        timesteps = torch.linspace(1, self.t_eps, 30)
        t = range(0,30)
        y_major_loator = plt.MultipleLocator(2)


        plt.figure()
        fig, ax1 = plt.subplots()

        ax1.set_ylabel('value')  # we already handled the x-label with ax1
        ax1.set_xlabel('steps(t)')

        line = self.sde.band_step(timesteps, gamma, self.t_eps)*(1/self.sde.band_step(alpha, gamma, self.t_eps)) 
        line = torch.asarray([1 if b > 1 else b for b in line ])
        # line = line*0.25
        print("bandwidth step :",np.asarray(line))
        ax1.plot(t, line)
        ax1.tick_params(axis='y')
        ax1.spines[['top']].set_visible(False)
        ax1.set_ylim(0,1.0625)
        ax1.grid( linestyle = '--')


        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('bandwidth(k)')  # we already handled the x-label with ax1
        ax2.tick_params(axis='y')
        ax2.spines[['top']].set_visible(False)
        ax2.set_ylim(8,16.5)
        ax2.yaxis.set_major_locator(y_major_loator)

        plt.title(r"$ \alpha $=" + str(alpha) + r"$,\gamma$=" + str(gamma))
        plt.savefig(f"Bstep_plt/{model_name}.png" , dpi=300, format = 'png')
