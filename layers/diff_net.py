import abc
import lpips
import functools
import numpy as np
from random import sample
from scipy import integrate
import torch
from torch import nn
from torch.nn import functional as F
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP

from layers import *

@register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""
    def __init__(self, config, feat_dim):
        super().__init__()
        self.config = config

        """Get activation functions from the config file."""
        if config.activation.lower() == 'elu':
            self.act = nn.ELU()
        elif config.activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif config.activation.lower() == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif config.activation.lower() == 'swish':
            self.act = nn.SiLU()
        else:
            raise NotImplementedError('activation function does not exist!')
        
        self.register_buffer('sigmas', torch.tensor(np.exp(
          np.linspace(np.log(config.sigma[1]), np.log(config.sigma[0]), config.num_scales)
          )))
        
        self.feat_dim = feat_dim
        self.centered = True
        self.num_res_blocks = config.num_res_blocks
        self.attn_resolutions = config.attn_resolutions
        ch_mult = config.ch_mult
        self.num_resolutions = len(ch_mult)
        all_resolutions = [config.seq_len // (2 ** i) for i in range(self.num_resolutions)]

        self.conditional = config.is_conditional  # noise-conditional
        self.resblock_type = config.resblock_type.lower()
        self.embedding_type = config.embedding_type.lower()
        assert self.embedding_type in ['fourier', 'positional']

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            assert "Fourier features are only used for continuous training."
            modules.append(GaussianFourierProjection(
              embedding_size=feat_dim, scale=16
            ))
            embed_dim = 2 * feat_dim
        elif self.embedding_type == 'positional':
            embed_dim = feat_dim
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            modules.append(nn.Linear(embed_dim, feat_dim * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(feat_dim * 4, feat_dim * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        attnblock = functools.partial(AttnBlockpp,
                                      init_scale=0., skip_rescale=True)
        upsample = functools.partial(Upsample,
                                    with_conv=True, fir=False, fir_kernel=[1, 3, 3, 1])
        downsample = functools.partial(Downsample,
                                   with_conv=True, fir=False, fir_kernel=[1, 3, 3, 1])
        if self.resblock_type == 'ddpm':
            resnetblock = functools.partial(ResnetBlockDDPMpp,
                                          act=self.act,
                                          dropout=config.dropout,
                                          init_scale=0.,
                                          skip_rescale=True,
                                          temb_dim=nf * 4)
        elif self.resblock_type == 'biggan':
            resnetblock = functools.partial(ResnetBlockBigGANpp,
                                          act=self.act,
                                          dropout=config.dropout,
                                          fir=False,
                                          fir_kernel=[1, 3, 3, 1],
                                          init_scale=0.,
                                          skip_rescale=True,
                                          temb_dim=feat_dim * 4)
        else:
            raise ValueError(f'resblock type {self.resblock_type} unrecognized.')

        modules.append(conv3x3(feat_dim, feat_dim))
        hs_c = [feat_dim]

        in_ch = feat_dim
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                out_ch = feat_dim * ch_mult[i_level]
                modules.append(resnetblock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in self.attn_resolutions:
                    modules.append(attnblock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    modules.append(downsample(in_ch=in_ch))
                else:
                    modules.append(resnetblock(down=True, in_ch=in_ch))

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(resnetblock(in_ch=in_ch))
        modules.append(attnblock(channels=in_ch))
        modules.append(resnetblock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                out_ch = feat_dim * ch_mult[i_level]
                modules.append(resnetblock(in_ch=in_ch + hs_c.pop(),
                                          out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in self.attn_resolutions:
                modules.append(attnblock(channels=in_ch))

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    modules.append(upsample(in_ch=in_ch))
                else:
                    modules.append(resnetblock(in_ch=in_ch, up=True))
        
        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, feat_dim, init_scale=0.))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = get_timestep_embedding(timesteps, self.feat_dim)
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered: # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                hi = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if hi.shape[-1] in self.attn_resolutions:
                    hi = modules[m_idx](hi)
                    m_idx += 1
                hs.append(hi)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    hi = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    hi = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                hs.append(hi)

        hi = hs[-1]
        hi = modules[m_idx](hi, temb)
        m_idx += 1
        hi = modules[m_idx](hi)
        m_idx += 1
        hi = modules[m_idx](hi, temb)
        m_idx += 1


        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hi = modules[m_idx](torch.cat([hi, hs.pop()], dim=1), temb)
                m_idx += 1
            
            if hi.shape[-1] in self.attn_resolutions:
                hi = modules[m_idx](hi)
                m_idx += 1

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    hi = modules[m_idx](hi)
                    m_idx += 1
                else:
                    hi = modules[m_idx](hi, temb)
                    m_idx += 1
        assert not hs

        hi = self.act(modules[m_idx](hi))
        m_idx += 1
        hi = modules[m_idx](hi)
        m_idx += 1

        assert m_idx == len(modules)

        return hi

class BandMatrixKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x, diag=False, **params):
        # Calculate the RBF kernel
        distance = self.covar_dist(x, x, diag=diag, **params)
        K = torch.exp(-0.5 * distance.pow(2) / self.lengthscale)
        # Apply the sparsity pattern by creating a mask
        n = K.size(-2)
        m = K.size(-1)
        mask = torch.eye(n, m, dtype=K.dtype, device=K.device)
        if n > 1:
            for i in range(n - 1):
                mask[i, i + 1] = 1
        K_sparse = K * mask
        
        return K_sparse
        
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, is_linear=False):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if is_linear:
            self.covar_module = BandMatrixKernel()
        else:
            self.covar_module = ScaleKernel(RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return MultivariateNormal(mean_x, covar_x)

class RectifiedFlow():
    def __init__(self, config, reflow_flag=False, sigma_var=0.0, ode_tol=1e-5, sample_N=None):   
        if sample_N is not None:
            self.sample_N = sample_N
            # print('Number of sampling steps:', self.sample_N)
    
        self.device = config.device
        self.init_type = config.init_type
        self.noise_scale = config.init_noise_scale
        self.use_ode_sampler = config.use_ode_sampler
        self.ode_tol = ode_tol
        self.sigma_t = lambda t: (1. - t) * sigma_var
              
        self.reflow_flag = reflow_flag
        if self.reflow_flag:
            self.reflow_t_schedule = config.reflow_t_schedule # t0, t1, uniform, or an integer k > 1
            self.reflow_loss = config.reflow_loss # l2, lpips, lpips+l2
            if 'lpips' in self.reflow_loss:
                self.lpips_model = lpips.LPIPS(net='vgg')
                self.lpips_model = self.lpips_model.cuda()
                for p in self.lpips_model.parameters():
                    p.requires_grad = False

    @property
    def T(self):
        return 1.

    @torch.no_grad()
    def ode(self, init_input, model, reverse=False):
        ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
        rtol=1e-5
        atol=1e-5
        method='RK45'
        eps=1e-3

        # Initial sample
        x = init_input.detach().clone()

        model.eval()
        shape = init_input.shape
        device = init_input.device

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model(x, vec_t*999)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse:
            solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(x),
                                                      rtol=rtol, atol=atol, method=method)
        else:
            solution = integrate.solve_ivp(ode_func, (eps, self.T), to_flattened_numpy(x),
                                      rtol=rtol, atol=atol, method=method)
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        nfe = solution.nfev


        return x

    @torch.no_grad()
    def euler_ode(self, init_input, model, reverse=False, N=100):
        ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
        eps=1e-3
        dt = 1./N

        # Initial sample
        x = init_input.detach().clone()

        model.eval()
        shape = init_input.shape
        device = init_input.device
        
        for i in range(N):  
            num_t = i / N * (self.T - eps) + eps      
            t = torch.ones(shape[0], device=device) * num_t
            pred = model(x, t*999)
            
            x = x.detach().clone() + pred * dt         

        return x

    def get_z0(self, batch, train=True):
        b, k, c = batch.shape 

        if self.init_type == 'gaussian':
            ### standard gaussian #+ 0.5
            cur_shape = (b, k, c)
            return (torch.randn(cur_shape)*self.noise_scale).to(self.device)
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 
    
    def get_gp_z0(self, batch, cond_mask, training_iter=10, train=True):
        b, k, c = batch.shape 

        if self.init_type == 'gaussian':
            batch_inp = torch.arange(0, c, device=batch.device, dtype=torch.float).reshape(1, -1, 1).repeat(k, 1, 1)
            samples = []
            for i in range(b):
                batch_oup = batch[i]
                batch_mask = cond_mask[i]
                train_inp = batch_inp[:, batch_mask.any(dim=0)]
                train_oup = batch_oup[:, batch_mask.any(dim=0)]
                if len(train_inp) != 0:
                    with torch.enable_grad():
                        likelihood = GaussianLikelihood()
                        gpmodel = GPModel(train_inp, train_oup, likelihood).to(self.device)
                        gpmodel.train()
                        likelihood.train()
                        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.2)
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpmodel)
                        for i in range(training_iter):  
                            optimizer.zero_grad()
                            output = gpmodel(train_inp)
                            loss = -mll(output, train_oup).sum()
                            # print(f'loss-{i}:{loss}')
                            loss.backward(retain_graph=True)
                            optimizer.step()
                    gpmodel.eval()
                    likelihood.eval()
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        test_inp = batch_inp
                        observed_pred = likelihood(gpmodel(test_inp))
                        y_mean = observed_pred.mean
                        y_var = observed_pred.variance
                        y_samples = torch.normal(y_mean, y_var.sqrt())
                    del gpmodel, likelihood
                else:
                    y_samples = (torch.randn((k, c))*self.noise_scale).to(self.device)
                samples.append(y_samples)
            return torch.stack(samples)
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 
