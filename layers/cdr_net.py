
import math
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy import integrate
_MODELS = {}

class CALayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_du = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, num_channels, reduction, res_scale):
        super().__init__()

        body = [
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
        ]
        body.append(CALayer(num_channels, reduction))

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, num_channels, num_blocks, reduction, res_scale=1.0):
        super().__init__()

        body = list()
        for _ in range(num_blocks):
            body += [RCAB(num_channels, reduction, res_scale)]
        body += [nn.Conv1d(num_channels, num_channels, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls
  #print(cls, name)
  if cls is None:
    return _register
  else:
    return _register(cls)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'rectified_flow':
    sampling_fn = get_rectified_flow_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def get_rectified_flow_sampler(sde, shape, inverse_scaler, device='cuda'):
  """
  Get rectified flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  def euler_sampler(model, z=None):
    """The probability flow ODE sampler with simple Euler discretization.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      model.eval()
      ### Uniform
      dt = 1./sde.sample_N
      eps = 1e-3 # default: 1e-3

      for i in range(sde.sample_N):
        num_t = i /sde.sample_N * (sde.T - eps) + eps
        t = torch.ones(shape[0], device=device) * num_t
        pred = model(x, t*999) ### Copy from models/utils.py 

        # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
        sigma_t = sde.sigma_t(num_t)
        pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.detach().clone())

        x = x.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
      
      x = inverse_scaler(x)
      nfe = sde.sample_N

      return x
  
  def rk45_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=sde.ode_tol
      method='RK45'
      eps=1e-3

      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model.eval()

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model(x, vec_t*999)

        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      x = inverse_scaler(x)
      
      return x, nfe
  

  print('Type of Sampler:', sde.use_ode_sampler)
  
  if sde.use_ode_sampler=='rk45':
      return rk45_sampler
  elif sde.use_ode_sampler=='euler':
      return euler_sampler
  else:
      assert False, 'Not Implemented!'

def default_initializer(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')
    
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)
  
class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_initializer(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 2, 1)
    
def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_initializer(init_scale)(conv.weight.data.shape)
  if bias:
      nn.init.zeros_(conv.bias)
  return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                    dilation=dilation, bias=bias)
  conv.weight.data = default_initializer(init_scale)(conv.weight.data.shape)
  if bias:
      nn.init.zeros_(conv.bias)
  return conv

def naive_downsample_1d(x, factor=2):
  _N, C, L = x.shape
  if L>1:
    x = torch.reshape(x, (_N, C, L // factor, factor))
    x = torch.mean(x, dim=3)
  return x

def naive_upsample_1d(x, factor=2):
  _N, C, L = x.shape
  if L>1:
    x = torch.reshape(x, (-1, C, L, 1))
    x = x.repeat(1, 1, 1, factor)
    x = torch.reshape(x, (-1, C, L * factor))
  return x


def upfirdn1d(input, kernel, up=1, down=1, pad=(0, 0)):
  batch, channels, lens = input.shape

  if up > 1:
      input = F.pad(input, (0, up - 1)).view(batch, channels, -1)

  input = F.pad(input, pad)

  kernel = torch.flip(kernel, [0]).view(1, 1, -1)
  input = input.view(1, -1, lens + sum(pad))
  output = F.conv1d(input, kernel)
  output = output.view(batch, channels, -1)

  if down > 1:
      output = output[:, :, ::down]

  return output

def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
      k /= np.sum(k)
  assert k.ndim == 1
  return k

def upsample_1d(x, k=None, factor=2, gain=1):
  assert isinstance(factor, int) and factor >= 1
  if k is None:
      k = [1] * factor
  k = _setup_kernel(k) * (gain * factor)
  p = len(k) - factor
  return upfirdn1d(x, torch.tensor(k, device=x.device), up=factor, pad=(p // 2, (p + 1) // 2))
  
def downsample_1d(x, k=None, factor=2, gain=1):
  assert isinstance(factor, int) and factor >= 1
  if k is None:
      k = [1] * factor
  k = _setup_kernel(k) * gain
  p = len(k) - factor
  return upfirdn1d(x, torch.tensor(k, device=x.device), down=factor, pad=(p // 2, (p + 1) // 2))

def upsample_conv_1d(x, w, k=None, factor=2, gain=1):
  assert isinstance(factor, int) and factor >= 1

  # Setup filter kernel for 1D
  if k is None:
      k = [1] * factor
  k = _setup_kernel(k) * (gain * factor)
  p = len(k) - factor

  # Transpose weights for 1D convolution
  inC = w.shape[1]
  outC = w.shape[0]
  w = w.view(outC, inC, -1)

  # 1D transposed convolution
  x = F.conv_transpose1d(x, w, stride=factor, padding=0)

  # Apply 1D FIR filter (upfirdn1d)
  return upfirdn1d(x, torch.tensor(k, device=x.device), pad=(p // 2, (p + 1) // 2))

def conv_downsample_1d(x, w, k=None, factor=2, gain=1):
  assert isinstance(factor, int) and factor >= 1

  # Setup filter kernel for 1D
  if k is None:
      k = [1] * factor
  k = _setup_kernel(k) * gain
  p = len(k) - factor

  # 1D convolution
  x = F.conv1d(x, w, stride=factor, padding=0)

  # Apply 1D FIR filter (upfirdn1d)
  return upfirdn1d(x, torch.tensor(k, device=x.device), pad=(p // 2, (p + 1) // 2))

class Conv1d(nn.Module):
  """Conv1d layer with optimal upsampling and downsampling."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
                resample_kernel=(1, 3, 3, 1),
                use_bias=True,
                kernel_init=None):
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel))
    if kernel_init is not None:
        self.weight.data = kernel_init(self.weight.data.shape)
    if use_bias:
        self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel
    self.kernel = kernel
    self.use_bias = use_bias

  def forward(self, x):
    if self.up:
        x = upsample_conv_1d(x, self.weight, k=self.resample_kernel)
    elif self.down:
        x = conv_downsample_1d(x, self.weight, k=self.resample_kernel)
    else:
        x = F.conv1d(x, self.weight, stride=1, padding=self.kernel // 2)

    if self.use_bias:
        x = x + self.bias.reshape(1, -1, 1)

    return x

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')

class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, L = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bcl,bcd->bld', q, k) * (C ** (-0.5))
    w = F.softmax(w, dim=-1)
    h = torch.einsum('bld,bcd->bcl', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv1d_0 = Conv1d(in_ch, out_ch,
                              kernel=3, up=True,
                              resample_kernel=fir_kernel,
                              use_bias=True,
                              kernel_init=default_initializer())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, L = x.shape
    if not self.fir:
        h = F.interpolate(x, scale_factor=2, mode='linear')
        if self.with_conv:
            h = self.Conv_0(h)
    else:
        if not self.with_conv:
            h = upsample_1d(x, self.fir_kernel, factor=2)
        else:
            h = self.Conv1d_0(x)
    return h

class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv1d_0 = Conv1d(in_ch, out_ch,
                              kernel=3, down=True,
                              resample_kernel=fir_kernel,
                              use_bias=True,
                              kernel_init=default_initializer())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, L = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (1, 0))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
    else:
      if not self.with_conv:
        x = downsample_1d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv1d_0(x)

    return x

class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_initializer()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_initializer()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    if self.up:
      if self.fir:
        h = upsample_1d(h, self.fir_kernel, factor=2)
        x = upsample_1d(x, self.fir_kernel, factor=2)
      else:
        h = naive_upsample_1d(h, factor=2)
        x = naive_upsample_1d(x, factor=2)
    elif self.down:
      if self.fir:
        h = downsample_1d(h, self.fir_kernel, factor=2)
        x = downsample_1d(x, self.fir_kernel, factor=2)
      else:
        h = naive_downsample_1d(h, factor=2)
        x = naive_downsample_1d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
