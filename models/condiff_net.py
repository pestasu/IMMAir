import numpy as np
from random import sample
from scipy import integrate
import torch
from torch import nn
from torch.nn import functional as F


from layers import Group, RectifiedFlow, NCSNpp

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, pred, real):
        p = F.softmax(pred, dim=1)
        q = F.softmax(real, dim=1)

        log_p = torch.log(p + 1e-10) 
        log_q = torch.log(q + 1e-10)

        kl = torch.sum(p * (log_p - log_q), dim=1)
        return kl.mean()  

class cdr_layer(nn.Module):
    def __init__(self, config, feat_dim_a, feat_dim_m, feat_dim_o):
        super().__init__()
        self.device = config.device
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.train_mode = config.train_mode
        self.use_gaussian = config.use_gaussian
        self.loss_fn = KLDivergence() if config.rec_type == 'disb' else MSE()
        self.reduce_op = torch.mean if config.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        # Initialize RectifiedFlow for each modality
        self.reflow = {
            'a': RectifiedFlow(config, reflow_flag=True),
            'm': RectifiedFlow(config, reflow_flag=True),
            'o': RectifiedFlow(config, reflow_flag=True)
        }

        self.score = {
            'a': NCSNpp(config, feat_dim=feat_dim_a),
            'm': NCSNpp(config, feat_dim=feat_dim_m),
            'o': NCSNpp(config, feat_dim=feat_dim_o)
        }

        # transformation layers
        self.cat_m2a = nn.Conv1d(feat_dim_a * 3, feat_dim_a, kernel_size=1, padding=0)
        self.cat_o2a = nn.Conv1d(feat_dim_m * 3, feat_dim_m, kernel_size=1, padding=0)
        self.cat_a2m = nn.Conv1d(feat_dim_o * 3, feat_dim_o, kernel_size=1, padding=0)
        self.cat_o2m = nn.Conv1d(feat_dim_a * 3, feat_dim_a, kernel_size=1, padding=0)
        self.cat_a2o = nn.Conv1d(feat_dim_m * 2, feat_dim_m, kernel_size=1, padding=0)
        self.cat_m2o = nn.Conv1d(feat_dim_o * 2, feat_dim_o, kernel_size=1, padding=0)

        self.cat_a = nn.Conv1d(feat_dim_a * 2, feat_dim_a, kernel_size=1, padding=0)
        self.cat_m = nn.Conv1d(feat_dim_m * 2, feat_dim_m, kernel_size=1, padding=0)
        self.cat_o = nn.Conv1d(feat_dim_o * 2, feat_dim_o, kernel_size=1, padding=0)

        # reconstruction layers
        self.rec_a = nn.Sequential(
            nn.Conv1d(feat_dim_a, feat_dim_a*2, 1),
            Group(num_channels=feat_dim_a*2, num_blocks=config.n_block, reduction=config.reduction),
            nn.Conv1d(feat_dim_a*2, feat_dim_a, 1)
        )
        self.rec_m = nn.Sequential(
            nn.Conv1d(feat_dim_m, feat_dim_m*2, 1),
            Group(num_channels=feat_dim_m*2, num_blocks=config.n_block, reduction=config.reduction),
            nn.Conv1d(feat_dim_m*2, feat_dim_m, 1)
        )
        self.rec_o = nn.Sequential(
            nn.Conv1d(feat_dim_o, feat_dim_o*2, 1),
            Group(num_channels=feat_dim_o*2, num_blocks=config.n_block, reduction=config.reduction),
            nn.Conv1d(feat_dim_o*2, feat_dim_o, 1)
        )
    
    def get_rf_pair(self, proj_z, phi_z, M, is_train, eps=1e-3):
        if self.reflow[M].reflow_flag:
            if self.reflow[M].reflow_t_schedule=='t0': ### distill for t = 0 (k=1)
                t = torch.zeros(proj_z.shape[0], device=self.device) * (self.reflow[M].T - eps) + eps
            elif self.reflow[M].reflow_t_schedule=='t1': ### reverse distill for t=1 (fast embedding)
                t = torch.ones(proj_z.shape[0], device=self.device) * (self.reflow[M].T - eps) + eps
            elif self.reflow[M].reflow_t_schedule=='uniform': ### train new rectified flow with reflow
                t = torch.rand(proj_z.shape[0], device=self.device) * (self.reflow[M].T - eps) + eps
            elif type(self.reflow[M].reflow_t_schedule)==int: ### k > 1 distillation
                t = torch.randint(0, self.reflow[M].reflow_t_schedule, (proj_z.shape[0], ), device=self.device) * (self.reflow[M].T - eps) / self.reflow[M].reflow_t_schedule + eps
            else:
                assert False, 'Not implemented'
        else:### standard rectified flow loss
            t = torch.rand(proj_z.shape[0], device=self.device) * (self.reflow[M].T - eps) + eps

        t_expand = t.view(-1, 1, 1).repeat(1, proj_z.shape[1], proj_z.shape[2])

        if is_train:
            inferproj_z = t_expand * proj_z + (1.-t_expand) * phi_z
        else: 
            inferproj_z = phi_z
            t = torch.zeros(proj_z.shape[0], device=self.device)

        target = proj_z - phi_z

        score = self.score[M](inferproj_z, t*999) 
        
        return target, score

    def get_rectified_flow_loss(self, target, score):

        losses = torch.square(score - target)

        losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        return torch.mean(losses)

    def get_randmask(self, observed_mask, K):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio->(0, 1)
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        cond_mask = cond_mask.unsqueeze(1).repeat(1, K, 1)
        return cond_mask

    def random_modality_missing(self, proj_x_a, proj_x_m, proj_x_o, obs_a, obs_m, obs_o, 
                                z_a_0, z_m_0, z_o_0, cond_mask_a, cond_mask_m, cond_mask_o, num_modal=None, is_train=True):
        
        inferproj_x_a, inferproj_x_m, inferproj_x_o = proj_x_a, proj_x_m, proj_x_o
        modal_idx = ['a', 'm', 'o']  # (0:aqi, 1:weather, 2:outphoto)
        ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
        
        if num_modal == 1:  # one modality is available
            if ava_modal_idx[0] == 'a':  # has aqi
                z_a2m_0 = self.cat_a2m(torch.cat((proj_x_a, obs_m, z_m_0), 1))
                z_a2o_0 = self.cat_a2o(torch.cat((proj_x_a[...,-proj_x_o.shape[-1]:], z_o_0), 1))
                target_a2m, score_a2m = self.get_rf_pair(proj_x_m, z_m_0, z_a2m_0, 'm', is_train)
                target_a2o, score_a2o = self.get_rf_pair(proj_x_o, z_o_0, z_a2o_0, 'o', is_train)
                loss_a2m = self.get_rectified_flow_loss(target_a2m, score_a2m) if is_train else 0.
                loss_a2o = self.get_rectified_flow_loss(target_a2o, score_a2o) if is_train else 0.
                inferproj_x_m = self.rec_m((z_a2m_0 + score_a2m).detach())
                inferproj_x_m = inferproj_x_m*(1-cond_mask_m)+obs_m
                inferproj_x_o = self.rec_o((z_a2o_0 + score_a2o).detach())
                loss_trans = (loss_a2m + loss_a2o)/ 2
                loss_rec = (self.loss_fn(inferproj_x_m, proj_x_m.detach()) + self.loss_fn(inferproj_x_o, proj_x_o.detach())) / 2
            elif ava_modal_idx[0] == 'm':  # has weather
                z_m2a_0 = self.cat_m2a(torch.cat((proj_x_m, obs_a, z_a_0), 1))
                z_m2o_0 = self.cat_m2o(torch.cat((proj_x_m[...,-proj_x_o.shape[-1]:], z_o_0), 1))
                target_m2a, score_m2a = self.get_rf_pair(proj_x_a, z_m2a_0, 'a', is_train)
                target_m2o, score_m2o = self.get_rf_pair(proj_x_o, z_m2o_0, 'o', is_train)
                loss_m2a = self.get_rectified_flow_loss(target_m2a, score_m2a) if is_train else 0.
                loss_m2o = self.get_rectified_flow_loss(target_m2o, score_m2o) if is_train else 0.
                inferproj_x_a = self.rec_a((z_m2a_0 + score_m2a).detach())
                inferproj_x_a = inferproj_x_a*(1-cond_mask_a)+obs_a
                inferproj_x_o = self.rec_o((z_m2o_0 + score_m2o).detach())
                loss_trans = (loss_m2a + loss_m2o)/ 2
                loss_rec = (self.loss_fn(inferproj_x_a, proj_x_a.detach()) + self.loss_fn(inferproj_x_o, proj_x_o.detach())) / 2
            else:  # has outphoto
                z_o2a_0 = self.cat_o2a(torch.cat((proj_x_o.repeat(1,1,proj_x_a.shape[-1]), obs_a, z_a_0), 1))
                z_o2m_0 = self.cat_o2m(torch.cat((proj_x_o.repeat(1,1,proj_x_m.shape[-1]), obs_m, z_m_0), 1))            
                target_o2a, score_o2a = self.get_rf_pair(proj_x_a, z_o2a_0, 'a', is_train)
                target_o2m, score_o2m = self.get_rf_pair(proj_x_m, z_o2m_0, 'm', is_train)
                loss_o2a = self.get_rectified_flow_loss(target_o2a, score_o2a) if is_train else 0.
                loss_o2m = self.get_rectified_flow_loss(target_o2m, score_o2m) if is_train else 0.
                inferproj_x_a = self.rec_a((z_o2a_0 + score_o2a).detach())
                inferproj_x_a = inferproj_x_a*(1-cond_mask_a)+obs_a
                inferproj_x_m = self.rec_m((z_o2m_0 + score_o2m).detach())
                inferproj_x_m = inferproj_x_m*(1-cond_mask_m)+obs_m
                loss_trans = (loss_o2a + loss_o2m)/ 2
                loss_rec = (self.loss_fn(inferproj_x_a, proj_x_a.detach()) + self.loss_fn(inferproj_x_m, inferproj_x_m.detach())) / 2
        if num_modal == 2:  # two modalities are available
            if set(modal_idx)-set(ava_modal_idx) == {'a'}:  # aqi is missing (weather,outphoto available)
                z_m2a_0 = self.cat_m2a(torch.cat((proj_x_m, obs_a, z_a_0), 1))
                z_o2a_0 = self.cat_o2a(torch.cat((proj_x_o.repeat(1,1,proj_x_a.shape[-1]), obs_a, z_a_0), 1))
                target_m2a, score_m2a = self.get_rf_pair(proj_x_a, z_m2a_0, 'a', is_train)
                target_o2a, score_o2a = self.get_rf_pair(proj_x_a, z_o2a_0, 'a', is_train)
                loss_m2a = self.get_rectified_flow_loss(target_m2a, score_m2a) if is_train else 0.
                loss_o2a = self.get_rectified_flow_loss(target_o2a, score_o2a) if is_train else 0.
                inferproj_x_a = self.cat_a(torch.cat([(z_m2a_0 + score_m2a).detach(), (z_o2a_0 + score_o2a).detach()], dim=1))
                inferproj_x_a = self.rec_a(inferproj_x_a)
                inferproj_x_a = inferproj_x_a*(1-cond_mask_a)+obs_a
                loss_trans = (loss_m2a + loss_o2a)/ 2
                loss_rec = self.loss_fn(inferproj_x_a, proj_x_a.detach())
            if set(modal_idx)-set(ava_modal_idx) == {'m'}:  # weather is missing (aqi,outphoto available)
                z_a2m_0 = self.cat_a2m(torch.cat((proj_x_a, obs_m, z_m_0), 1))
                z_o2m_0 = self.cat_o2m(torch.cat((proj_x_o.repeat(1,1,proj_x_m.shape[-1]), obs_m, z_m_0), 1))
                target_a2m, score_a2m = self.get_rf_pair(proj_x_m, z_a2m_0, 'm', is_train)
                target_o2m, score_o2m = self.get_rf_pair(proj_x_m, z_o2m_0, 'm', is_train)
                loss_a2m = self.get_rectified_flow_loss(target_a2m, score_a2m) if is_train else 0.
                loss_o2m = self.get_rectified_flow_loss(target_o2m, score_o2m) if is_train else 0.
                inferproj_x_m = self.cat_m(torch.cat([(z_a2m_0 + score_a2m).detach(), (z_o2m_0 + score_o2m).detach()], dim=1))
                inferproj_x_m = self.rec_m(inferproj_x_m)
                inferproj_x_m = inferproj_x_m*(1-cond_mask_m)+obs_m
                loss_trans = (loss_a2m + loss_o2m)/ 2
                loss_rec = self.loss_fn(inferproj_x_m, proj_x_m.detach())
            if set(modal_idx)-set(ava_modal_idx) == {'o'}:  # outphoto is missing (aqi,weather available)
                z_a2o_0 = self.cat_a2o(torch.cat((proj_x_a[...,-proj_x_o.shape[-1]:], z_o_0), 1))
                z_m2o_0 = self.cat_m2o(torch.cat((proj_x_m[...,-proj_x_o.shape[-1]:], z_o_0), 1))
                target_a2o, score_a2o = self.get_rf_pair(proj_x_o, z_a2o_0, 'o', is_train)
                target_m2o, score_m2o = self.get_rf_pair(proj_x_o, z_m2o_0, 'o', is_train)
                loss_a2o = self.get_rectified_flow_loss(target_a2o, score_a2o) if is_train else 0.
                loss_m2o = self.get_rectified_flow_loss(target_m2o, score_m2o) if is_train else 0.
                inferproj_x_o = self.cat_o(torch.cat([(z_a2o_0 + score_a2o).detach(), (z_m2o_0 + score_m2o).detach()], dim=1))
                inferproj_x_o = self.rec_o(inferproj_x_o)
                loss_trans = (loss_a2o + loss_m2o)/ 2
                loss_rec = self.loss_fn(inferproj_x_o, proj_x_o.detach())
        if num_modal == 3:  # no missing
            loss_trans = torch.tensor(0)
            loss_rec = torch.tensor(0)
        
        return inferproj_x_a, inferproj_x_m, inferproj_x_o, loss_trans, loss_rec, ava_modal_idx
    
    def forward(self, proj_x_a, proj_x_m, proj_x_o, num_modal, is_train=True):
        
        observed_mask_a = torch.ones((proj_x_a.shape[0], proj_x_a.shape[-1])).to(proj_x_a.device)
        observed_mask_m = torch.ones((proj_x_m.shape[0], proj_x_m.shape[-1])).to(proj_x_m.device)
        observed_mask_o = torch.zeros_like(proj_x_o).to(proj_x_o.device)

        cond_mask_a = self.get_randmask(observed_mask_a, proj_x_a.shape[1])
        obs_a = proj_x_a*cond_mask_a
        cond_mask_m = self.get_randmask(observed_mask_m, proj_x_m.shape[1])
        obs_m = proj_x_m*cond_mask_m
        cond_mask_o = observed_mask_o
        obs_o = observed_mask_o

        if not self.use_gaussian:
            z_a_0 = self.reflow_a.get_z0(proj_x_a)
            z_m_0 = self.reflow_m.get_z0(proj_x_m)
            z_o_0 = self.reflow_o.get_z0(proj_x_o)
        else:
            z_a_0 = self.reflow_a.get_gp_z0(proj_x_a, cond_mask_a)
            z_m_0 = self.reflow_m.get_gp_z0(proj_x_m, cond_mask_m)
            z_o_0 = self.reflow_o.get_gp_z0(proj_x_o, cond_mask_o)

        #  select modality
        inferproj_x_a, inferproj_x_m, inferproj_x_o, loss_trans, loss_rec, ava_modal_idx = \
            self.random_modality_missing(proj_x_a, proj_x_m, proj_x_o, obs_a, obs_m, obs_o, 
                                z_a_0, z_m_0, z_o_0, cond_mask_a, cond_mask_m, cond_mask_o, num_modal, is_train)

        inferproj_x_a = inferproj_x_a.permute(0, 2, 1)
        inferproj_x_m = inferproj_x_m.permute(0, 2, 1)
        inferproj_x_o = inferproj_x_o.permute(0, 2, 1)

        return inferproj_x_a, inferproj_x_m, inferproj_x_o, loss_trans, loss_rec, ava_modal_idx
