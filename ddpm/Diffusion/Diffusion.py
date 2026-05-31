
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision.transforms
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))   # out.shape:[1]


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):  # beta_1 0.0001, beta_T 0.02, T 1000
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))


    def forward(self, x_0, labels, train=True):
        """
        Algorithm 1.
        """

        noise = torch.randn_like(x_0)
        save_image(x_0, 'base.png', nrow=4, padding=1)
        # print(self.sqrt_alphas_bar, self.sqrt_alphas_bar)
        if train:
            b = x_0.shape[0]
            t0 = torch.randint(1, size=(b,), device=x_0.device)
            t = torch.randint(self.T, size=(b,), device=x_0.device)
            # t = torch.randint(10, size=(b,), device=x_0.device)
            x_t = (
                     extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                     extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
            x_t = torch.cat((x_0[:int(b / 2)], x_t[int(b / 2): b]))
            t = torch.cat((t0[:int(b / 2)], t[int(b / 2): b]))
            # t = t0

        else:
            t = torch.randint(1, size=(x_0.shape[0],), device=x_0.device)
            x_t = x_0

        predict_noise, feature_vector = self.model(x_t, t, labels=labels)
        loss = F.mse_loss(predict_noise, noise, reduction='none')
        return loss, feature_vector, t


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, PEDCC_feature_vector=None, labels=None):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        if PEDCC_feature_vector is not None:
            # add PEDCC , eps is a List[noise, feature_vector]
            if labels is not None:
                eps = self.model(x_t, t, PEDCC_feature_vector=PEDCC_feature_vector, labels=labels)
                nonEps = self.model(x_t, t, PEDCC_feature_vector=PEDCC_feature_vector,
                                    labels=torch.zeros_like(labels).to(labels.device))
                ### In the classifier free guidence paper, w is the key to control the gudience.
                ### w = 0 and with label = 0 means no guidence.
                ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
                # w = 1.8
                eps = (1. + 1.8) * eps[0] - 1.8 * nonEps[0], eps[1]
            else:
                eps = self.model(x_t, t, PEDCC_feature_vector=PEDCC_feature_vector)
        elif labels is not None:
            eps = self.model(x_t, t, PEDCC_feature_vector=None, labels=labels)
            nonEps = self.model(x_t, t, PEDCC_feature_vector=None,
                                labels=torch.zeros_like(labels).to(labels.device))
            # w = 1.8
            eps = (1. + 1.8) * eps[0] - 1.8 * nonEps[0], eps[1]


        else:
            eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps[0])

        return xt_prev_mean, var, eps[1]

    def forward(self, x_T, PEDCC_feature_vector=None, labels=None):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            if PEDCC_feature_vector is not None:
                if labels is not None:
                    mean, var, latent_feature = self.p_mean_variance(x_t=x_t, t=t, PEDCC_feature_vector=PEDCC_feature_vector,
                                                     labels=labels)
                else:
                    mean, var, latent_feature = self.p_mean_variance(x_t=x_t, t=t, PEDCC_feature_vector=PEDCC_feature_vector)

            elif labels is not None:
                mean, var, latent_feature = self.p_mean_variance(x_t=x_t, t=t, PEDCC_feature_vector=PEDCC_feature_vector,
                                                 labels=labels)
            else:
                mean, var, latent_feature = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
                # classify
                from Diffusion.Train import latent_variable_dim, class_num, CosineLinear_PEDCC
                fc = CosineLinear_PEDCC(latent_variable_dim, class_num)
                output = fc(latent_feature)  # out [b 10]
                assert output.shape == (x_t.size(0), class_num)
                _, pred_label = output.max(1)

            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1), pred_label


