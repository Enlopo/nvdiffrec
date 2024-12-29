import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierEmbedding(nn.Module):
    def __init__(self, num_frequencies: int, include_input: bool = True, input_dim: int = 3):
        super(FourierEmbedding, self).__init__()
        self.input_dims = input_dim
        self.out_dims = 0
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if include_input:
            self.out_dims += input_dim

        if num_frequencies >= 1:
            frequency_dims = 2 * num_frequencies * input_dim  # 2 (sin, cos) * num_frequencies * num input_dim
            self.out_dims += frequency_dims

        self.scales = torch.tensor(
            [2.0 ** i for i in range(num_frequencies)], dtype=torch.float32
        )

    def forward(self, x):
        assert x.shape[-1] == self.input_dims, f"Channel dimension is {x.shape[-1]} but should be {self.input_dims}"
        x_shape = x.shape
        xb = (x.unsqueeze(-2) * self.scales.view(-1, 1)).reshape(*x_shape[:-1], -1)
        four_feat = torch.sin(torch.cat([xb, xb + math.pi / 2], dim=-1))

        ret = []
        if self.include_input:
            ret.append(x)

        return torch.cat(ret + [four_feat], dim=-1)

    def get_output_dimensionality(self):
        return self.out_dims


class AnnealedFourierEmbedding(nn.Module):
    def __init__(self, num_frequencies, include_input: bool = True, input_dim: int = 3):
        super(AnnealedFourierEmbedding, self).__init__()
        self.input_dims = input_dim
        self.base_embedder = FourierEmbedding(
            num_frequencies, include_input=False, input_dim=input_dim
        )
        self.num_frequencies = num_frequencies
        self.include_input = include_input

    def forward(self, x, alpha):
        embed_initial = self.base_embedder(x)
        embed = embed_initial.reshape(-1, 2, self.num_frequencies, self.input_dims)
        window = self.cosine_easing_window(alpha).reshape(1, 1, -1, 1)
        embed = window * embed
        embed = embed.reshape(embed_initial.shape)

        if self.include_input:
            embed = torch.cat([x, embed], dim=-1)

        return embed

    def cosine_easing_window(self, alpha):
        x = torch.clamp(alpha - torch.arange(0, self.num_frequencies, dtype=torch.float32), 0.0, 1.0)
        return 0.5 * (1 + torch.cos(math.pi * x + math.pi))

    @classmethod
    def calculate_alpha(cls, num_frequencies, step, end_step):
        return (num_frequencies * step) / end_step

    def get_output_dimensionality(self):
        return self.base_embedder.get_output_dimensionality() + (
            self.input_dims if self.include_input else 0
        )


def split_sigma_and_payload(raw: torch.Tensor):
    sigma = raw[..., :1]
    payload = raw[..., 1:]
    return sigma, payload


def volumetric_rendering(
        sigma: torch.Tensor,
        payload: torch.Tensor,
        z_samples: torch.Tensor,
        rays_direction: torch.Tensor,
        payload_to_parameters,
        white_background_parameters=[],
        sigma_activation=F.softplus,
):
    eps = 1e-10
    sigma = sigma[..., 0]
    dists = z_samples[..., 1:] - z_samples[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], eps)], dim=-1)
    dists = dists * torch.linalg.norm(rays_direction[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-sigma_activation(sigma) * dists)
    accum_prod = torch.cumprod(1.0 - alpha + eps, dim=-1)
    accum_prod = torch.cat([torch.ones_like(accum_prod[..., :1]), accum_prod[..., :-1]], dim=-1)
    weights = alpha * accum_prod
    accumulated_weights = torch.sum(weights, dim=-1)
    depth = torch.sum(weights * z_samples, dim=-1)
    disp = accumulated_weights / (depth + eps)
    disparity = torch.where((disp > 0) & (disp < 1 / eps) & (accumulated_weights > eps), disp, 1 / eps)
    payload_dict = payload_to_parameters(payload)
    payload_raymarched = {k: torch.sum(weights[..., None] * v, dim=-2) for k, v in payload_dict.items()}
    for parameter in white_background_parameters:
        payload_raymarched[parameter] = payload_raymarched[parameter] * (1 - accumulated_weights[..., None]) + \
                                        accumulated_weights[..., None]
    payload_raymarched['depth'] = depth
    payload_raymarched['disparity'] = disparity
    payload_raymarched['acc_alpha'] = accumulated_weights
    payload_raymarched['individual_alphas'] = alpha
    return payload_raymarched, weights


# Test cases
def test_nerf_layers():
    x = torch.randn(4, 3)
    embed = FourierEmbedding(num_frequencies=4)(x)
    print("FourierEmbedding Output Shape:", embed.shape)

    annealed_embed = AnnealedFourierEmbedding(num_frequencies=4)(x, alpha=0.5)
    print("AnnealedFourierEmbedding Output Shape:", annealed_embed.shape)

    raw = torch.randn(4, 5)
    sigma, payload = split_sigma_and_payload(raw)
    print("Sigma Shape:", sigma.shape, "Payload Shape:", payload.shape)

    z_samples = torch.linspace(0, 1, 5).repeat(4, 1)
    rays_dir = torch.randn(4, 3)
    payload_raymarched, weights = volumetric_rendering(
        sigma, payload, z_samples, rays_dir, lambda x: {'payload': x}
    )
    print("Payload Raymarched Keys:", payload_raymarched.keys())
    print("Weights Shape:", weights.shape)


if __name__ == "__main__":
    test_nerf_layers()
