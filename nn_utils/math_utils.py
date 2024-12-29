import torch
import math
import torch.nn.functional as F

EPS = 1e-7


def ev100_to_exp(ev100):
    maxL = 1.2 * torch.pow(2.0, ev100)
    return torch.maximum(1.0 / torch.maximum(maxL, torch.tensor(EPS, device=ev100.device)), torch.ones_like(maxL) * EPS)


def repeat(x, n, axis):
    repeat_dims = [1] * x.dim()
    repeat_dims[axis] = n
    return x.repeat(*repeat_dims)


def saturate(x, low=0.0, high=1.0):
    return torch.clamp(x, low, high)


def mix(x, y, a):
    a = torch.clamp(a, min=0, max=1)
    return x * (1 - a) + y * a


def fill_like(x, val):
    return torch.ones_like(x) * val


def background_compose(x, y, mask):
    mask_clip = saturate(mask)
    return x * mask_clip + (1.0 - mask_clip) * y


def white_background_compose(x, mask):
    return background_compose(x, torch.ones_like(x), mask)


def srgb_to_linear(x):
    x = saturate(x)
    switch_val = 0.04045
    return torch.where(x >= switch_val, torch.pow((torch.clamp(x, min=switch_val) + 0.055) / 1.055, 2.4), x / 12.92)


def linear_to_srgb(x):
    x = saturate(x)
    switch_val = 0.0031308
    return torch.where(x >= switch_val, 1.055 * torch.pow(torch.clamp(x, min=switch_val), 1.0 / 2.4) - 0.055, x * 12.92)


def safe_sqrt(x):
    return torch.sqrt(torch.maximum(x, torch.tensor(EPS, device=x.device)))


def dot(x, y):
    return torch.sum(x * y, dim=-1, keepdim=True)


def magnitude(x):
    return safe_sqrt(dot(x, x))


def normalize(x):
    magn = magnitude(x)
    return torch.where(magn <= safe_sqrt(torch.tensor(0.0)), torch.zeros_like(x), x / magn)


def cross(x, y):
    return torch.cross(x, y, dim=-1)


def reflect(d, n):
    return d - 2 * dot(d, n) * n


def spherical_to_cartesian(theta, phi, r=1):
    x = r * torch.sin(phi) * torch.sin(theta)
    y = r * torch.cos(phi)
    z = r * torch.sin(phi) * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_spherical(vec):
    x, y, z = vec[..., 0:1], vec[..., 1:2], vec[..., 2:3]
    r = magnitude(vec)
    theta = torch.atan2(x, z)
    theta = torch.where(theta > 0, theta, 2 * math.pi + theta)
    theta = torch.fmod(theta, 2 * math.pi - EPS)
    phi = torch.acos(torch.clamp(y / r, -1, 1))
    return theta, phi, r


def uncharted2_tonemap_partial(x: torch.Tensor):
    A = 0.15
    B = 0.50
    C = 0.10
    D = 0.20
    E = 0.02
    F = 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def uncharted2_filmic(v: torch.Tensor):
    exposure_bias = 2.0
    curr = uncharted2_tonemap_partial(v * exposure_bias)
    W = 11.2
    white_scale = 1.0 / uncharted2_tonemap_partial(W)
    return curr * white_scale


def safe_expm1(x: torch.Tensor):
    return torch.expm1(torch.clamp(x, max=87.5))


def l2Norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * x, dim=-1, keepdim=True)


def spherical_to_uv(spherical: torch.Tensor) -> torch.Tensor:
    theta = torch.clamp(spherical[..., 0], min=0, max=2 * math.pi - EPS)
    phi = torch.clamp(spherical[..., 1], min=0, max=math.pi)
    u = theta / (2 * math.pi)
    v = phi / math.pi
    return torch.abs(torch.stack([u, v], dim=-1))


def direction_to_uv(d: torch.Tensor) -> torch.Tensor:
    theta, phi, r = cartesian_to_spherical(d)
    return spherical_to_uv(torch.stack([theta, phi], dim=-1))


def direction_to_uv(d: torch.Tensor) -> torch.Tensor:
    theta, phi, r = cartesian_to_spherical(d)
    spherical = torch.stack([theta, phi], dim=-1)
    return spherical_to_uv(spherical)


def uv_to_spherical(uvs: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uvs[..., 0], min=0, max=1)
    v = torch.clamp(uvs[..., 1], min=0, max=1)
    theta = torch.clamp(2 * u * math.pi, min=0, max=2 * math.pi - EPS)
    phi = torch.clamp(math.pi * v, min=0, max=math.pi)
    return torch.stack([theta, phi], dim=-1)


def uv_to_direction(uvs: torch.Tensor) -> torch.Tensor:
    spherical = uv_to_spherical(uvs)
    theta = spherical[..., 0:1]
    phi = spherical[..., 1:2]
    return spherical_to_cartesian(theta, phi)


def shape_to_uv(height: int, width: int) -> torch.Tensor:
    us, vs = torch.meshgrid(
        torch.linspace(
            0.0 + 0.5 / width,
            1.0 - 0.5 / width,
            width
        ),
        torch.linspace(
            0.0 + 0.5 / height,
            1.0 - 0.5 / height,
            height
        ),
        indexing='xy'
    )
    return torch.stack([us, vs], dim=-1).float()


# Test functions
def test_math_utils():
    mock_1_3_tensor_x = torch.tensor([[1.0, 2.0, 3.0]])
    mock_1_3_tensor_y = torch.tensor([[4.0, 5.0, 6.0]])
    mock_1_tensor = torch.tensor([0.5])
    mock_2_tensor = torch.tensor([1.0, 2.0])
    mock_1_3_tensor_z = torch.tensor([[1.0, 0.0, 0.0]])
    print("uncharted2_tonemap_partial:", uncharted2_tonemap_partial(mock_1_3_tensor_x))
    print("uncharted2_filmic:", uncharted2_filmic(mock_1_3_tensor_x))
    print("safe_expm1:", safe_expm1(mock_1_3_tensor_x))
    print("l2Norm:", l2Norm(mock_1_3_tensor_x))
    print("spherical_to_uv:", spherical_to_uv(torch.tensor([[math.pi, math.pi / 2]])))
    print("uv_to_spherical:", uv_to_spherical(torch.tensor([[0.5, 0.5]])))
    print("shape_to_uv:", shape_to_uv(4, 4))
    print("dot:", dot(mock_1_3_tensor_x, mock_1_3_tensor_y))
    print("normalize:", normalize(mock_1_3_tensor_x))
    print("mix:", mix(mock_1_3_tensor_x, mock_1_3_tensor_y, torch.tensor(0.5)))
    print("saturate:", saturate(mock_1_3_tensor_x, 1.5, 2.5))
    print("reflect:", reflect(mock_1_3_tensor_x, mock_1_3_tensor_y))
    print("cross:", cross(mock_1_3_tensor_x, mock_1_3_tensor_y))
    print("ev100_to_exp:", ev100_to_exp(mock_1_tensor))
    print("fill_like:", fill_like(mock_2_tensor, 3.0))
    print("srgb_to_linear:", srgb_to_linear(mock_1_tensor))
    print("linear_to_srgb:", linear_to_srgb(mock_1_tensor))
    print("direction_to_uv:", direction_to_uv(mock_1_3_tensor_z))


if __name__ == '__main__':
    test_math_utils()
