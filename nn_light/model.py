import torch
import torch.nn as nn

from nn_utils import math_utils
from nn_utils.activations import to_hdr_activation
from nn_utils.film_siren_layers import FiLMSiren
from nn_utils.initializers import SIRENFirstLayerInitializer, SIRENInitializer, apply_initializer
from nn_utils.nerf_layers import FourierEmbedding


class ConditionalMappingNet(nn.Module):
    def __init__(self, args):
        super(ConditionalMappingNet, self).__init__()
        units = 1 if args.single_offset else args.net_width

        self.embedder = FourierEmbedding(args.cond_frequency, input_dim=1)

        self.mapping_net = nn.Sequential(
            nn.Linear(self.embedder.get_output_dimensionality(), args.cond_width),
            nn.ELU(),
            nn.Linear(args.cond_width, units * 2),
        )

    def forward(self, conditional):
        x = self.embedder(conditional)
        params = self.mapping_net(x)
        phase_shift, frequency = params[:, 0], params[:, 1]
        return phase_shift, frequency


class FiLMMappingNet(nn.Module):
    def __init__(self, args):
        super(FiLMMappingNet, self).__init__()

        units = 1 if args.single_offset else args.net_width
        output_units = units * args.net_depth

        self.latent_dim = args.latent_dim

        layers = [nn.Linear(self.latent_dim, args.mapping_width), nn.ELU()]
        for _ in range(args.mapping_depth - 1):
            layers.append(nn.Linear(args.mapping_width, args.mapping_width))
            layers.append(nn.ELU())
        layers.append(nn.Linear(args.mapping_width, output_units * 2))

        self.mapping_net = nn.Sequential(*layers)

    def forward(self, latent_vector):
        params = self.mapping_net(latent_vector)
        phase_shift, frequency = params[:, 0], params[:, 1]
        return phase_shift, frequency


class MainNetwork(nn.Module):
    def __init__(self, args):
        super(MainNetwork, self).__init__()

        self.film_layer = FiLMSiren(w0=1)

        self.net = nn.ModuleList()
        input_size = 3
        for i in range(args.net_depth):
            layer = nn.Linear(input_size, args.net_width, bias=True)
            initializer = SIRENFirstLayerInitializer(0.5) if i == 0 else SIRENInitializer(w0=2)
            apply_initializer(layer, initializer)
            self.net.append(layer)
            input_size = args.net_width

        self.conditional_layer = nn.Linear(input_size, args.net_width)
        apply_initializer(self.conditional_layer, SIRENInitializer(w0=2))

        self.final_layer = nn.Linear(input_size, 3)

    def forward(self, direction, main_params, conditional_params):
        main_phase_shift, main_frequency = main_params
        conditional_phase_shift, conditional_frequency = conditional_params

        x = direction
        for i, layer in enumerate(self.net):
            fre = main_frequency[:, i]
            shif = main_phase_shift[:, i]
            x = layer(x)
            x = self.film_layer(x, fre, shif)

        x = self.conditional_layer(x)
        x = self.film_layer(x, conditional_frequency, conditional_phase_shift)

        x = self.final_layer(x)
        x = to_hdr_activation(x)

        return x


class FiLMIlluminationNetwork(nn.Module):
    def __init__(self, args):
        super(FiLMIlluminationNetwork, self).__init__()

        self.latent_units = args.latent_dim

        self.main_network = MainNetwork(args)
        self.mapping_network = FiLMMappingNet(args)
        self.conditional_network = ConditionalMappingNet(args)

    def forward(self, direction, conditional, latent):
        main_params = self.mapping_network(latent)
        conditional_params = self.conditional_network(conditional)
        return self.main_network(direction, main_params, conditional_params)

    def forward_multi_samples(self, direction, conditional, latent):
        latent_samples = latent.unsqueeze(1).expand(-1, direction.size(1), -1)

        latent_flat = latent_samples.view(-1, latent.size(-1))
        directions_flat = direction.view(-1, 3)
        cond_flat = conditional.view(-1, 1)

        recon_flat = self.forward(directions_flat, cond_flat, latent_flat)

        recon_shape_restored = recon_flat.view(
            latent.size(0),
            direction.size(1) if direction.size(1) is not None else -1,
            recon_flat.size(-1),
        )

        return recon_shape_restored

    def eval_env_map(self, latent, conditional, img_height=128):
        uvs = math_utils.shape_to_uv(img_height, img_height * 2)  # H, W, 2
        directions = math_utils.uv_to_direction(uvs)  # H, W, 3
        directions_flat = directions.view(-1, 3)

        directions_batched = directions_flat.unsqueeze(0).expand(latent.size(0), -1, -1)
        directions_batched = directions_batched.float()

        cond_batched = torch.ones_like(directions_batched[..., :1]) * conditional
        latent = latent.float()

        recon = self.forward_multi_samples(directions_batched, cond_batched, latent)

        recon_shape_restore = recon.view(latent.size(0), img_height, img_height * 2, 3)

        return recon_shape_restore

    def eval_env_map_multi_rghs(self, latent, roughness_levels, img_height=128):
        import numpy as np

        rghs = np.linspace(0, 1, roughness_levels)

        ret = []

        for r in rghs:
            ret.append(self.eval_env_map(latent, r, img_height=img_height))

        return ret


class Args:
    def __init__(self):
        self.single_offset = False
        self.net_width = 128
        self.cond_frequency = 10
        self.cond_width = 64
        self.mapping_width = 128
        self.mapping_depth = 2
        self.net_depth = 2
        self.latent_dim = 128


args = Args()


# 1. 测试 ConditionalMappingNet
def test_conditional_mapping_net():
    model = ConditionalMappingNet(args)
    print("Testing ConditionalMappingNet...")

    # 创建一个随机的条件输入
    conditional_input = torch.randn(2, 1)  # 假设批次大小为 2，输入维度为 1
    phase_shift, frequency = model(conditional_input)

    print(f"Phase Shift Shape: {phase_shift.shape}")
    print(f"Frequency Shape: {frequency.shape}")
    assert phase_shift.shape == torch.Size([2]), f"Expected shape [2], but got {phase_shift.shape}"
    assert frequency.shape == torch.Size([2]), f"Expected shape [2], but got {frequency.shape}"


# 2. 测试 FiLMMappingNet
def test_film_mapping_net():
    model = FiLMMappingNet(args)
    print("Testing FiLMMappingNet...")

    # 创建一个随机的潜在向量输入
    latent_input = torch.randn(2, args.latent_dim)  # 假设批次大小为 2，latent_dim 为 128
    phase_shift, frequency = model(latent_input)

    print(f"Phase Shift Shape: {phase_shift.shape}")
    print(f"Frequency Shape: {frequency.shape}")
    assert phase_shift.shape == torch.Size([2]), f"Expected shape [2], but got {phase_shift.shape}"
    assert frequency.shape == torch.Size([2]), f"Expected shape [2], but got {frequency.shape}"


# 3. 测试 MainNetwork
def test_main_network():
    model = MainNetwork(args)
    print("Testing MainNetwork...")

    # 创建随机的方向和主参数、条件参数输入
    direction = torch.randn(3)
    # NOTE: FiLM-Siren块每层实际上要的是两个标量，但是原论文代码里用的是两个net_width的张量，共2 * net_depth个，得找找原来的实现细节
    # TODO: 如果是标量填充或其他处理，为什么不做进FiLM块里呢
    # TODO: 不对啊，按照原实现，batch_size放哪了（
    main_params = (torch.randn(args.net_width, args.net_depth), torch.randn(args.net_width, args.net_depth))  # 假设输出为 (phase_shift, frequency)
    conditional_params = (torch.randn(args.net_width), torch.randn(args.net_width))  # 同样的 shape

    output = model(direction, main_params, conditional_params)

    print(f"Output Shape: {output.shape}")
    assert output.shape == torch.Size([3]), f"Expected shape [2, 3], but got {output.shape}"


# 4. 测试 FiLMIlluminationNetwork
def test_film_illumination_network():
    model = FiLMIlluminationNetwork(args)
    print("Testing FiLMIlluminationNetwork...")

    # 创建随机的输入
    direction = torch.randn(2, 3)  # 假设批次大小为 2，方向维度为 3
    conditional = torch.randn(2, 1)  # 假设批次大小为 2，条件输入维度为 1
    latent = torch.randn(2, args.latent_dim)  # 假设批次大小为 2，latent_dim 为 128

    # 调用前向函数
    output = model(direction, conditional, latent)

    print(f"Output Shape: {output.shape}")
    assert output.shape == torch.Size([2, 3]), f"Expected shape [2, 3], but got {output.shape}"


# 5. 测试多样本前向传播
def test_forward_multi_samples():
    model = FiLMIlluminationNetwork(args)
    print("Testing forward_multi_samples...")

    # 创建随机的输入
    direction = torch.randn(2, 10, 3)  # 假设批次大小为 2，图像大小为 10x10，方向维度为 3
    conditional = torch.randn(2, 10, 1)  # 假设条件输入维度为 1
    latent = torch.randn(2, args.latent_dim)  # 假设 latent_dim 为 128

    output = model.forward_multi_samples(direction, conditional, latent)

    print(f"Output Shape: {output.shape}")
    assert output.shape == torch.Size([2, 10, 10, 3]), f"Expected shape [2, 10, 10, 3], but got {output.shape}"


# 6. 测试环境映射（单一粗糙度）
def test_eval_env_map():
    model = FiLMIlluminationNetwork(args)
    print("Testing eval_env_map...")

    latent = torch.randn(2, args.latent_dim)  # 假设批次大小为 2，latent_dim 为 128
    conditional = 0.5  # 设定一个粗糙度值

    output = model.eval_env_map(latent, conditional)

    print(f"Output Shape: {output.shape}")
    assert output.shape == torch.Size([2, 128, 256, 3]), f"Expected shape [2, 128, 256, 3], but got {output.shape}"


# 7. 测试多粗糙度环境映射
def test_eval_env_map_multi_rghs():
    model = FiLMIlluminationNetwork(args)
    print("Testing eval_env_map_multi_rghs...")

    latent = torch.randn(2, args.latent_dim)  # 假设批次大小为 2，latent_dim 为 128
    roughness_levels = 5  # 设置粗糙度层级数

    output = model.eval_env_map_multi_rghs(latent, roughness_levels)

    print(f"Output Length: {len(output)}")
    assert len(output) == roughness_levels, f"Expected output length {roughness_levels}, but got {len(output)}"
    assert output[0].shape == torch.Size(
        [2, 128, 256, 3]), f"Expected shape [2, 128, 256, 3], but got {output[0].shape}"


# 调用测试函数
test_conditional_mapping_net()
test_film_mapping_net()
test_main_network()
test_film_illumination_network()
test_forward_multi_samples()
test_eval_env_map()
test_eval_env_map_multi_rghs()

