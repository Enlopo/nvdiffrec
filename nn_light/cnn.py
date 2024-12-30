import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nn_utils.activations import from_hdr_activation, to_hdr_activation
from nn_utils.coord_conv import AddCoords


class CnnEncoder(nn.Module):
    def __init__(
            self,
            output_units: int,
            activation="elu",
            embedding_function=None,
            img_height: int = 128,
    ):
        super(CnnEncoder, self).__init__()

        self.latent_dim = output_units
        self.activation = getattr(F, activation)
        self.embedding_activation = embedding_function if embedding_function else nn.Identity()

        # HDR预处理
        self.from_hdr_layer = from_hdr_activation  # HDR env map转SDR
        self.add_coords1 = AddCoords()
        # AddCoords 添加2个通道，所以输入通道数为 3 + 2
        self.conv1 = nn.Conv2d(in_channels=3 + 2, out_channels=8, kernel_size=3, stride=1, padding=1)

        # 计算下采样次数
        down_conv_needed = int(np.log2(img_height)) // 2

        # 构建下采样和卷积层
        self.coord_conv_blocks = nn.ModuleList()
        self.down_sampling_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        prev_nf = 8
        for i in range(down_conv_needed):
            half_output = self.latent_dim // 2
            cur_nf = int((half_output / down_conv_needed) * (i + 1))

            # AddCoords -> Conv2D -> AddCoords -> Conv2D
            self.coord_conv_blocks.append(AddCoords())
            self.down_sampling_blocks.append(
                nn.Conv2d(
                    in_channels=prev_nf + 2,  # AddCoords 添加2个通道，所以输入通道数为 prev_nf + 2
                    out_channels=cur_nf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.coord_conv_blocks.append(AddCoords())
            self.conv_blocks.append(
                nn.Conv2d(
                    in_channels=cur_nf + 2,  # AddCoords 添加2个通道，所以输入通道数为 curNf + 2
                    out_channels=cur_nf,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            prev_nf = cur_nf

        # Flatten 层
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc = nn.Linear(prev_nf * (img_height // (2 ** down_conv_needed)) ** 2, self.latent_dim)

    def forward(self, img_data):
        # from_hdr_activation
        x = self.from_hdr_layer(img_data)

        # AddCoords -> Conv2D
        x = self.add_coords1(x)
        x = self.conv1(x)
        x = self.activation(x)

        # 多层下采样和卷积
        for i in range(len(self.down_sampling_blocks)):
            x = self.coord_conv_blocks[2 * i](x)
            x = self.down_sampling_blocks[i](x)
            x = self.activation(x)
            x = self.coord_conv_blocks[2 * i + 1](x)
            x = self.conv_blocks[i](x)
            x = self.activation(x)

        # Flatten
        x = self.flatten(x)

        # 全连接层
        x = self.fc(x)
        x = self.embedding_activation(x)

        return x


class CnnDecoder(nn.Module):
    def __init__(
            self,
            input_units: int,
            output_units: int,
            activation="elu",
            output_activation=None,
            img_height: int = 128,
    ):
        super(CnnDecoder, self).__init__()

        # 获取激活函数
        self.activation = getattr(F, activation)
        self.output_activation = output_activation if output_activation else to_hdr_activation

        # 初始通道数
        initial_nf = input_units // 2

        # 第一层: Reshape 和 Conv2DTranspose (恢复纵横比)
        self.input_layer = nn.ConvTranspose2d(
            in_channels=input_units,
            out_channels=initial_nf,
            kernel_size=(1, 2),
            stride=1,
            padding=0,
        )

        # 计算上采样次数
        up_conv_needed = int(np.log2(img_height))

        # 构建上采样和卷积层
        self.up_conv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        prev_nf = initial_nf

        for i in range(up_conv_needed):
            cur_nf = initial_nf - int((initial_nf / (up_conv_needed + 3)) * (i + 1))

            # Conv2DTranspose (上采样)
            self.up_conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=prev_nf,
                    out_channels=cur_nf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

            # Conv2D (特征提取)
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=cur_nf,
                    out_channels=cur_nf,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            prev_nf = cur_nf

        # 最后一层 Conv2D
        self.final_conv = nn.Conv2d(
            in_channels=prev_nf,
            out_channels=output_units,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # 将输入重塑为 (batch_size, input_units, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)  # 添加两个维度，匹配 (B, C, H, W)

        # 第一个转置卷积层
        x = self.input_layer(x)
        x = self.activation(x)

        # 逐层执行上采样和卷积
        for up_conv, conv in zip(self.up_conv_layers, self.conv_layers):
            x = up_conv(x)
            x = self.activation(x)
            x = conv(x)
            x = self.activation(x)

        # 最后一层卷积
        x = self.final_conv(x)
        x = self.output_activation(x)

        return x


class CnnDiscriminator(nn.Module):
    def __init__(
            self,
            discriminator_units: int = 32,
            activation="relu",
            img_height: int = 128,
    ):
        super(CnnDiscriminator, self).__init__()

        self.activation = getattr(F, activation)

        # HDR预处理
        self.from_hdr_layer = from_hdr_activation  # HDR env map转SDR

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 计算下采样次数
        down_conv_needed = int(np.log2(img_height)) - 3

        self.coord_conv_blocks = nn.ModuleList()
        self.down_sampling_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(down_conv_needed):
            # AddCoords -> Conv2D -> Conv2D
            self.coord_conv_blocks.append(AddCoords())  # 第一个 AddCoords
            self.down_sampling_blocks.append(
                nn.Conv2d(
                    in_channels=10 if i == 0 else discriminator_units + 2,
                    out_channels=discriminator_units,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.conv_blocks.append(
                nn.Conv2d(
                    in_channels=discriminator_units,
                    out_channels=discriminator_units,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        # Flatten 层
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc = nn.Linear(
            discriminator_units * (img_height // (2 ** down_conv_needed)) ** 2, 1
        )

    def forward(self, img_data):
        # Lambda 层: from_hdr_activation
        x = self.from_hdr_layer(img_data)

        # 第一层卷积
        x = self.conv1(x)
        x = self.activation(x)

        # 多层下采样和卷积
        for i in range(len(self.down_sampling_blocks)):
            x = self.coord_conv_blocks[i](x)
            x = self.down_sampling_blocks[i](x)
            x = self.activation(x)
            x = self.conv_blocks[i](x)
            x = self.activation(x)

        # Flatten
        x = self.flatten(x)

        # 全连接层
        log_its = self.fc(x)

        return log_its


def test_models():
    # 设定输入张量的形状
    img_height = 128
    # TODO: tf和torch的输入张量形状不同，其他抄来的地方记得改
    # TODO: 要不要加上对非正方形图像的支持
    input_tensor = torch.ones(1, 3, img_height, img_height)  # torch输入图像的大小为 (B, C, W, H)

    latent_dim = 128  # latent_dim 的大小
    output_units = 64  # decoder 输出的单元数

    # 创建 Encoder, Decoder 和 Discriminator
    encoder = CnnEncoder(output_units=latent_dim, activation="elu", img_height=img_height)
    decoder = CnnDecoder(input_units=latent_dim, output_units=3, activation="elu", img_height=img_height)
    discriminator = CnnDiscriminator(discriminator_units=32, activation="relu", img_height=img_height)

    # 测试 Encoder
    print("Testing Encoder...")
    encoder_output = encoder(input_tensor)
    print(f"Encoder output shape: {encoder_output.shape}")  # 期望是 (batch_size, latent_dim)

    # 测试 Decoder
    print("Testing Decoder...")
    decoder_output = decoder(encoder_output)
    print(f"Decoder output shape: {decoder_output.shape}")  # 期望是 (batch_size, 3, img_height, img_height*2)

    # 测试 Discriminator
    print("Testing Discriminator...")
    discriminator_output = discriminator(input_tensor)
    print(f"Discriminator output shape: {discriminator_output.shape}")  # 期望是 (batch_size, 1)


if __name__ == "__main__":
    # 运行测试函数
    test_models()
