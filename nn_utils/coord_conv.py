import torch
import torch.nn as nn


class AddCoords(nn.Module):
    """Add coords to a tensor"""

    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        input_tensor: (batch_size, channels, height, width)
        Output: tensor with 2 additional channels (xx_channel, yy_channel)
        """
        batch_size, channels, height, width = input_tensor.shape

        # 创建 xx 和 yy 坐标通道
        xx_range = torch.arange(width, dtype=torch.float32, device=input_tensor.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, height, 1)
        yy_range = torch.arange(height, dtype=torch.float32, device=input_tensor.device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, width)

        # 标准化坐标到 [-1, 1] 范围
        xx_channel = xx_range / (width - 1) * 2 - 1  # [batch_size, height, width]
        yy_channel = yy_range / (height - 1) * 2 - 1  # [batch_size, height, width]

        # 添加一个新的通道维度，变成 [batch_size, 1, height, width]
        xx_channel = xx_channel.unsqueeze(1)  # [batch_size, 1, height, width]
        yy_channel = yy_channel.unsqueeze(1)  # [batch_size, 1, height, width]

        # 将 xx_channel 和 yy_channel 与输入张量拼接
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)  # 在 channels 维度上拼接

        return ret


def test_add_coords():
    input_tensor = torch.randn((2, 3, 4, 4), dtype=torch.float32)  # 输入张量的大小为 (batch_size, channels, height, width
    add_coords = AddCoords()
    output_tensor = add_coords(input_tensor)

    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)
    print("Output Tensor:", output_tensor)
    assert output_tensor.shape[1] == input_tensor.shape[
        1] + 2, "Output tensor should have 2 additional channels for coords"
    print("Test passed: AddCoords module is working as expected.")


if __name__ == '__main__':
    test_add_coords()
