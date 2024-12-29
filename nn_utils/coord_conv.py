import torch
import torch.nn as nn


def test_add_coords():
    input_tensor = torch.tensor([
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
            [[25.0, 26.0, 27.0], [28.0, 29.0, 30.0], [31.0, 32.0, 33.0], [34.0, 35.0, 36.0]],
            [[37.0, 38.0, 39.0], [40.0, 41.0, 42.0], [43.0, 44.0, 45.0], [46.0, 47.0, 48.0]]
        ],
        [[[49.0, 50.0, 51.0], [52.0, 53.0, 54.0], [55.0, 56.0, 57.0], [58.0, 59.0, 60.0]],
         [[61.0, 62.0, 63.0], [64.0, 65.0, 66.0], [67.0, 68.0, 69.0], [70.0, 71.0, 72.0]],
         [[73.0, 74.0, 75.0], [76.0, 77.0, 78.0], [79.0, 80.0, 81.0], [82.0, 83.0, 84.0]],
         [[85.0, 86.0, 87.0], [88.0, 89.0, 90.0], [91.0, 92.0, 93.0], [94.0, 95.0, 96.0]]]
    ])
    add_coords = AddCoords()
    output_tensor = add_coords(input_tensor)

    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)
    print("Output Tensor:", output_tensor)
    assert output_tensor.shape[-1] == input_tensor.shape[
        -1] + 2, "Output tensor should have 2 additional channels for coords"
    print("Test passed: AddCoords module is working as expected.")


class AddCoords(nn.Module):
    """Add coords to a tensor"""

    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        input_tensor: (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
        In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
        In the second case, skip tile, just concat
        """
        batch_size, x_dim, y_dim, _ = input_tensor.shape
        xx_ones = torch.ones((batch_size, x_dim, 1), dtype=torch.int32, device=input_tensor.device)
        xx_range = torch.arange(y_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0).repeat(batch_size, 1)
        xx_channel = torch.matmul(xx_ones, xx_range.unsqueeze(1))
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones((batch_size, y_dim, 1), dtype=torch.int32, device=input_tensor.device)
        yy_range = torch.arange(x_dim, dtype=torch.int32, device=input_tensor.device).unsqueeze(0).repeat(batch_size, 1)
        yy_channel = torch.matmul(yy_range.unsqueeze(-1), yy_ones.transpose(1, 2))
        yy_channel = yy_channel.unsqueeze(-1)

        x_dim = float(x_dim)
        y_dim = float(y_dim)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=-1)

        return ret


if __name__ == '__main__':
    test_add_coords()
