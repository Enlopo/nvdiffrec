import torch
import torch.nn.functional as F


def to_hdr_activation(x):
    return torch.exp(F.relu(x)) - 1


def from_hdr_activation(x):
    return torch.log(1 + F.relu(x))


def softplus_1m(x):
    return F.softplus(x - 1)


def padded_sigmoid(x, padding: float, upper_padding=True, lower_padding=True):
    # If padding is positive it can have values from 0-padding < x < 1+padding,
    # if negative 0+padding < x 1-padding
    x = torch.sigmoid(x)
    mult = int(upper_padding) + int(lower_padding)  # Cast bool to int
    x = x * (1 + mult * padding)
    if lower_padding:
        x = x - padding
    return x


# Test functions
def test_activations():
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    print("Input Tensor:", x)
    print("to_hdr_activation:", to_hdr_activation(x))
    print("from_hdr_activation:", from_hdr_activation(x))
    print("softplus_1m:", softplus_1m(x))
    print("padded_sigmoid (padding=0.1):", padded_sigmoid(x, padding=0.1))


if __name__ == '__main__':
    test_activations()
