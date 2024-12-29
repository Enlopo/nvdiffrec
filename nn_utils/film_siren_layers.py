import torch
import torch.nn as nn
import math


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, inputs):
        return torch.sin(self.w0 * inputs)

    def get_config(self):
        return {'w0': self.w0}


class FiLMSiren(nn.Module):
    def __init__(self, w0=30.0):
        super(FiLMSiren, self).__init__()
        self.sine = Sine(w0=w0)

    def forward(self, tensor, frequency, phase_shift):
        return self.sine(tensor * frequency + phase_shift)


# Test functions
def test_film_siren_layers():
    sine_layer = Sine(w0=20.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # Constant tensor for testing
    print('Sine Output:', sine_layer(x))
    filmsiren_layer = FiLMSiren(w0=15.0)
    print('FiLMSiren Output:', filmsiren_layer(x, frequency=2, phase_shift=math.pi))


if __name__ == "__main__":
    test_film_siren_layers()
