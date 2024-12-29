import torch
import math
import torch.nn as nn


class ConfigInitializer:
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def get_config(self):
        base_config = super().get_config()
        config = {"w0": self.w0, "c": self.c, "main_fan_in": self.main_fan_in}
        return dict(list(base_config.items()) + list(config.items()))


class ScaledHyperInitializer:
    def __init__(self, scale=1, seed=None):
        self.scale = scale
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, shape):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
        limit = math.sqrt(6 / fan_in) * self.scale
        return torch.empty(shape).uniform_(-limit, limit)


class HyperSirenFirstLayerInitializer:
    def __init__(self, main_fan_in, scale=1.0, seed=None):
        self.main_fan_in = main_fan_in
        self.scale = scale
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, shape):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
        main_limit = math.sqrt(3 / fan_in)
        siren_limit = self.scale / max(1.0, fan_in)
        limit = main_limit * siren_limit
        return torch.empty(shape).uniform_(-limit, limit)


class HyperSirenInitializer:
    def __init__(self, main_fan_in, w0=30.0, c=6.0, seed=None):
        self.main_fan_in = main_fan_in
        self.w0 = w0
        self.c = c
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, shape):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
        main_limit = math.sqrt(3 / fan_in)
        siren_limit = math.sqrt(self.c / max(1.0, self.main_fan_in)) / self.w0
        limit = main_limit * siren_limit
        return torch.empty(shape).uniform_(-limit, limit)


class HyperInitializer:
    def __init__(self, main_fan, seed=None):
        self.main_fan = main_fan
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, shape):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
        limit = math.sqrt(3 * 2 / (fan_in * self.main_fan))
        return torch.empty(shape).uniform_(-limit, limit)


class SIRENFirstLayerInitializer:
    def __init__(self, scale=1.0, seed=None):
        self.scale = scale
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, shape):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
        limit = self.scale / max(1.0, fan_in)
        return torch.empty(shape).uniform_(-limit, limit)


class SIRENInitializer:
    def __init__(self, w0=30.0, c=6.0, seed=None):
        self.w0 = w0
        self.c = c
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, shape):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
        limit = math.sqrt(self.c / max(1.0, fan_in)) / self.w0
        return torch.empty(shape).uniform_(-limit, limit)


# Test Initializers
def test_initializers():
    shape = (3, 3)
    seed = 1
    print("ScaledHyperInitializer:", ScaledHyperInitializer(scale=0.5, seed=seed)(shape))
    print("HyperSirenFirstLayerInitializer:", HyperSirenFirstLayerInitializer(64, scale=1.0, seed=seed)(shape))
    print("HyperSirenInitializer:", HyperSirenInitializer(64, seed=seed)(shape))
    print("HyperInitializer:", HyperInitializer(64, seed=seed)(shape))
    print("SIRENFirstLayerInitializer:", SIRENFirstLayerInitializer(scale=1.0, seed=seed)(shape))
    print("SIRENInitializer:", SIRENInitializer(seed=seed)(shape))


if __name__ == '__main__':
    test_initializers()
