import torch


class NormalizeByKey(object):
    """Normalizes pixel values"""

    def __init__(self, key, mu=0, std=1):
        self.key = key
        self.mu = mu
        self.std = std

    def __call__(self, data):

        try:
            data[self.key] = (data[self.key].float() - self.mu) / self.std
        except AttributeError:
            data[self.key] = (data[self.key] - self.mu) / self.std
        return data


class SwapAxisByKey(object):
    """Normalizes pixel values"""

    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        data[self.key] = data[self.key].transpose(2, 1).transpose(1, 0)
        return data


class ToTensor(object):
    """Performs device conversion to data"""

    def __call__(self, data):
        new_data = {}
        for key, value in data.items():
            new_data[key] = torch.tensor(value, dtype=torch.float)
        return new_data


class ToDevice(object):
    """Performs device conversion to data"""

    def __init__(self, device):
        self.device = device

    def __call__(self, data):
        new_data = {}
        for key, tensor in data.items():
            new_data[key] = tensor.to(self.device)
        return new_data



