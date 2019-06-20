import os

import torch


def load_torch_checkpoint(pth_path):
    """Load an arbitrary dict used as a pytorch checkpoint."""
    assert pth_path[-4:] == ".pth"
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Invalid path. The .pth file does not exist: '{pth_path}'")

    return torch.load(pth_path)


def save_torch_checkpoint(data, pth_path, overwrite=True):
    """Save an arbitrary dict used as a pytorch checkpoint."""
    assert pth_path[-4:] == ".pth"
    if overwrite:
        if os.path.isfile(pth_path):
            os.remove(pth_path)
    else:
        if os.path.isfile(pth_path):
            raise FileExistsError(f"Invalid path. The .pth file already exists: '{pth_path}'")

    pth_path_part = pth_path + ".part"
    torch.save(data, pth_path_part)
    os.rename(pth_path_part, pth_path)
