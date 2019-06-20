from functools import partial

import numpy as np
import torch

from nablasim.mnist_loader import get_mnist_iterators
from nablasim.engine import Engine, run_experiment
from nablasim.losses import DoubleNablaSimLoss, DoubleNLLLoss
from nablasim.metrics import AccumulatedAccuracyMetric
from nablasim.models import InnerConvNet

# pylint: disable=invalid-name


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # global variables
    data_path = "./data"
    BATCH_SIZE = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_CLASSES = 10

    # iterators
    global_train_iter, global_valid_iter = get_mnist_iterators(data_path, BATCH_SIZE, DEVICE)
    PartialEngine = partial(
        Engine,
        global_train_iter=global_train_iter,
        global_valid_iter=global_valid_iter,
        device=DEVICE,
        n_epochs=1,
        print_period=100,
        checkpoint_period=100,
    )

    # run
    specs = [
        {
            "name": "baseline",
            "engine_cls": partial(
                PartialEngine,
                inner_net=partial(InnerConvNet, n_classes=N_CLASSES),
                losses={"scores_loss": DoubleNLLLoss(1.0)},
                metric_cls=AccumulatedAccuracyMetric)
        },
        {
            "name": "nablasim",
            "engine_cls": partial(
                PartialEngine,
                inner_net=partial(InnerConvNet, n_classes=N_CLASSES),
                losses={
                    "scores_loss": DoubleNLLLoss(1.0),
                    "gradients_loss": DoubleNablaSimLoss(BATCH_SIZE**.5, mask_kind="posnegflat")},
                metric_cls=AccumulatedAccuracyMetric)
        },
    ]
    run_experiment(specs, json_dir="./runs/MNIST", n_curves=60, max_batch_idx=512)
