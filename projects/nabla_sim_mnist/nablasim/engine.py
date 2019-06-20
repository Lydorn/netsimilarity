import datetime as dt
import json
import os
from collections.abc import Iterator
from timeit import default_timer as timer

import numpy as np
import torch
from torch import optim

from nablasim.checkpointer import save_torch_checkpoint
from nablasim.models import DoubleNet


def run_experiment(specs, json_dir, n_curves=1, max_batch_idx=None):
    run_dt = dt.datetime.now()
    for curve_idx in range(n_curves):
        for spec in specs:
            # init
            name, engine_cls = spec["name"], spec["engine_cls"]
            json_paths = [
                f"{json_dir}/{run_dt:%H_%M_%S}/{name}_{curve_idx:02d}.json",
                f"{json_dir}/last/{name}_{curve_idx:02d}.json"]
            print(f"<<<<<<<< {name} {curve_idx + 1}/{n_curves} >>>>>>>>")

            # run
            engine = engine_cls(json_paths=json_paths)
            tic = timer()
            engine.run(max_batch_idx=max_batch_idx)
            print(f"{timer() - tic:.3f}s\n")


class Engine:

    def __init__(self, global_train_iter, global_valid_iter, device,
                 inner_net, losses, metric_cls,
                 n_epochs, print_period, checkpoint_period, json_paths):
        self.global_train_iter = global_train_iter
        self.global_valid_iter = global_valid_iter

        assert device in ["cpu", "cuda"]
        self.device = device
        self.model = DoubleNet(inner_net=inner_net()).to(device)
        self.losses = losses
        self.metric = metric_cls()
        self.metric_valid = metric_cls()

        self.params = list(self.model.parameters())
        self.optimizer = optim.Adam(self.params)

        self.n_epochs = n_epochs
        self.print_period = print_period
        self.checkpoint_period = checkpoint_period

        self.json_paths = json_paths
        for json_path in self.json_paths:
            if not os.path.exists(os.path.dirname(json_path)):
                os.makedirs(os.path.dirname(json_path))
            if os.path.exists(json_path):
                os.remove(json_path)

    def run(self, max_batch_idx=None):
        counter = 0
        convergence_curve = []
        loss_vals, metric_vals = [], []
        self.model.train()

        for epoch_idx in range(self.n_epochs):
            epoch_train_iter = next(self.global_train_iter)
            self.metric.reset()

            tic = timer()
            for batch_idx, batch_data in enumerate(epoch_train_iter):
                inputs, targets = batch_data["inputs"], batch_data["targets"]
                counter += 1

                # train
                self.optimizer.zero_grad()
                outputs, scores = self.model(*inputs)
                loss = self.losses["scores_loss"](scores, targets)
                if "gradients_loss" in self.losses:
                    loss += self.losses["gradients_loss"](self.params, outputs, targets)
                loss.backward()
                self.optimizer.step()

                # accumulate losses and metrics
                loss_vals.append(loss.item())
                metric_vals.append(self.metric(scores, targets))
                convergence_curve.append(self.metric.get_accumulated_value())

                # checkpoint
                if counter == 1 or counter % self.checkpoint_period == 0:
                    save_torch_checkpoint(
                        self.model.state_dict(),
                        pth_path=self.json_paths[0][:-5] + "_model.pth")
                    save_torch_checkpoint(
                        self.optimizer.state_dict(),
                        pth_path=self.json_paths[0][:-5] + "_optim.pth")

                # validate
                vl_acc = self.run_validation()
                print(str(batch_idx) + "\r", end="")

                # print to stdout
                if counter == 1 or counter % self.print_period == 0:
                    iter_len = max_batch_idx if max_batch_idx is not None else batch_data["length"]
                    msg = ""
                    msg += f"epoch_idx:{epoch_idx + 1}/{self.n_epochs}\t|  "
                    msg += f"timer:{timer() - tic:.1f}s\t|  "
                    msg += f"item:{batch_idx + 1}/{iter_len}\t|  "
                    msg += f"progress:{100. * (batch_idx + 1) / iter_len: 3.0f}%  |  "
                    msg += f"loss:{np.mean(loss_vals): 6.3f}  |  "
                    msg += f"tr_acc_local:{np.mean(metric_vals): 8.3f}  |  "
                    msg += f"tr_acc_cumul:{self.metric.get_accumulated_value(): 8.3f}  |  "
                    msg += f"vl_acc:{vl_acc: 8.3f}  |  "
                    print(msg)

                    # reset buffers
                    loss_vals, metric_vals = [], []

                # log to json
                for json_path in self.json_paths:
                    data = {"batch_idx": batch_idx,
                            "loss": np.mean(loss_vals),
                            "tr_acc_local": np.mean(metric_vals),
                            "tr_acc_cumul": self.metric.get_accumulated_value(),
                            "vl_acc": vl_acc}
                    with open(json_path, "at") as file:
                        file.write(json.dumps(data, indent=None, separators=(", ", ":")) + "\n")

                # exit early if necessary
                if max_batch_idx is not None and batch_idx == max_batch_idx:
                    break

        return convergence_curve

    def run_validation(self):
        gv_iter = self.global_valid_iter
        self.model.eval()
        with torch.no_grad():
            self.metric_valid.reset()
            epoch_valid_iter = next(gv_iter) if isinstance(gv_iter, Iterator) else gv_iter
            for _, (inputs, targets) in enumerate(epoch_valid_iter):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                scores = self.model.inner_net(inputs)[1]
                self.metric_valid([scores], [targets])  # accumulate
                break
        self.model.train()
        return self.metric_valid.get_accumulated_value()
