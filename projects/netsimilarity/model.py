import torch
import torch.nn as nn


class Simple1DInputNet(nn.Module):
    def __init__(self, config, capacity=64):
        super(Simple1DInputNet, self).__init__()
        self.config = config

        module_list = [
            nn.Linear(self.config["input_channel_count"], capacity),
            nn.ReLU()
        ]
        for i in range(5):
            module_list.append(nn.Linear(capacity, capacity))
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(capacity, 1))
        self.net = nn.Sequential(*module_list)

    def forward(self, batch):
        batched_x = batch["x"]
        yb = self.net(batched_x)
        yb = yb.squeeze()
        return yb

    def compute_grads(self, sample, return_pred=False):
        batched_x = sample["x"]
        yb = self.net(batched_x)
        y = yb[0, :]  # Remove batch dim
        # Propagate values of yb seperatly
        d = 1
        all_dim_grads = []
        for i in range(d):
            self.zero_grad()  # Start over
            y[i].backward(retain_graph=True)
            grads = []
            for tensor in self.parameters():
                grad_flat = tensor.grad.view(-1)
                grads.append(grad_flat)
            grads = torch.cat(grads)
            all_dim_grads.append(grads)
        all_dim_grads = torch.stack(all_dim_grads, dim=1)
        if return_pred:
            return all_dim_grads, y
        else:
            return all_dim_grads
