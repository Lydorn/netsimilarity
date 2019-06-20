import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F


class DoubleNLLLoss(nn.Module):

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.loss_fct = nn.NLLLoss()

    def forward(self, scores, targets):
        assert len(scores) == len(targets) == 2
        loss_0 = self.loss_fct(scores[0], targets[0])
        loss_1 = self.loss_fct(scores[1], targets[1])
        return self.weight * (loss_0 + loss_1) / 2.0


class DoubleNablaSimLoss(nn.Module):

    def __init__(self, weight, mask_kind):
        super().__init__()
        self.weight = weight
        assert mask_kind in ["random", "posnegflat"]
        self.mask_kind = mask_kind

    def forward(self, params, outputs, targets):
        assert isinstance(outputs, tuple), isinstance(targets, tuple)
        assert len(outputs) == len(targets) == 2

        # gather grads
        mask = self.get_mask(outputs[0].size(-1), targets[1], device=outputs[0].device)
        batch_outputs = [(F.log_softmax(logit, dim=-1) * mask).sum(-1) for logit in outputs]
        single_outputs = [logit.mean(0) for logit in batch_outputs]  # sum over the batch_dimension
        model_grads = [autograd.grad(logit, params, create_graph=True) for logit in single_outputs]
        flat_grads = [torch.cat([dw.flatten() for dw in all_dws]) for all_dws in model_grads]

        # compute nablasim loss
        grad_A, grad_B = flat_grads
        assert grad_A.dim() == grad_B.dim() == 1
        similarity = F.cosine_similarity(grad_A, grad_B, dim=0)
        # similarity = ((grad_A - grad_B)**2).sum() / (grad_A.norm() * grad_B.norm())
        return self.weight * similarity

    def get_mask(self, output_size, pos_idx, device):
        if self.mask_kind == "random":
            return torch.randn(output_size)
        if self.mask_kind == "posnegflat":
            mask = torch.nn.init.constant_(torch.empty(output_size), -1 / (output_size - 1))
            mask[pos_idx] = 1.0
            mask = mask.to(device)
            return mask
        raise RuntimeError(f"Unknown mask kind: '{self.mask_kind}'.'")
