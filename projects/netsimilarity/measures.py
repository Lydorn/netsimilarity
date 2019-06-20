import torch


def l2_error(pred, gt):
    return torch.mean(torch.sqrt(torch.sum(torch.pow(pred - gt, 2), dim=-1)))


def l1_loss(pred, batch):
    gt = batch["gt"]
    return torch.mean(torch.abs(pred - gt))


def l2_loss(pred, batch):
    gt_pos = batch["gt_pos"]
    return l2_error(pred, gt_pos)


def noisy_l2_loss(pred, batch):
    gt = batch["gt"]
    noise = batch["noise"]
    noisy_gt = gt + noise
    return l2_error(pred, noisy_gt)


def l2_loss_combine_color_and_type(pred, batch):
    gt_color = batch["color"]
    gt_type = batch["type"].view(-1, 1)
    gt_mean_color = gt_color.mean(dim=-1, keepdim=True)
    gt_res = (gt_type + gt_mean_color) / 2

    return l1_loss(pred, gt_res)
