import os
import sys
from tqdm import tqdm

import torch

sys.path.append("../utils")
import run_utils
import python_utils


class Trainer:
    def __init__(self, config, model, optimizer, loss_func, init_checkpoints_dirpath, run_dirpath):
        self.config = config
        self.model = model
        self.optimizer = optimizer

        self.loss_func = loss_func

        self.init_checkpoints_dirpath = init_checkpoints_dirpath
        logs_dirpath, checkpoints_dirpath = run_utils.setup_run_subdirs(run_dirpath)
        self.logs_dirpath = logs_dirpath
        self.checkpoints_dirpath = checkpoints_dirpath

    def loss_batch(self, batch, opt=None):
        # print("-"*100)
        # print("image min:")
        # print(batch["image"][0].min().item())
        # print("image max:")
        # print(batch["image"][0].max().item())
        # print("gt_pos: {}".format(batch["gt_pos"][0]))
        pred = self.model(batch)
        loss = self.loss_func(pred, batch)

        # Detect if loss is nan
        contains_nan = bool(torch.sum(torch.isnan(loss)).item())
        # import random
        # contains_nan = random.random() < 0.001
        if contains_nan:
            raise ValueError("NaN values detected, aborting...")

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss, len(batch)

    def run_epoch(self, name, dl, update_steps, opt=None):
        pbar = tqdm(dl, desc="{}: ".format(name), leave=False)
        running_loss = 0
        running_nums = 0
        for batch_index, batch in enumerate(pbar):
            loss, nums = self.loss_batch(batch, opt=opt)
            running_loss = running_loss * running_nums + loss.item() * nums
            running_nums += nums
            running_loss /= running_nums
            if batch_index % update_steps == 0:
                pbar.set_postfix(loss="{:04f}".format(loss.item()))

        return running_loss

    def fit(self, config, train_dl, val_dl):

        # Try loading previous model
        checkpoint = self.load_checkpoint(self.checkpoints_dirpath)  # Try last checkpoint
        if checkpoint is None and self.init_checkpoints_dirpath is not None:
            # Try with init_checkpoints_dirpath:
            checkpoint = self.load_checkpoint(self.init_checkpoints_dirpath)
        if checkpoint is None:
            checkpoint = {
                "epoch": -1,
            }
        start_epoch = checkpoint["epoch"] + 1  # Start at next epoch

        fit_pbar = tqdm(range(start_epoch, config["max_epoch"]), desc="Fitting: ", initial=start_epoch, total=config["max_epoch"])
        train_loss = None
        val_loss = None
        for epoch_index, epoch in enumerate(fit_pbar):
            self.model.train()
            train_loss = self.run_epoch("Train", train_dl, config["train_log_step"], opt=self.optimizer)

            self.model.eval()
            with torch.no_grad():
                val_loss = self.run_epoch("Val", val_dl, max(len(val_dl) // 4, 1))

            fit_pbar.set_postfix(train_loss="{:04f}".format(train_loss), val_loss="{:04f}".format(val_loss))

            if epoch_index % config["checkpoint_epoch"] == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
        self.save_checkpoint(epoch, train_loss, val_loss)

        python_utils.save_json(os.path.join(self.logs_dirpath, "final_losses.json"), {
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

    def load_checkpoint(self, checkpoints_dirpath):
        """
        Loads last checkpoint in checkpoints_dirpath
        :param checkpoints_dirpath:
        :return:
        """
        try:
            filepaths = python_utils.get_filepaths(checkpoints_dirpath, endswith_str=".tar", startswith_str="checkpoint")
            if len(filepaths) == 0:
                return None

            filepaths = sorted(filepaths)
            filepath = filepaths[-1]  # Last checkpoint

            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']

            return {
                "epoch": epoch,
            }
        except NotADirectoryError:
            return None

    def save_checkpoint(self, epoch, train_loss, val_loss):
        filename_format = "checkpoint.epoch_{:06d}.tar"
        filepath = os.path.join(self.checkpoints_dirpath, filename_format.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, filepath)
