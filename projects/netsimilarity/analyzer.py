import os
import sys
from tqdm import tqdm

import numpy as np
import torch

sys.path.append("../utils")
import run_utils
import python_utils
import print_utils


class Analyzer:
    def __init__(self, config, model, run_dirpath):
        self.config = config
        self.model = model
        _, checkpoints_dirpath = run_utils.setup_run_subdirs(run_dirpath)
        self.checkpoints_dirpath = checkpoints_dirpath
        self.grads_dirpath = os.path.join(run_dirpath, "grads")
        if not os.path.exists(self.grads_dirpath):
            os.makedirs(self.grads_dirpath)

    def loss_batch(self, batch, opt=None):
        # print("-"*100)
        # print("image min:")
        # print(batch["image"][0].min().item())
        # print("image max:")
        # print(batch["image"][0].max().item())
        # print("gt_pos: {}".format(batch["gt_pos"][0]))
        pred = self.model(batch)
        # print("pred: {}".format(pred[0]))
        loss = self.loss_func(pred, batch)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss, len(batch)

    def compute_and_save_grads(self, dl):
        # Loading model
        if self.load_checkpoint(self.checkpoints_dirpath) is None:  # Try last checkpoint
            print_utils.print_error("Checkpoint {} could not be loaded. Aborting...".format(self.checkpoints_dirpath))
            exit()

        self.model.train()
        pbar = tqdm(dl, desc="Compute grads: ")
        for batch_index, batch in enumerate(pbar):
            grads, pred = self.model.compute_grads(batch, return_pred=True)
            grads = grads.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            # Save grads in run_dirpath
            grads_filename = "grads.{:05d}.npy".format(batch_index)
            pred_filename = "pred.{:05d}.npy".format(batch_index)
            grads_filepath = os.path.join(self.grads_dirpath, grads_filename)
            pred_filepath = os.path.join(self.grads_dirpath, pred_filename)
            np.save(grads_filepath, grads)
            np.save(pred_filepath, pred)

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
            return True
        except NotADirectoryError:
            return None
