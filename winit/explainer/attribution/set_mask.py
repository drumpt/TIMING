import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch.nn import Softmax

from winit.explainer.attribution.perturbation import Perturbation, SetDropReference
from winit.explainer.dynamaskutils.metrics import get_entropy, get_information


class SetMask:
    """
    Handles set-based dynamic masks with missing value information.
    """

    def __init__(
        self,
        perturbation: SetDropReference,
        device,
        task: str = "regression",
        verbose: bool = False,
        random_seed: int = 42,
        deletion_mode: bool = False,
        eps: float = 1.0e-7,
    ):
        # Same initialization as before
        self.verbose = verbose
        self.device = device
        self.random_seed = random_seed
        self.deletion_mode = deletion_mode
        self.perturbation = perturbation
        self.eps = eps
        self.task = task
        self.X = None
        self.mask = None  # Original mask
        self.mask_tensor = None  # Importance mask
        self.T = None
        self.N_features = None
        self.Y_target = None
        self.f = None
        self.n_epoch = None
        self.hist = None
        self.loss_function = None
        self.log = logging.getLogger(SetMask.__name__)

    def get_error(self):
        """Returns the error between unperturbed and perturbed input."""
        if self.deletion_mode:
            X_pert, mask_pert = self.perturbation.apply(
                X=self.X, mask_in=self.mask, mask_tensor=1 - self.mask_tensor
            )
        else:
            X_pert, mask_pert = self.perturbation.apply(
                X=self.X, mask_in=self.mask, mask_tensor=self.mask_tensor
            )
        Y_pert = self.f(X_pert, mask_pert)
        if self.task == "classification":
            Y_pert = torch.log(Softmax(dim=1)(Y_pert))
        return self.loss_function(Y_pert, self.Y_target)

    def get_error_multiple(self):
        """Handles multiple samples."""
        if self.deletion_mode:
            X_pert, mask_pert = self.perturbation.apply_multiple(
                X=self.X, mask_in=self.mask, mask_tensor=1 - self.mask_tensor
            )
        else:
            X_pert, mask_pert = self.perturbation.apply_multiple(
                X=self.X, mask_in=self.mask, mask_tensor=self.mask_tensor
            )
        Y_pert = self.f(X_pert, mask_pert)
        if self.task == "classification":
            Y_pert = torch.log(Softmax(dim=2)(Y_pert))
        return self.loss_function(Y_pert.unsqueeze(0), self.Y_target.unsqueeze(0))

    # Other methods remain similar but visualization methods need to account for mask
    def plot_mask(
        self, ids_time=None, ids_feature=None, smooth: bool = False, sigma: float = 1.0
    ):
        """Plots mask with original mask information."""
        sns.set()
        if smooth:
            mask_tensor = self.get_smooth_mask(sigma)
        else:
            mask_tensor = self.mask_tensor

        submask_tensor_np = self.extract_submask(
            mask_tensor, ids_time, ids_feature
        ).numpy()
        original_mask_np = self.extract_submask(
            self.mask, ids_time, ids_feature
        ).numpy()

        # Create DataFrame with both masks
        df = pd.DataFrame(
            data=np.transpose(submask_tensor_np), index=ids_feature, columns=ids_time
        )

        # Plot heatmap with original mask overlay
        color_map = sns.diverging_palette(10, 133, as_cmap=True)
        heat_map = sns.heatmap(
            data=df,
            cmap=color_map,
            cbar_kws={"label": "Importance Mask"},
            vmin=0,
            vmax=1,
            mask=np.transpose(original_mask_np),  # Mask out missing values
        )
        plt.xlabel("Time")
        plt.ylabel("Feature Number")
        plt.title("Mask coefficients over time (white areas are missing)")
        plt.show()
