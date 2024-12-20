import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim

from winit.explainer.attribution.set_mask import SetMask
from winit.explainer.attribution.perturbation import Perturbation


class SetMaskGroup:
    """This class allows to fit several mask of different areas simultaneously for set-based inputs.

    Attributes:
        perturbation (SetDropReference):
            Perturbation class that uses mask to generate set-based perturbations
        device: Device used for torch tensors
        verbose (bool): True if messages should be displayed during optimization
        random_seed (int): Random seed for reproducibility
        deletion_mode (bool): True if mask should identify most impactful deletions
        eps (float): Small number for numerical stability
        masks_tensor (torch.tensor): Tensor containing coefficient of each mask
        T (int): Number of time steps
        N_features (int): Number of features
        Y_target (torch.tensor): Black-box prediction
        hist (torch.tensor): History tensor containing metrics at different epochs
    """

    def __init__(
        self,
        perturbation: Perturbation,
        device,
        random_seed: int = 987,
        deletion_mode: bool = False,
        verbose: bool = True,
    ):
        self.perturbation = perturbation
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose
        self.deletion_mode = deletion_mode
        self.mask_list = None
        self.area_list = None
        self.f = None
        self.X = None
        self.mask_in = None
        self.n_epoch = None
        self.T = None
        self.N_features = None
        self.Y_target = None
        self.masks_tensor = None
        self.hist = None
        self.log = logging.getLogger(SetMaskGroup.__name__)

    def fit_multiple(
        self,
        X,
        mask_in,
        f,
        area_list,
        loss_function_multiple,
        use_last_timestep_only: bool = False,
        n_epoch: int = 1000,
        initial_mask_coeff: float = 0.5,
        size_reg_factor_init: float = 0.1,
        size_reg_factor_dilation: float = 100,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        time_reg_factor: float = 0,
    ):
        # Ensure that the area list is sorted
        area_list.sort()
        self.area_list = area_list
        N_area = len(area_list)

        # Create a list of masks
        mask_list = []

        # Initialize the random seed and the attributes
        t_fit = time.time()
        torch.manual_seed(self.random_seed)
        reg_factor = size_reg_factor_init
        error_factor = (
            1 - 2 * self.deletion_mode
        )  # In deletion mode, error has to be maximized
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epoch)

        # Store attributes
        self.f = f
        self.X = X
        self.mask_in = mask_in
        self.n_epoch = n_epoch
        num_samples, self.T, self.N_features = X.shape

        # Get target outputs
        self.Y_target = f(X, mask_in)  # num_samples, num_time, num_state=2
        if use_last_timestep_only:
            self.Y_target = self.Y_target[:, -1:, :]

        # Initialize mask tensor
        self.masks_tensor = initial_mask_coeff * torch.ones(
            size=(N_area, num_samples, self.T, self.N_features), device=self.device
        )

        # Repeat target for each area
        Y_target_group = (
            self.Y_target.clone().detach().unsqueeze(0).repeat(N_area, 1, 1, 1)
        )

        # Setup optimization
        masks_tensor_new = self.masks_tensor.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([masks_tensor_new], lr=learning_rate, momentum=momentum)
        metrics = []

        # Initialize reference vector for regulator
        reg_ref = torch.ones(
            (N_area, num_samples, self.T * self.N_features),
            dtype=torch.float32,
            device=self.device,
        )
        for i, area in enumerate(self.area_list):
            reg_ref[i, :, : int((1 - area) * self.T * self.N_features)] = 0.0

        # Run optimization
        for k in range(n_epoch):
            t_loop = time.time()

            # Generate perturbed input and mask
            if self.deletion_mode:
                X_pert, mask_pert = self.perturbation.apply_extremal(
                    X=X, mask_in=mask_in, mask_tensor=1 - masks_tensor_new
                )
            else:
                X_pert, mask_pert = self.perturbation.apply_extremal(
                    X=X, mask_in=mask_in, mask_tensor=masks_tensor_new
                )

            # Reshape for model input
            X_pert_flatten = X_pert.reshape(
                N_area * num_samples, self.T, self.N_features
            )
            mask_pert_flatten = mask_pert.reshape(
                N_area * num_samples, self.T, self.N_features
            )

            # Get model predictions
            Y_pert_flatten = f(X_pert_flatten, mask_pert_flatten)

            if use_last_timestep_only:
                Y_pert = Y_pert_flatten.reshape(N_area, num_samples, 1, -1)
            else:
                Y_pert = Y_pert_flatten.reshape(N_area, num_samples, self.T, -1)

            # Calculate losses
            error = loss_function_multiple(Y_pert, Y_target_group)
            masks_tensor_sorted = masks_tensor_new.reshape(
                N_area, num_samples, self.T * self.N_features
            ).sort(dim=2)[0]
            size_reg = ((reg_ref - masks_tensor_sorted) ** 2).mean(dim=[0, 2])
            masks_tensor_diff = (
                masks_tensor_new[:, :, 1 : self.T - 1, :]
                - masks_tensor_new[:, :, : self.T - 2, :]
            )
            time_reg = (torch.abs(masks_tensor_diff)).mean(dim=[0, 2, 3])

            # Total loss
            loss = (
                error_factor * error
                + reg_factor * size_reg
                + time_reg_factor * time_reg
            )

            # Optimization step
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()

            # Clamp masks between 0 and 1
            masks_tensor_new.data = masks_tensor_new.data.clamp(0, 1)

            # Save metrics
            metric = (
                torch.stack([error, size_reg, time_reg], dim=1).detach().cpu().numpy()
            )
            metrics.append(metric)

            # Update regulator coefficient
            reg_factor *= reg_multiplicator

            # Logging
            t_loop = time.time() - t_loop
            if self.verbose:
                self.log.info(
                    f"Epoch {k + 1}/{n_epoch}: error = {error.mean().data:.3g} ; "
                    f"size regulator = {size_reg.mean().data:.3g} ; time regulator = {time_reg.mean().data:.3g} ;"
                    f" time elapsed = {t_loop:.3g} s"
                )

        # Save final results
        self.masks_tensor = masks_tensor_new.clone().detach().requires_grad_(False)
        self.hist = torch.from_numpy(np.stack(metrics, axis=2))
        t_fit = time.time() - t_fit

        # Final logging
        if self.verbose:
            for i, e in enumerate(error.data):
                self.log.info(
                    f"The optimization finished: error = {e:.3g} ; size regulator = {size_reg[i].data:.3g} ;"
                    f" time regulator = {time_reg[i].data:.3g} ; time elapsed = {t_fit:.3g} s"
                )
        else:
            self.log.info(
                f"The optimization finished: error = {error.mean().data:.3g} ; size regulator = {size_reg.mean().data:.3g} ;"
                f" time regulator = {time_reg.mean().data:.3g} ; time elapsed = {t_fit:.3g} s"
            )

        # Create individual mask objects
        for index, mask_tensor in enumerate(self.masks_tensor.unbind(dim=0)):
            mask = SetMask(
                perturbation=self.perturbation,
                device=self.device,
                verbose=False,
                deletion_mode=self.deletion_mode,
            )
            mask.mask_tensor = mask_tensor
            mask.hist = self.hist
            mask.f = self.f
            mask.X = self.X
            mask.mask = self.mask_in
            mask.n_epoch = self.n_epoch
            mask.T, mask.N_features = self.T, self.N_features
            mask.Y_target = self.Y_target
            mask.loss_function = loss_function_multiple
            mask_list.append(mask)
        self.mask_list = mask_list

    def get_best_mask(self):
        """Returns the mask with lowest error."""
        error_list = [mask.get_error() for mask in self.mask_list]
        best_index = error_list.index(min(error_list))
        self.log.info(
            f"The mask of area {self.area_list[best_index]:.2g} is"
            f" the best with error = {error_list[best_index]:.3g}."
        )
        return self.mask_list[best_index]

    def get_extremal_mask_multiple(self, thresholds):
        """Returns the extremal mask for the acceptable error threshold."""
        error_list = torch.stack(
            [mask.get_error_multiple() for mask in self.mask_list], dim=1
        )
        mask_stacked = torch.stack([mask.mask_tensor for mask in self.mask_list])
        num_area, num_samples, num_times, num_features = mask_stacked.shape

        # If minimal error above threshold, select mask with lowest error
        thres_mask = torch.min(error_list, dim=1)[0] > thresholds
        best_mask = torch.argmin(error_list, dim=1)  # (num_sample)
        error_mask = (error_list < thresholds.view(-1, 1)) * torch.arange(
            -len(self.mask_list), 0
        ).view(1, -1).to(self.device)
        first_mask = torch.argmin(error_mask, dim=1)
        indexes = torch.where(thres_mask, best_mask, first_mask)  # (num_sample)

        selected_masks = torch.gather(
            mask_stacked,
            0,
            indexes.view(1, num_samples, 1, 1).expand(
                1, num_samples, num_times, num_features
            ),
        )
        return selected_masks.reshape(num_samples, num_times, num_features)

    def plot_errors(self):
        """Plots error as function of mask size."""
        sns.set()
        error_list = [mask.get_error() for mask in self.mask_list]
        plt.plot(self.area_list, error_list)
        plt.title("Errors for the various masks")
        plt.xlabel("Mask area")
        plt.ylabel("Error")
        plt.show()
