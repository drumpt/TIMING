"""Most of the bellow baselines rely on their captum implementation.

For more information, please check https://github.com/pytorch/captum

Note that these implementations are mainly used in the rate time and feature experiment.
For the state and mimic experiment, we use the results produced by FIT.
For more details on the FIT implementations, please check https://github.com/sanatonek/time_series_explainability
"""

import torch
from captum.attr import (
    FeaturePermutation,
    GradientShap,
    IntegratedGradients,
    Occlusion,
    ShapleyValueSampling,
)
from torch.distributions import Beta
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import itertools
import numpy as np
import torch
import torch.nn as nn
import random

from utils.tensor_manipulation import normalize as normal

# Perturbation methods:


class FO:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = Occlusion(forward_func=self.f)
        baseline = torch.mean(
            X, dim=0, keepdim=True
        )  # The baseline is chosen to be the average value for each feature
        attr = explainer.attribute(X, sliding_window_shapes=(1,), baselines=baseline)
        if normalize:
            attr = normal(
                torch.abs(attr)
            )  # The absolute value of the FO attribution gives the feature importance
        return attr


class FP:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = FeaturePermutation(forward_func=self.f)
        attr = explainer.attribute(X)
        if normalize:
            attr = normal(
                torch.abs(attr)
            )  # The absolute value of the FP attribution gives the feature importance
        return attr


# Integrated Gradient:


class IG:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = IntegratedGradients(forward_func=self.f)
        baseline = X * 0  # The baseline is chosen to be zero for all features
        attr = explainer.attribute(X, baselines=baseline)
        if normalize:
            attr = normal(
                torch.abs(attr)
            )  # The absolute value of the IG attribution gives the feature importance
        return attr


# Shapley methods:


class GradShap:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = GradientShap(forward_func=self.f, multiply_by_inputs=False)
        attr = explainer.attribute(X, baselines=torch.cat([0 * X, 1 * X]))
        if normalize:
            attr = normal(
                torch.abs(attr)
            )  # The absolute value of the GradShap attribution gives the feature importance
        return attr


class SVS:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = ShapleyValueSampling(forward_func=self.f)
        baseline = torch.mean(X, dim=0, keepdim=True)
        attr = explainer.attribute(X, baselines=baseline)
        if normalize:
            attr = normal(
                torch.abs(attr)
            )  # The absolute value of the SVS attribution gives the feature importance
        return attr


class OUR:
    def __init__(self, model):
        self.model = model

    def get_batch_masks(self, B, T, D, n_samples, num_segments, min_seg_len, max_seg_len, device):
        # Generate all random numbers at once
        dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        t_starts = torch.randint(0, T, (n_samples, B, num_segments), device=device)
        
        # Initialize mask
        time_mask = torch.ones(n_samples, B, T, D, device=device)
        
        # Vectorized masking
        batch_idx = torch.arange(B, device=device)
        sample_idx = torch.arange(n_samples, device=device)
        
        for i, b, s in itertools.product(range(n_samples), range(B), range(num_segments)):
            time_mask[i, b, t_starts[i,b,s]:t_starts[i,b,s]+seg_lens[i,b,s], dims[i,b,s]] = 0
            
        return time_mask

    def attribute_ori_ig(
        self, inputs, baselines, targets, additional_forward_args, n_samples=50
    ):
        """
        Compute Integrated Gradients attributions for a time series model.
        Args:
            inputs (torch.Tensor): Input tensor [B, T, D]
            baselines (torch.Tensor): Baseline tensor [B, T, D]
            targets (torch.Tensor): Target tensor [B] (integer class targets)
            additional_forward_args: tuple containing (mask, ...)
            n_samples (int): Number of interpolation steps
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        # Get mask from additional_forward_args
        mask = additional_forward_args[0]

        # 1. Generate interpolation path
        alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
        alphas = alphas.view(-1, 1, 1, 1)  # shape: [n_samples, 1, 1, 1]

        # Expand inputs and baselines
        expanded_inputs = inputs.unsqueeze(0)  # [1, B, T, D]
        expanded_baselines = baselines.unsqueeze(0)  # [1, B, T, D]

        # Interpolate between baseline and input
        interpolated = expanded_baselines + alphas * (
            expanded_inputs - expanded_baselines
        )
        interpolated.requires_grad = True

        # 2. Forward pass
        predictions = self.model(
            interpolated.view(-1, inputs.shape[1], inputs.shape[2]),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )

        # Handle predictions shape
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        predictions = predictions.view(n_samples, inputs.shape[0], -1)

        # Gather predictions for target class
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0)
            .unsqueeze(-1)
            .expand(n_samples, inputs.shape[0], 1),
        ).squeeze(
            -1
        )  # shape [n_samples, B]

        # Sum for gradient computation
        total = gathered.sum()

        # 3. Compute gradients
        grad = torch.autograd.grad(
            outputs=total, inputs=interpolated, retain_graph=True
        )[
            0
        ]  # shape: [n_samples, B, T, D]

        # Average gradients over interpolation steps
        grads = grad.mean(dim=0)  # [B, T, D]

        # 4. Compute final attributions
        attributions = grads * (inputs - baselines)

        return attributions

    def attribute_random(
        self, inputs, baselines, targets, additional_forward_args, n_samples=50
    ):
        """
        inputs:  [B, T, D]
        baselines:  [B, T, D] (same shape as inputs)
        targets:  [B] (integer class targets)
        additional_forward_args: unused except . . . [2] for 'return_all'
        n_samples: number of interpolation steps
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        # -------------------------------------------------
        # 1) Build interpolation from baseline --> inputs
        # -------------------------------------------------
        alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
        alphas = alphas.view(-1, 1, 1, 1)  # shape: [n_samples, 1, 1, 1]

        # Start from "start_pos" so that alpha=0 means "baselines"
        # and alpha=1 means "inputs" (plus small noise).
        start_pos = baselines

        # Expand to shape [n_samples, B, T, D]
        expanded_inputs = inputs.unsqueeze(0)  # [1, B, T, D]
        expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]

        # Interpolate
        noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

        # Add a little random noise
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise

        # -------------------------------------------------
        # 2) Time-aware masking
        # -------------------------------------------------
        # For each interpolation step i (0..n_samples-1),
        # we randomly fix a subset of time/feature points.
        # Let's sample a [n_samples, B, T, D] mask in which:
        #   mask[i, b, t, d] = 1 -> use interpolation
        #   mask[i, b, t, d] = 0 -> fix to the real input
        #
        # You can adapt this logic to fix entire timesteps, or
        # fix with a certain probability, etc.
        #
        # Also, by using inputs.detach() for fixed values,
        # there's no gradient flowing for those positions.

        # Example: 50% chance to fix each [t, d]
        fix_probability = 0.5  # tweak as needed
        rand_mask = torch.rand_like(noisy_inputs)  # shape [n_samples, B, T, D]
        # Convert to {0,1} by comparing to fix_probability
        # 1 = keep interpolation, 0 = fix to actual input
        time_mask = (rand_mask > fix_probability).float()

        # Detach actual inputs so no gradient is assigned to them
        fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
        # broadcast to match [n_samples, B, T, D]
        # The random mask has the same shape as `noisy_inputs`
        # => we combine them:
        noisy_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

        # Turn on gradient for only the interpolation portion
        noisy_inputs.requires_grad = True

        # -------------------------------------------------
        # 3) Forward pass & gather target predictions
        # -------------------------------------------------
        predictions = self.model(
            noisy_inputs.view(-1, inputs.shape[1], inputs.shape[2]),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )

        # Make sure predictions has shape [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        predictions = predictions.view(n_samples, inputs.shape[0], -1)

        # Gather the logit of the correct class for each sample
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0)
            .unsqueeze(-1)
            .expand(n_samples, inputs.shape[0], 1),
        ).squeeze(
            -1
        )  # shape [n_samples, B]

        # Sum across all n_samples and batch for gradient
        total_for_target = gathered.sum()

        # -------------------------------------------------
        # 4) Compute gradients wrt `noisy_inputs`
        # -------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=noisy_inputs,
            retain_graph=True,
            allow_unused=True,
        )[
            0
        ]  # shape: [n_samples, B, T, D]

        # Average gradients over n_samples
        grads = grad.sum(dim=0)  # [B, T, D]

        # -------------------------------------------------
        # 5) Build final attributions
        # -------------------------------------------------
        # For positions we left "interpolating,"
        # attributions = grads * (inputs - baselines).
        # For positions that were forced to real input,
        # we want zero attributions.
        # One simple approach is to also average the mask
        # across the n_samples.
        # Alternatively, if you want *every sample* to fix
        # that position, you can incorporate a different rule.

        # If you want each example's final attributions to
        # reflect the fraction of times it was “interpolated”:
        mask_avg = time_mask.sum(dim=0)  # shape [B, T, D]
        raw_attr = grads * (inputs - baselines)
        # Zero out contributions in proportion to how often
        # they were "fixed".
        final_attributions = raw_attr / mask_avg

        # You can also choose to skip mask_avg if you prefer
        # to keep the raw average gradient, which might
        # partially incorporate the fact that sometimes it
        # was fixed or not.

        return final_attributions

    # def attribute_temporal_dependency_ig(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]
    #     timesteps = additional_forward_args[1]
    #     attributions = torch.zeros_like(inputs)

    #     for t in range(T):
    #         # Create baseline with clone to avoid in-place issues
    #         expanded_baselines = (
    #             baselines.unsqueeze(0).expand(n_samples, B, T, F).clone()
    #         )

    #         #######################################################
    #         # if t > 0:
    #         #     time_diffs = torch.arange(t, device=inputs.device)
    #         #     normalized_diffs = time_diffs / (t - 1)
    #         #     k = 0  # Decay rate
    #         #     decay_weights = torch.exp(-k * normalized_diffs)
    #         #     decay_weights = decay_weights.view(1, 1, -1, 1)  # [1, 1, t, 1]

    #         #     # Apply decay to past values
    #         #     expanded_baselines[..., :t, :] = (inputs[None, :, :t, :] * decay_weights)

    #         # # Set current timestep to zero
    #         # expanded_baselines[..., t, :] = 0

    #         #######################################################
    #         # Method 1: Use windowed approach
    #         window_size = 1  # or another appropriate size
    #         start_t = max(0, t - window_size)
    #         expanded_baselines = inputs[None].expand(n_samples, B, T, F).clone()
    #         # Only modify recent context
    #         expanded_baselines[..., start_t : t + 1, :] = baselines[
    #             ..., start_t : t + 1, :
    #         ]

    #         #######################################################
    #         # Generate IG paths and interpolate
    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(-1, 1, 1, 1)

    #         # Calculate difference
    #         diff = inputs[None].expand_as(expanded_baselines) - expanded_baselines

    #         # Interpolate
    #         interpolated = expanded_baselines + alphas * diff

    #         #######################################################
    #         # interpolated = inputs[None].expand_as(expanded_baselines).clone()
    #         # interpolated[..., t:t+1, :] = expanded_baselines[..., t:t+1, :] + alphas * (
    #         #     inputs[None, :, t:t+1, :] - expanded_baselines[..., t:t+1, :]
    #         # )
    #         #######################################################
    #         interpolated.requires_grad = True

    #         # Forward pass with expanded dimensions
    #         expanded_masks = masks.unsqueeze(0).expand(n_samples, B, T, F)
    #         expanded_timesteps = timesteps.unsqueeze(0).expand(n_samples, B, T)

    #         predictions = self.model(
    #             interpolated.reshape(-1, T, F),
    #             mask=expanded_masks.reshape(-1, T, F),
    #             timesteps=expanded_timesteps.reshape(-1, T),
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, B, -1)

    #         gathered = predictions.gather(
    #             dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #         ).squeeze(-1)

    #         total_for_target = gathered.sum()

    #         # Compute gradients
    #         grads = torch.autograd.grad(
    #             outputs=total_for_target,
    #             inputs=interpolated,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0]

    #         # Average gradients and compute attribution with correct dimensions
    #         grads = grads.mean(0)  # [B, T, F]
    #         attributions[:, t, :] = grads[:, t, :] * (
    #             inputs[:, t, :] - expanded_baselines[0, :, t, :]
    #         )

    #     return attributions

    # def attribute_pointwise(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     Point-wise IG: Interpolates each (t,f) point independently while keeping others fixed
    #     inputs: [B, T, F]
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, F = inputs.shape
    #     attributions = torch.zeros_like(inputs)

    #     # Iterate over each point (t,f)
    #     for t in range(T):
    #         for f in range(F):
    #             alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #             alphas = alphas.view(n_samples, 1, 1, 1)

    #             # Expand inputs and baselines
    #             expanded_inputs = inputs.unsqueeze(0).expand(n_samples, B, T, F)
    #             expanded_baselines = baselines.unsqueeze(0).expand(n_samples, B, T, F)

    #             # Create interpolated inputs where only point (t,f) varies
    #             interpolated = expanded_inputs.clone()
    #             interpolated[:, :, t : t + 1, f : f + 1] = expanded_baselines[
    #                 :, :, t : t + 1, f : f + 1
    #             ] + alphas * (
    #                 expanded_inputs[:, :, t : t + 1, f : f + 1]
    #                 - expanded_baselines[:, :, t : t + 1, f : f + 1]
    #             )

    #             interpolated.requires_grad = True

    #             # Forward pass
    #             predictions = self.model(
    #                 interpolated.view(-1, T, F),
    #                 mask=None,
    #                 timesteps=None,
    #                 return_all=additional_forward_args[2],
    #             )

    #             if predictions.dim() == 1:
    #                 predictions = predictions.unsqueeze(1)
    #             predictions = predictions.view(n_samples, B, -1)

    #             gathered = predictions.gather(
    #                 dim=2,
    #                 index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1),
    #             ).squeeze(-1)

    #             total_for_target = gathered.sum()

    #             # Compute gradients
    #             grad = torch.autograd.grad(
    #                 outputs=total_for_target,
    #                 inputs=interpolated,
    #                 retain_graph=True,
    #                 allow_unused=True,
    #             )[0]

    #             # Average gradients over samples
    #             grads = grad.mean(dim=0)

    #             # Store attribution for this point (t,f)
    #             attributions[:, t, f] = grads[:, t, f] * (
    #                 inputs[:, t, f] - baselines[:, t, f]
    #             )

    #     return attributions

    # def attribute_groupwise_current(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     Time-wise IG with mask-based grouping using carry-forward baseline
    #     Args:
    #         inputs: Input tensor [B, T, F]
    #         targets: Target indices [B]
    #         additional_forward_args: Tuple of (masks, timesteps, return_all)
    #             - masks: Boolean tensor [B, T, F]
    #             - timesteps: Timestep tensor [B, T]
    #             - return_all: Boolean flag for model output
    #         n_samples: Number of samples for integral approximation
    #     Returns:
    #         attributions: Attribution tensor [B, T, F]
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]  # [B, T, F]
    #     timesteps = additional_forward_args[1]  # Original timesteps
    #     attributions = torch.zeros_like(inputs)

    #     # Get all indices where mask is 1
    #     batch_idx, time_idx, feat_idx = torch.where(masks == 1)
    #     alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #     alphas = alphas.view(n_samples, 1, 1, 1)  # [n_samples, 1, 1, 1]

    #     # Create baseline by carrying forward last observed values
    #     baseline_current = inputs.clone()
    #     for b in range(B):
    #         for f in range(F):
    #             # Get observation times for this feature
    #             obs_times = time_idx[(batch_idx == b) & (feat_idx == f)]
    #             if len(obs_times) == 0:
    #                 continue

    #             # Fill in carried forward values
    #             last_val = inputs[b, 0, f]  # Start with initial value
    #             for t_idx in range(T):
    #                 if t_idx in obs_times:
    #                     last_val = inputs[b, t_idx, f]
    #                 else:
    #                     baseline_current[b, t_idx, f] = last_val

    #     # Expand inputs and baselines with proper dimensions
    #     expanded_inputs = inputs.unsqueeze(0).expand(n_samples, B, T, F)
    #     expanded_baselines = baseline_current.unsqueeze(0).expand(n_samples, B, T, F)
    #     interpolated = expanded_inputs.clone()

    #     # For each observed point, interpolate its group
    #     for b, t, f in zip(batch_idx, time_idx, feat_idx):
    #         # Find next observation point for this (batch, feature)
    #         next_t = time_idx[(batch_idx == b) & (feat_idx == f) & (time_idx > t)]
    #         end = next_t[0] if len(next_t) > 0 else T

    #         # Interpolate the entire group
    #         interpolated[:, b : b + 1, t:end, f : f + 1] = expanded_baselines[
    #             :, b : b + 1, t:end, f : f + 1
    #         ] + alphas * (
    #             expanded_inputs[:, b : b + 1, t:end, f : f + 1]
    #             - expanded_baselines[:, b : b + 1, t:end, f : f + 1]
    #         )

    #     # Prepare for forward pass
    #     interpolated_flat = interpolated.reshape(-1, T, F)
    #     interpolated_flat.requires_grad = True

    #     # Expand masks and timesteps
    #     expanded_mask = masks.unsqueeze(0).expand(n_samples, B, T, F)
    #     expanded_mask = expanded_mask.reshape(-1, T, F)

    #     expanded_timesteps = timesteps.unsqueeze(0).expand(n_samples, B, T)
    #     expanded_timesteps = expanded_timesteps.reshape(-1, T)

    #     # Calculate baseline predictions
    #     with torch.no_grad():
    #         base_preds = self.model(
    #             baseline_current,
    #             mask=masks,
    #             timesteps=timesteps,
    #             return_all=additional_forward_args[2],
    #         )
    #         if base_preds.dim() == 1:
    #             base_preds = base_preds.unsqueeze(1)
    #         base_preds = base_preds.gather(1, targets.unsqueeze(1)).squeeze(1)

    #     # Get predictions for interpolated inputs
    #     predictions = self.model(
    #         interpolated_flat,
    #         mask=expanded_mask,
    #         timesteps=expanded_timesteps,
    #         return_all=additional_forward_args[2],
    #     )

    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     gathered = predictions.gather(
    #         dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)

    #     pred_diff = gathered - base_preds.unsqueeze(0)

    #     # Calculate gradients
    #     grads = torch.autograd.grad(
    #         outputs=pred_diff.sum(),
    #         inputs=interpolated_flat,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]

    #     grads = grads.reshape(n_samples, B, T, F).mean(0)

    #     # Compute attributions for all groups
    #     for b, t, f in zip(batch_idx, time_idx, feat_idx):
    #         next_t = time_idx[(batch_idx == b) & (feat_idx == f) & (time_idx > t)]
    #         end = next_t[0] if len(next_t) > 0 else T

    #         current_diff = inputs[b, t:end, f] - baseline_current[b, t:end, f]
    #         step_size = 1.0 / (n_samples - 1)
    #         attributions[b, t:end, f] = (
    #             grads[b, t:end, f] * current_diff * step_size * n_samples
    #         )

    #     return attributions

    # def attribute_groupwise(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     Time-wise IG with mask-based grouping - Single forward pass version
    #     inputs: [B, T, F]
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]  # [B, T, F]
    #     attributions = torch.zeros_like(inputs)

    #     # Get all indices where mask is 1
    #     batch_idx, time_idx, feat_idx = torch.where(masks == 1)
    #     alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #     alphas = alphas.view(n_samples, 1, 1, 1)  # [n_samples, 1, 1, 1]

    #     # Expand inputs and baselines once
    #     expanded_inputs = inputs.unsqueeze(0).expand(n_samples, B, T, F)
    #     expanded_baselines = baselines.unsqueeze(0).expand(n_samples, B, T, F)
    #     interpolated = expanded_inputs.clone()

    #     # For each observed point, zero out its group in the interpolated input
    #     for b, t, f in zip(batch_idx, time_idx, feat_idx):
    #         # Find next observation point for this (batch, feature)
    #         next_t = time_idx[(batch_idx == b) & (feat_idx == f) & (time_idx > t)]
    #         end = next_t[0] if len(next_t) > 0 else T

    #         # Interpolate the entire group
    #         interpolated[:, b : b + 1, t:end, f : f + 1] = expanded_baselines[
    #             :, b : b + 1, t:end, f : f + 1
    #         ] + alphas * (
    #             expanded_inputs[:, b : b + 1, t:end, f : f + 1]
    #             - expanded_baselines[:, b : b + 1, t:end, f : f + 1]
    #         )

    #     # Single forward pass with all interpolations
    #     interpolated.requires_grad = True
    #     expanded_mask = masks.unsqueeze(0).expand(n_samples, -1, -1, -1)
    #     expanded_mask = expanded_mask.reshape(-1, T, F)

    #     predictions = self.model(
    #         interpolated.view(-1, T, F),
    #         mask=expanded_mask,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )

    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     gathered = predictions.gather(
    #         dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)

    #     total_for_target = gathered.sum()

    #     # Single gradient computation
    #     grads = torch.autograd.grad(
    #         outputs=total_for_target,
    #         inputs=interpolated,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0].mean(
    #         dim=0
    #     )  # Average over samples

    #     # Compute attributions for all groups at once
    #     for b, t, f in zip(batch_idx, time_idx, feat_idx):
    #         next_t = time_idx[(batch_idx == b) & (feat_idx == f) & (time_idx > t)]
    #         end = next_t[0] if len(next_t) > 0 else T
    #         attributions[b, t:end, f] = grads[b, t:end, f] * (
    #             inputs[b, t:end, f] - baselines[b, t:end, f]
    #         )

    #     return attributions

    # # def attribute_timewise_random(self, inputs, baselines, targets, additional_forward_args, n_samples=50, n_combinations=10):
    # #     """
    # #     Parallelized timewise IG with random combinations of all features
    # #     inputs: [B, T, F]
    # #     """
    # #     B, T, F = inputs.shape
    # #     masks = additional_forward_args[0]  # [B, T, F]
    # #     attributions = torch.zeros_like(inputs)

    # #     # For each timestep
    # #     for t in range(T):
    # #         # Process all batches and combinations at once
    # #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    # #         alphas = alphas.view(n_samples, 1, 1, 1)  # [n_samples, 1, 1, 1]

    # #         # Generate random combinations for all batches at once
    # #         # [n_combinations, F]
    # #         feature_mask = torch.rand(n_combinations, F, device=inputs.device) > 0.5

    # #         # Expand inputs and baselines
    # #         # [n_samples, n_combinations, B, T, F]
    # #         expanded_inputs = inputs.unsqueeze(0).unsqueeze(0).expand(n_samples, n_combinations, -1, -1, -1)
    # #         expanded_baselines = baselines.unsqueeze(0).unsqueeze(0).expand(n_samples, n_combinations, -1, -1, -1)

    # #         # Create interpolated inputs for all combinations
    # #         interpolated = expanded_inputs.clone()

    # #         # Set timestep t to baseline for all features
    # #         interpolated[..., t:t+1, :] = expanded_baselines[..., t:t+1, :]

    # #         # Prepare masks and alphas for interpolation
    # #         feature_mask_exp = feature_mask.view(1, n_combinations, 1, 1, F)  # [1, n_combinations, 1, 1, F]
    # #         alphas_exp = alphas.view(n_samples, 1, 1, 1, 1)  # [n_samples, 1, 1, 1, 1]

    # #         # Interpolate selected features
    # #         diff = expanded_inputs[..., t:t+1, :] - expanded_baselines[..., t:t+1, :]
    # #         update = expanded_baselines[..., t:t+1, :] + alphas_exp * diff
    # #         # Use broadcasting for the where operation
    # #         interpolated[..., t:t+1, :] = torch.where(
    # #             feature_mask_exp.expand_as(interpolated[..., t:t+1, :]),
    # #             update,
    # #             interpolated[..., t:t+1, :]
    # #         )

    # #         # Reshape for forward pass [n_samples*n_combinations*B, T, F]
    # #         interpolated = interpolated.reshape(-1, T, F)
    # #         interpolated.requires_grad = True

    # #         # Expand mask
    # #         expanded_mask = masks.unsqueeze(0).unsqueeze(0)\
    # #                         .expand(n_samples, n_combinations, -1, -1, -1)\
    # #                         .reshape(-1, T, F)

    # #         # Forward pass
    # #         predictions = self.model(
    # #             interpolated,
    # #             mask=expanded_mask,
    # #             timesteps=None,
    # #             return_all=additional_forward_args[2]
    # #         )

    # #         if predictions.dim() == 1:
    # #             predictions = predictions.unsqueeze(1)
    # #         predictions = predictions.view(n_samples, n_combinations, B, -1)

    # #         # Gather target predictions
    # #         gathered = predictions.gather(
    # #             dim=3,
    # #             index=targets.view(1, 1, B, 1).expand(n_samples, n_combinations, B, 1)
    # #         ).squeeze(-1)

    # #         total_for_target = gathered.sum()

    # #         # Compute gradients
    # #         grads = torch.autograd.grad(
    # #             outputs=total_for_target,
    # #             inputs=interpolated,
    # #             retain_graph=True,
    # #             allow_unused=True
    # #         )[0]

    # #         # Reshape gradients and average over samples
    # #         grads = grads.reshape(n_samples, n_combinations, B, T, F).mean(0)  # [n_combinations, B, T, F]

    # #         # Compute attributions for all combinations
    # #         diff = inputs - baselines
    # #         attributions[:, t, :] = (grads[..., t, :] * diff[:, t, :]).mean(0)

    # #     return attributions

    # def attribute_timewise_random(
    #     self,
    #     inputs,
    #     baselines,
    #     targets,
    #     additional_forward_args,
    #     n_samples=50,
    #     n_combinations=50,
    # ):
    #     """
    #     Parallelized timewise IG with random combinations of only observed features
    #     inputs: [B, T, F]
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]  # [B, T, F]
    #     attributions = torch.zeros_like(inputs)

    #     # For each timestep
    #     for t in range(T):
    #         # If no observations at this timestep, skip
    #         if not masks[:, t, :].any():
    #             continue

    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(n_samples, 1, 1, 1)  # [n_samples, 1, 1, 1]

    #         # Generate random combinations only for observed points
    #         obs_mask = masks[:, t, :].any(
    #             dim=0
    #         )  # [F] True if feature is observed in any batch
    #         feature_mask = torch.zeros(
    #             n_combinations, F, device=inputs.device, dtype=torch.bool
    #         )  # Changed to bool

    #         # Only sample from observed features
    #         obs_features = torch.where(obs_mask)[0]
    #         if len(obs_features) > 0:
    #             for i in range(n_combinations):
    #                 n_select = torch.randint(1, len(obs_features) + 1, (1,)).item()
    #                 selected = obs_features[
    #                     torch.randperm(len(obs_features))[:n_select]
    #                 ]
    #                 feature_mask[i, selected] = True  # Changed to True instead of 1

    #         # Expand inputs and baselines
    #         expanded_inputs = inputs.unsqueeze(0).unsqueeze(0)  # [1, 1, B, T, F]
    #         expanded_baselines = baselines.unsqueeze(0).unsqueeze(0)  # [1, 1, B, T, F]

    #         # Create interpolated inputs
    #         interpolated = expanded_inputs.expand(
    #             n_samples, n_combinations, B, T, F
    #         ).clone()

    #         # Set timestep t to baseline for all features
    #         interpolated[..., t, :] = (
    #             expanded_baselines[..., t : t + 1, :]
    #             .expand(n_samples, n_combinations, B, -1, F)
    #             .squeeze(3)
    #         )

    #         # Prepare masks and alphas for interpolation
    #         feature_mask_exp = feature_mask.view(
    #             1, n_combinations, 1, F
    #         )  # [1, n_combinations, 1, F]
    #         alphas_exp = alphas.view(n_samples, 1, 1, 1)  # [n_samples, 1, 1, 1]

    #         # Interpolate selected features
    #         diff = expanded_inputs[..., t, :] - expanded_baselines[..., t, :]
    #         update = expanded_baselines[..., t, :] + alphas_exp * diff
    #         mask_for_where = feature_mask_exp.expand(n_samples, n_combinations, B, F)
    #         interpolated[..., t, :] = torch.where(
    #             mask_for_where,
    #             update.expand_as(mask_for_where),
    #             interpolated[..., t, :],
    #         )

    #         # Reshape for forward pass
    #         interpolated_flat = interpolated.reshape(-1, T, F)
    #         interpolated_flat.requires_grad = True

    #         # Expand mask for all samples and combinations
    #         expanded_mask = (
    #             masks.unsqueeze(0)
    #             .unsqueeze(0)
    #             .expand(n_samples, n_combinations, B, T, F)
    #         )
    #         expanded_mask = expanded_mask.reshape(-1, T, F)

    #         # Forward pass
    #         predictions = self.model(
    #             interpolated_flat,
    #             mask=expanded_mask,
    #             timesteps=None,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, n_combinations, B, -1)

    #         gathered = predictions.gather(
    #             dim=3,
    #             index=targets.view(1, 1, B, 1).expand(n_samples, n_combinations, B, 1),
    #         ).squeeze(-1)

    #         total_for_target = gathered.sum()

    #         grads = torch.autograd.grad(
    #             outputs=total_for_target,
    #             inputs=interpolated_flat,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0]

    #         grads = grads.reshape(n_samples, n_combinations, B, T, F).mean(
    #             0
    #         )  # [n_combinations, B, T, F]

    #         diff = inputs - baselines
    #         attributions[:, t, :] = (grads[..., t, :] * diff[:, t, :]).mean(0)

    #     return attributions

    # def attribute_timewise_delta(
    #     self,
    #     inputs,
    #     baselines,
    #     targets,
    #     additional_forward_args,
    #     n_samples=50,
    #     n_combinations=10,
    # ):
    #     """
    #     Compute attributions timestep by timestep using prediction differences
    #     inputs: [B, T, F]
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]  # [B, T, F]
    #     timesteps = additional_forward_args[1]  # Original timesteps
    #     attributions = torch.zeros_like(inputs)

    #     # Get initial prediction (t=0)
    #     prev_preds = self.model(
    #         inputs[:, :1, :],
    #         mask=masks[:, :1, :],
    #         timesteps=timesteps[:, :1],
    #         return_all=additional_forward_args[2],
    #     )
    #     if prev_preds.dim() == 1:
    #         prev_preds = prev_preds.unsqueeze(1)
    #     prev_preds = prev_preds.gather(1, targets.unsqueeze(1)).squeeze(1)

    #     # For each timestep
    #     for t in range(1, T):
    #         if not masks[:, t, :].any():
    #             continue

    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(n_samples, 1, 1, 1)

    #         # Generate random combinations for current timestep
    #         obs_mask = masks[:, t, :].any(dim=0)
    #         feature_mask = torch.zeros(
    #             n_combinations, F, device=inputs.device, dtype=torch.bool
    #         )

    #         obs_features = torch.where(obs_mask)[0]
    #         if len(obs_features) > 0:
    #             for i in range(n_combinations):
    #                 n_select = torch.randint(1, len(obs_features) + 1, (1,)).item()
    #                 selected = obs_features[
    #                     torch.randperm(len(obs_features))[:n_select]
    #                 ]
    #                 feature_mask[i, selected] = True

    #         # Expand inputs and baselines up to current timestep
    #         expanded_inputs = inputs[:, : t + 1].unsqueeze(0).unsqueeze(0)
    #         expanded_baselines = baselines[:, : t + 1].unsqueeze(0).unsqueeze(0)

    #         interpolated = expanded_inputs.expand(
    #             n_samples, n_combinations, B, t + 1, F
    #         ).clone()
    #         interpolated[..., t, :] = (
    #             expanded_baselines[..., t : t + 1, :]
    #             .expand(n_samples, n_combinations, B, -1, F)
    #             .squeeze(3)
    #         )

    #         feature_mask_exp = feature_mask.view(1, n_combinations, 1, F)
    #         alphas_exp = alphas.view(n_samples, 1, 1, 1)

    #         diff = expanded_inputs[..., t, :] - expanded_baselines[..., t, :]
    #         update = expanded_baselines[..., t, :] + alphas_exp * diff
    #         mask_for_where = feature_mask_exp.expand(n_samples, n_combinations, B, F)
    #         interpolated[..., t, :] = torch.where(
    #             mask_for_where,
    #             update.expand_as(mask_for_where),
    #             interpolated[..., t, :],
    #         )

    #         interpolated_flat = interpolated.reshape(-1, t + 1, F)
    #         interpolated_flat.requires_grad = True

    #         # Expand mask and timesteps
    #         expanded_mask = (
    #             masks[:, : t + 1]
    #             .unsqueeze(0)
    #             .unsqueeze(0)
    #             .expand(n_samples, n_combinations, B, t + 1, F)
    #         )
    #         expanded_mask = expanded_mask.reshape(-1, t + 1, F)

    #         expanded_timesteps = (
    #             timesteps[:, : t + 1]
    #             .unsqueeze(0)
    #             .unsqueeze(0)
    #             .expand(n_samples, n_combinations, B, t + 1)
    #         )
    #         expanded_timesteps = expanded_timesteps.reshape(-1, t + 1)

    #         predictions = self.model(
    #             interpolated_flat,
    #             mask=expanded_mask,
    #             timesteps=expanded_timesteps,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, n_combinations, B, -1)

    #         gathered = predictions.gather(
    #             dim=3,
    #             index=targets.view(1, 1, B, 1).expand(n_samples, n_combinations, B, 1),
    #         ).squeeze(-1)

    #         pred_diff = gathered - prev_preds.view(1, 1, B)

    #         # Update previous predictions
    #         with torch.no_grad():
    #             current_preds = self.model(
    #                 inputs[:, : t + 1],
    #                 mask=masks[:, : t + 1],
    #                 timesteps=timesteps[:, : t + 1],
    #                 return_all=additional_forward_args[2],
    #             )
    #             if current_preds.dim() == 1:
    #                 current_preds = current_preds.unsqueeze(1)
    #             prev_preds = current_preds.gather(1, targets.unsqueeze(1)).squeeze(1)

    #         grads = torch.autograd.grad(
    #             outputs=pred_diff.sum(),
    #             inputs=interpolated_flat,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0]

    #         grads = grads.reshape(n_samples, n_combinations, B, t + 1, F).mean(0)

    #         diff = inputs - baselines
    #         step_size = 1.0 / (n_samples - 1)
    #         attributions[:, t, :] = (grads[..., t, :] * diff[:, t, :]).mean(
    #             0
    #         ) * step_size

    #         attribution_sum_t = attributions[:, t, :].sum(axis=-1)  # [B]
    #         pred_diff_t = (gathered - prev_preds.view(1, 1, B)).sum(axis=(0, 1))  # [B]

    #         print(f"{pred_diff_t.shape=}")
    #         print(f"{attribution_sum_t.shape=}")
    #         print(f"{pred_diff_t=}")
    #         print(f"{attribution_sum_t=}")

    #     return attributions

    # def attribute_timewise_delta_current(
    #     self,
    #     inputs,
    #     baselines,
    #     targets,
    #     additional_forward_args,
    #     n_samples=50,
    #     n_combinations=10,
    # ):
    #     """
    #     Compute attributions timestep by timestep using prediction differences with carry-forward baseline
    #     Args:
    #         inputs: Input tensor [B, T, F]
    #         targets: Target indices [B]
    #         additional_forward_args: Tuple containing (masks, timesteps, return_all)
    #             - masks: Boolean tensor [B, T, F]
    #             - timesteps: Timestep tensor [B, T]
    #             - return_all: Boolean flag for model output
    #         n_samples: Number of samples for integral approximation
    #         n_combinations: Number of feature combinations to try
    #     Returns:
    #         attributions: Attribution tensor [B, T, F]
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]  # [B, T, F]
    #     timesteps = additional_forward_args[1]  # Original timesteps
    #     attributions = torch.zeros_like(inputs)

    #     # For each timestep
    #     for t in range(1, T):
    #         if not masks[:, t, :].any():
    #             continue

    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(n_samples, 1, 1, 1)

    #         # Generate random combinations for current timestep
    #         obs_mask = masks[:, t, :].any(dim=0)
    #         feature_mask = torch.zeros(
    #             n_combinations, F, device=inputs.device, dtype=torch.bool
    #         )

    #         obs_features = torch.where(obs_mask)[0]
    #         if len(obs_features) > 0:
    #             for i in range(n_combinations):
    #                 n_select = torch.randint(1, len(obs_features) + 1, (1,)).item()
    #                 selected = obs_features[
    #                     torch.randperm(len(obs_features))[:n_select]
    #                 ]
    #                 feature_mask[i, selected] = True

    #         # Create carry-forward baseline by copying the last timestep's values
    #         baseline_current = inputs[:, : t + 1].clone()
    #         baseline_current[:, t, :] = inputs[:, t - 1, :]  # Carry forward from t-1

    #         # Expand inputs up to current timestep with proper dimensions
    #         expanded_inputs = (
    #             inputs[:, : t + 1].unsqueeze(0).unsqueeze(0)
    #         )  # [1, 1, B, t+1, F]
    #         expanded_baselines = baseline_current.unsqueeze(0).unsqueeze(
    #             0
    #         )  # [1, 1, B, t+1, F]

    #         interpolated = expanded_inputs.expand(
    #             n_samples, n_combinations, B, t + 1, F
    #         ).clone()
    #         interpolated[..., t, :] = (
    #             expanded_baselines[..., t : t + 1, :]
    #             .expand(n_samples, n_combinations, B, -1, F)
    #             .squeeze(3)
    #         )

    #         feature_mask_exp = feature_mask.view(1, n_combinations, 1, F)
    #         alphas_exp = alphas.view(n_samples, 1, 1, 1)

    #         diff = expanded_inputs[..., t, :] - expanded_baselines[..., t, :]
    #         update = expanded_baselines[..., t, :] + alphas_exp * diff
    #         mask_for_where = feature_mask_exp.expand(n_samples, n_combinations, B, F)
    #         interpolated[..., t, :] = torch.where(
    #             mask_for_where,
    #             update.expand_as(mask_for_where),
    #             interpolated[..., t, :],
    #         )

    #         interpolated_flat = interpolated.reshape(-1, t + 1, F)
    #         interpolated_flat.requires_grad = True

    #         # Expand mask and timesteps
    #         expanded_mask = (
    #             masks[:, : t + 1]
    #             .unsqueeze(0)
    #             .unsqueeze(0)
    #             .expand(n_samples, n_combinations, B, t + 1, F)
    #         )
    #         expanded_mask = expanded_mask.reshape(-1, t + 1, F)

    #         expanded_timesteps = (
    #             timesteps[:, : t + 1]
    #             .unsqueeze(0)
    #             .unsqueeze(0)
    #             .expand(n_samples, n_combinations, B, t + 1)
    #         )
    #         expanded_timesteps = expanded_timesteps.reshape(-1, t + 1)

    #         # Calculate prev_preds using current timestep but with carry-forward values
    #         with torch.no_grad():
    #             prev_preds = self.model(
    #                 baseline_current,
    #                 mask=masks[:, : t + 1],
    #                 timesteps=timesteps[:, : t + 1],
    #                 return_all=additional_forward_args[2],
    #             )
    #             if prev_preds.dim() == 1:
    #                 prev_preds = prev_preds.unsqueeze(1)
    #             prev_preds = prev_preds.gather(1, targets.unsqueeze(1)).squeeze(1)

    #         # Get predictions for interpolated inputs
    #         predictions = self.model(
    #             interpolated_flat,
    #             mask=expanded_mask,
    #             timesteps=expanded_timesteps,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, n_combinations, B, -1)

    #         gathered = predictions.gather(
    #             dim=3,
    #             index=targets.view(1, 1, B, 1).expand(n_samples, n_combinations, B, 1),
    #         ).squeeze(-1)

    #         pred_diff = gathered - prev_preds.view(1, 1, B)

    #         grads = torch.autograd.grad(
    #             outputs=pred_diff.sum(),
    #             inputs=interpolated_flat,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0]

    #         grads = grads.reshape(n_samples, n_combinations, B, t + 1, F).mean(0)

    #         # Calculate attribution using the difference at current timestep only
    #         current_diff = inputs[:, t, :] - baseline_current[:, t, :]  # [B, F]
    #         attributions[:, t, :] = (grads[..., t, :] * current_diff).mean(0)

    #         # attribution_sum_t = attributions[:, t, :].sum(axis=-1)  # [B]
    #         # pred_diff_t = (gathered - prev_preds.view(1, 1, B)).sum(axis=(0, 1))  # [B]

    #         # print(f"{pred_diff_t.shape=}")
    #         # print(f"{attribution_sum_t.shape=}")
    #         # print(f"{pred_diff_t=}")
    #         # print(f"{attribution_sum_t=}")

    #     return attributions

    # def attribute_timewise_shap(
    #     self,
    #     inputs,
    #     baselines,
    #     targets,
    #     additional_forward_args,
    #     n_samples=50,
    #     n_combinations=10,
    # ):
    #     """
    #     Compute attributions using KernelSHAP with efficient batching
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]
    #     timesteps = additional_forward_args[1]
    #     attributions = torch.zeros_like(inputs)

    #     # Process each timestep
    #     for t in range(1, T):
    #         if not masks[:, t, :].any():
    #             continue

    #         # Get observed features
    #         obs_mask = masks[:, t, :].any(dim=0)  # [F]
    #         obs_features = torch.where(obs_mask)[0]
    #         if len(obs_features) == 0:
    #             continue

    #         # Generate random coalitions directly for observed features
    #         n_features = len(obs_features)
    #         n_coalitions = min(2**n_features, n_combinations)

    #         # Generate random binary matrix for observed features only
    #         coalitions = (
    #             torch.rand(n_coalitions, n_features, device=inputs.device) > 0.5
    #         )

    #         # Convert to full feature space
    #         full_coalitions = torch.zeros(
    #             n_coalitions, F, device=inputs.device, dtype=torch.bool
    #         )
    #         full_coalitions[:, obs_features] = coalitions

    #         # Create base input tensor for all coalitions
    #         model_inputs = (
    #             inputs[:, : t + 1].clone().unsqueeze(0).repeat(n_coalitions, 1, 1, 1)
    #         )

    #         # Create timestep tensor with coalitions
    #         timestep_inputs = torch.where(
    #             full_coalitions.unsqueeze(1),
    #             inputs[:, t].unsqueeze(0),
    #             baselines[:, t].unsqueeze(0),
    #         )

    #         # Assign timestep values
    #         model_inputs[:, :, t, :] = timestep_inputs

    #         # Reshape and prepare for model
    #         model_inputs = model_inputs.reshape(-1, t + 1, F)
    #         expanded_masks = (
    #             masks[:, : t + 1]
    #             .unsqueeze(0)
    #             .expand(n_coalitions, -1, -1, -1)
    #             .reshape(-1, t + 1, F)
    #         )
    #         expanded_timesteps = (
    #             timesteps[:, : t + 1]
    #             .unsqueeze(0)
    #             .expand(n_coalitions, -1, -1)
    #             .reshape(-1, t + 1)
    #         )

    #         # Forward pass
    #         predictions = self.model(
    #             model_inputs,
    #             mask=expanded_masks,
    #             timesteps=expanded_timesteps,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_coalitions, B, -1)
    #         predictions = predictions.gather(
    #             2, targets.view(1, B, 1).expand(n_coalitions, -1, -1)
    #         ).squeeze(-1)

    #         # Compute attributions for each observed feature
    #         for i, feature in enumerate(obs_features):
    #             with_feature = coalitions[:, i]
    #             without_feature = ~with_feature

    #             if with_feature.any() and without_feature.any():
    #                 with_preds = predictions[with_feature].mean(0)
    #                 without_preds = predictions[without_feature].mean(0)
    #                 attributions[:, t, feature] = with_preds - without_preds

    #     return attributions

    # def attribute_pointwise_with_groups(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     Modified pointwise IG that also zeroizes:
    #     1. Other features at the same timestep
    #     2. The carry-forward group within the same feature

    #     inputs: [B, T, F]
    #     """
    #     B, T, F = inputs.shape
    #     masks = additional_forward_args[0]  # [B, T, F]
    #     attributions = torch.zeros_like(inputs)

    #     # Get all observation points
    #     batch_idx, time_idx, feat_idx = torch.where(masks == 1)
    #     alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #     alphas = alphas.view(n_samples, 1, 1, 1)

    #     # Expand inputs and baselines once
    #     expanded_inputs = inputs.unsqueeze(0).expand(n_samples, B, T, F)
    #     expanded_baselines = baselines.unsqueeze(0).expand(n_samples, B, T, F)

    #     # For each observation point
    #     for b, t, f in zip(batch_idx, time_idx, feat_idx):
    #         # Find group end for this point
    #         next_t = time_idx[(batch_idx == b) & (feat_idx == f) & (time_idx > t)]
    #         end = next_t[0].item() if len(next_t) > 0 else T

    #         # Create interpolated inputs
    #         interpolated = expanded_inputs.clone()

    #         # 1. Zeroize other features at timestep t
    #         interpolated[:, b : b + 1, t : t + 1, :] = expanded_baselines[
    #             :, b : b + 1, t : t + 1, :
    #         ]

    #         # 2. Zeroize the carry-forward group
    #         interpolated[:, b : b + 1, t:end, f : f + 1] = expanded_baselines[
    #             :, b : b + 1, t:end, f : f + 1
    #         ]

    #         # 3. Interpolate only the target point
    #         interpolated[:, b : b + 1, t : t + 1, f : f + 1] = expanded_baselines[
    #             :, b : b + 1, t : t + 1, f : f + 1
    #         ] + alphas * (
    #             expanded_inputs[:, b : b + 1, t : t + 1, f : f + 1]
    #             - expanded_baselines[:, b : b + 1, t : t + 1, f : f + 1]
    #         )

    #         interpolated.requires_grad = True

    #         # Forward pass
    #         expanded_mask = masks.unsqueeze(0).expand(n_samples, -1, -1, -1)
    #         expanded_mask = expanded_mask.reshape(-1, T, F)

    #         predictions = self.model(
    #             interpolated.view(-1, T, F),
    #             mask=expanded_mask,
    #             timesteps=None,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, B, -1)

    #         # Gather target predictions
    #         gathered = predictions.gather(
    #             dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #         ).squeeze(-1)

    #         total_for_target = gathered.sum()

    #         # Compute gradients
    #         grads = torch.autograd.grad(
    #             outputs=total_for_target,
    #             inputs=interpolated,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0].mean(
    #             dim=0
    #         )  # Average over samples

    #         # Store attribution for the point
    #         attributions[b, t, f] = grads[b, t, f] * (
    #             inputs[b, t, f] - baselines[b, t, f]
    #         )

    #     return attributions

    # def attribute_featurewise(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     Feature-wise Integrated Gradients for time series.

    #     inputs:  [B, T, F] - batch, timesteps, features
    #     baselines:  [B, T, F] - usually zeros
    #     targets:  [B] - integer class targets
    #     additional_forward_args: unused except . . . [2] for 'return_all'
    #     n_samples: number of interpolation steps
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, F = inputs.shape
    #     attributions = torch.zeros_like(inputs)

    #     # Iterate over features
    #     for f in range(F):
    #         # -------------------------------------------------
    #         # 1) Build interpolation from baseline --> inputs for this feature
    #         # -------------------------------------------------
    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(n_samples, 1, 1, 1)  # shape: [n_samples, 1, 1, 1]

    #         # Expand inputs and baselines
    #         expanded_inputs = inputs.unsqueeze(0).expand(
    #             n_samples, B, T, F
    #         )  # [n_samples, B, T, F]
    #         expanded_baselines = baselines.unsqueeze(0).expand(
    #             n_samples, B, T, F
    #         )  # [n_samples, B, T, F]

    #         # Create interpolated inputs where only feature f varies
    #         interpolated = expanded_inputs.clone()
    #         interpolated[..., f : f + 1] = expanded_baselines[
    #             ..., f : f + 1
    #         ] + alphas * (
    #             expanded_inputs[..., f : f + 1] - expanded_baselines[..., f : f + 1]
    #         )

    #         # Add small noise to feature f
    #         noise = torch.randn_like(interpolated[..., f : f + 1]) * 1e-4
    #         interpolated[..., f : f + 1] = interpolated[..., f : f + 1] + noise

    #         interpolated.requires_grad = True

    #         # -------------------------------------------------
    #         # 2) Forward pass & gather target predictions
    #         # -------------------------------------------------
    #         predictions = self.model(
    #             interpolated.view(-1, T, F),
    #             mask=None,
    #             timesteps=None,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, B, -1)

    #         # Gather predictions for target class
    #         gathered = predictions.gather(
    #             dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #         ).squeeze(
    #             -1
    #         )  # shape [n_samples, B]

    #         total_for_target = gathered.sum()

    #         # -------------------------------------------------
    #         # 3) Compute gradients
    #         # -------------------------------------------------
    #         grad = torch.autograd.grad(
    #             outputs=total_for_target,
    #             inputs=interpolated,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[
    #             0
    #         ]  # shape: [n_samples, B, T, F]

    #         # Average gradients over samples
    #         grads = grad.mean(dim=0)  # [B, T, F]

    #         # -------------------------------------------------
    #         # 4) Build attribution for feature f
    #         # -------------------------------------------------
    #         # Only keep gradients for feature f
    #         attributions[..., f] = grads[..., f] * (inputs[..., f] - baselines[..., f])

    #     return attributions

    # def attribute_timewise(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     Time-wise IG: Interpolates all features at each timestep together
    #     inputs: [B, T, F]
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, F = inputs.shape
    #     attributions = torch.zeros_like(inputs)

    #     # Iterate over timesteps
    #     for t in range(T):
    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(n_samples, 1, 1, 1)

    #         # Expand inputs and baselines
    #         expanded_inputs = inputs.unsqueeze(0).expand(n_samples, B, T, F)
    #         expanded_baselines = baselines.unsqueeze(0).expand(n_samples, B, T, F)

    #         # Create interpolated inputs where only timestep t varies
    #         interpolated = expanded_inputs.clone()
    #         interpolated[:, :, t : t + 1, :] = expanded_baselines[
    #             :, :, t : t + 1, :
    #         ] + alphas * (
    #             expanded_inputs[:, :, t : t + 1, :]
    #             - expanded_baselines[:, :, t : t + 1, :]
    #         )

    #         interpolated.requires_grad = True

    #         # Forward pass
    #         predictions = self.model(
    #             interpolated.view(-1, T, F),
    #             mask=None,
    #             timesteps=None,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, B, -1)

    #         gathered = predictions.gather(
    #             dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #         ).squeeze(-1)

    #         total_for_target = gathered.sum()

    #         # Compute gradients
    #         grad = torch.autograd.grad(
    #             outputs=total_for_target,
    #             inputs=interpolated,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[0]

    #         # Average gradients over samples
    #         grads = grad.mean(dim=0)

    #         # Store attributions for this timestep
    #         attributions[:, t, :] = grads[:, t, :] * (
    #             inputs[:, t, :] - baselines[:, t, :]
    #         )

    #     return attributions

    # def attribute_timewise_feature_coalition(
    #     self,
    #     inputs,
    #     baselines,
    #     targets,
    #     additional_forward_args,
    #     n_samples=50,
    #     n_groups=20,
    # ):
    #     """
    #     Time-wise IG with Correlation-Based Feature Clustering and Parallelized Processing.
    #     Args:
    #         inputs: [B, T, F] - Input tensor (Batch, Time, Features).
    #         baselines: [B, T, F] - Baseline tensor (same shape as inputs).
    #         targets: [B] - Target indices for each sample in the batch.
    #         additional_forward_args: Additional arguments for the forward pass.
    #         n_samples: Number of interpolation steps for IG.
    #         n_groups: Number of feature groups to process.
    #     Returns:
    #         attributions: [B, T, F] - Final attributions considering feature clusters.
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, F = inputs.shape
    #     attributions = torch.zeros_like(inputs)

    #     # Compute correlation matrix
    #     features = inputs.reshape(-1, F)
    #     features_std = (features - features.mean(0)) / (features.std(0) + 1e-8)
    #     feature_corr = torch.corrcoef(features_std.T)
    #     feature_corr = torch.nan_to_num(feature_corr, nan=0.0).abs()

    #     # Initialize partitions with random sampling
    #     partitions = []
    #     uncovered_features = set(range(F))

    #     for _ in range(n_groups):
    #         # Sample group size
    #         group_size = np.random.randint(int(F * 3 / 4), F)

    #         # Sample features based on correlations
    #         if uncovered_features:
    #             center = np.random.choice(list(uncovered_features))
    #             uncovered_features.remove(center)
    #             correlations = feature_corr[center]
    #         else:
    #             center = np.random.randint(0, F)
    #             correlations = feature_corr[center]

    #         # Sample rest of group based on correlation
    #         candidates = [(i, correlations[i]) for i in range(F) if i != center]
    #         candidates.sort(key=lambda x: x[1], reverse=True)
    #         group = [center] + [idx for idx, _ in candidates[: group_size - 1]]

    #         partitions.append(torch.tensor(group, device=inputs.device))

    #     # Ensure all features are covered
    #     for feat in uncovered_features:
    #         max_corr = -1
    #         best_group = 0
    #         for i, group in enumerate(partitions):
    #             corr = feature_corr[feat][group].mean()
    #             if corr > max_corr:
    #                 max_corr = corr
    #                 best_group = i

    #         partitions[best_group] = torch.cat(
    #             [partitions[best_group], torch.tensor([feat], device=inputs.device)]
    #         )

    #     # Calculate feature group counts
    #     feature_group_counts = torch.zeros(F, device=inputs.device)
    #     for group in partitions:
    #         feature_group_counts[group] += 1

    #     # Step 2: Iterate over timesteps
    #     for t in range(T):
    #         alphas = torch.linspace(0, 1, n_samples, device=inputs.device)
    #         alphas = alphas.view(n_samples, 1, 1, 1)

    #         # Expand inputs and baselines
    #         expanded_inputs = inputs.unsqueeze(0).expand(n_samples, B, T, F)
    #         expanded_baselines = baselines.unsqueeze(0).expand(n_samples, B, T, F)

    #         # Parallelized processing of partitions
    #         all_group_masks = []
    #         for group in partitions:
    #             group_mask = torch.zeros(F, device=inputs.device)
    #             group_mask[group] = 1
    #             all_group_masks.append(group_mask)

    #         group_masks = torch.stack(all_group_masks).view(1, n_groups, 1, 1, F)
    #         group_masks = group_masks.expand(n_samples, n_groups, B, 1, F)

    #         # Create interpolated inputs
    #         interpolated = (
    #             expanded_inputs.unsqueeze(1).expand(-1, n_groups, -1, -1, -1).clone()
    #         )
    #         interpolated[:, :, :, t : t + 1, :] = (
    #             expanded_inputs.unsqueeze(1)[:, :, :, t : t + 1, :] * (1 - group_masks)
    #             + (
    #                 expanded_baselines.unsqueeze(1)[:, :, :, t : t + 1, :]
    #                 + alphas.unsqueeze(1)
    #                 * (
    #                     expanded_inputs.unsqueeze(1)[:, :, :, t : t + 1, :]
    #                     - expanded_baselines.unsqueeze(1)[:, :, :, t : t + 1, :]
    #                 )
    #             )
    #             * group_masks
    #         )
    #         interpolated.requires_grad = True

    #         # Forward pass
    #         predictions = self.model(
    #             interpolated.view(-1, T, F),
    #             mask=None,
    #             timesteps=None,
    #             return_all=additional_forward_args[2],
    #         )

    #         if predictions.dim() == 1:
    #             predictions = predictions.unsqueeze(1)
    #         predictions = predictions.view(n_samples, n_groups, B, -1)

    #         gathered = predictions.gather(
    #             dim=3,
    #             index=targets.unsqueeze(0)
    #             .unsqueeze(0)
    #             .unsqueeze(-1)
    #             .expand(n_samples, n_groups, B, 1),
    #         ).squeeze(-1)

    #         total_for_target = gathered.sum()

    #         # Compute gradients
    #         grad = torch.autograd.grad(
    #             outputs=total_for_target,
    #             inputs=interpolated,
    #             retain_graph=True,
    #             allow_unused=True,
    #         )[
    #             0
    #         ]  # [n_samples, n_groups, B, T, F]

    #         # Average over samples
    #         grads = grad.mean(dim=0)  # [n_groups, B, T, F]

    #         # Sum attributions from each group and average by count
    #         feature_attributions = torch.zeros((B, F), device=inputs.device)
    #         for g, group in enumerate(partitions):
    #             feature_attributions[:, group] += grads[g, :, t, group]

    #         timestep_attributions = feature_attributions / (feature_group_counts + 1e-8)
    #         attributions[:, t, :] = timestep_attributions * (
    #             inputs[:, t, :] - baselines[:, t, :]
    #         )

    #     return attributions

    # def attribute_time_aware(
    #     self, inputs, baselines, targets, additional_forward_args, n_samples=50
    # ):
    #     """
    #     inputs:  [B, T, D]  (batch, time, features)
    #     baselines: [B, T, D]  (same shape)
    #     targets: [B]  (integer target indices)
    #     """

    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, D = inputs.shape
    #     device = inputs.device

    #     # Create alpha for interpolation (shape [n_samples, 1, 1, 1])
    #     alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)

    #     # Interpolation from start_pos to inputs + noise
    #     start_pos = baselines
    #     expanded_inputs = inputs.unsqueeze(0)  # [1, B, T, D]
    #     expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]
    #     noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)
    #     noise = torch.randn_like(noisy_inputs) * 1e-4
    #     noisy_inputs = noisy_inputs + noise

    #     # -------------------------------------------------
    #     # Time-series aware masking: fix a random time slice
    #     # -------------------------------------------------
    #     # For each sample i (0..n_samples-1) and each batch b (0..B-1),
    #     # pick a random segment of length seg_len and fix it to real input.

    #     seg_len = max(1, T // 3)  # e.g., fix ~1/3 of the time steps
    #     # We'll create a mask of shape [n_samples, B, T, D]
    #     # 1 => use the interpolation, 0 => fix to real input
    #     time_mask = torch.ones_like(noisy_inputs)  # start with all ones

    #     for i in range(n_samples):
    #         for b in range(B):
    #             t_start = torch.randint(low=0, high=T - seg_len + 1, size=(1,)).item()
    #             t_end = t_start + seg_len
    #             # set [t_start : t_end] to 0 => fix them to real input
    #             time_mask[i, b, t_start:t_end, :] = 0

    #     fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
    #     # Broadcast if needed
    #     # Combine
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

    #     masked_inputs.requires_grad = True

    #     # Forward pass
    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     # shape => [n_samples*B, num_classes] or [n_samples*B] if 1D output
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)  # [n_samples*B, 1]

    #     predictions = predictions.view(n_samples, B, -1)
    #     gathered = predictions.gather(
    #         dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)
    #     total_for_target = gathered.sum()

    #     grad = torch.autograd.grad(
    #         outputs=total_for_target,
    #         inputs=masked_inputs,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[
    #         0
    #     ]  # [n_samples, B, T, D]

    #     # Average gradient over n_samples
    #     grads = grad.sum(dim=0)  # [B, T, D]

    #     # -------------------------------------------------
    #     # 5) Build final attributions
    #     # -------------------------------------------------
    #     # For positions we left "interpolating,"
    #     # attributions = grads * (inputs - baselines).
    #     # For positions that were forced to real input,
    #     # we want zero attributions.
    #     # One simple approach is to also average the mask
    #     # across the n_samples.
    #     # Alternatively, if you want *every sample* to fix
    #     # that position, you can incorporate a different rule.

    #     # If you want each example's final attributions to
    #     # reflect the fraction of times it was “interpolated”:
    #     mask_avg = time_mask.sum(dim=0)  # shape [B, T, D]
    #     raw_attr = grads * (inputs - baselines)
    #     # Zero out contributions in proportion to how often
    #     # they were "fixed".
    #     final_attributions = raw_attr / mask_avg

    #     return final_attributions

    # def attribute_hierarchical_zeroing(
    #     self,
    #     inputs,
    #     baselines,
    #     targets,
    #     additional_forward_args,
    #     n_samples=50,
    #     n_hierarchy=3,
    # ):
    #     """
    #     Hierarchical Integrated Gradients with Adaptive Zeroing (Time- and Feature-Wise, Parallelized).

    #     Args:
    #         inputs: [B, T, F] - Time-series input (Batch, Time, Features).
    #         baselines: [B, T, F] - Baseline inputs (same shape as inputs).
    #         targets: [B] - Target indices for each sample in the batch.
    #         additional_forward_args: Additional arguments for the forward pass.
    #         n_samples: Number of interpolation steps for IG.
    #         n_hierarchy: Number of hierarchical levels for adaptive zeroing.

    #     Returns:
    #         attributions: [B, T, F] - Final attributions after hierarchical refinement.
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, F = inputs.shape
    #     device = inputs.device
    #     attributions = torch.zeros_like(inputs)

    #     # Start with an all-zero baseline
    #     current_baseline = torch.zeros_like(baselines)
    #     mask = torch.zeros_like(
    #         inputs, dtype=torch.bool
    #     )  # Initial mask: all points are zeroed

    #     for level in range(n_hierarchy):
    #         print(f"Level {level+1}/{n_hierarchy}")

    #         # Compute IG for the current baseline
    #         current_attributions = self.attribute_timewise(
    #             inputs, current_baseline, targets, additional_forward_args, n_samples
    #         )

    #         # Compute attribution importance for all points (time and features combined)
    #         attr_importance = current_attributions.abs()  # [B, T, F]
    #         flat_attr_importance = attr_importance.view(B, -1)  # Flatten to [B, T*F]
    #         flat_ranking = torch.argsort(
    #             flat_attr_importance, dim=1, descending=True
    #         )  # [B, T*F]

    #         # Determine how many points to zero (time- and feature-wise)
    #         n_to_zero = T * F // (2 ** (level + 1))  # Half the remaining points
    #         most_important_indices = flat_ranking[:, :n_to_zero]  # [B, n_to_zero]

    #         # Compute corresponding time and feature indices for all batches
    #         time_indices = most_important_indices // F  # [B, n_to_zero]
    #         feature_indices = most_important_indices % F  # [B, n_to_zero]

    #         # Create a batch index tensor
    #         batch_indices = (
    #             torch.arange(B, device=inputs.device)
    #             .view(-1, 1)
    #             .expand_as(time_indices)
    #         )  # [B, n_to_zero]

    #         # Update the mask in parallel
    #         new_mask = mask.clone()  # Clone the current mask
    #         new_mask[batch_indices, time_indices, feature_indices] = (
    #             True  # Mark these points in the mask
    #         )

    #         # Update baseline: Set the newly selected points to their original values in the baseline
    #         current_baseline = torch.where(new_mask, baselines, current_baseline)

    #         # Add attributions only for the points zeroed in this stage
    #         attributions += current_attributions * new_mask.float()

    #         # Update the mask for the next stage
    #         mask = new_mask  # Accumulate points already attributed

    #     return attributions

    # def attribute_time_aware_random_segments(
    #     self,
    #     inputs: torch.Tensor,
    #     baselines: torch.Tensor,
    #     targets: torch.Tensor,
    #     additional_forward_args,
    #     n_samples: int = 50,
    # ):
    #     """
    #     inputs:  shape [B, T, D]
    #     baselines:  shape [B, T, D]
    #     targets:  shape [B]  (one target class per batch item)
    #     additional_forward_args:  used only for e.g., return_all
    #     n_samples:  number of alpha interpolation steps
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, D = inputs.shape
    #     device = inputs.device

    #     # -------------------------------------------------
    #     # 1) Build interpolation from baseline --> inputs
    #     # -------------------------------------------------
    #     alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
    #     # Start from "start_pos" so alpha=0 means baselines,
    #     # alpha=1 means inputs (plus small noise).
    #     start_pos = baselines
    #     expanded_inputs = inputs.unsqueeze(0)  # [1, B, T, D]
    #     expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]
    #     noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

    #     # Add a little random noise
    #     noise = torch.randn_like(noisy_inputs) * 1e-4
    #     noisy_inputs = noisy_inputs + noise

    #     # -------------------------------------------------
    #     # 2) Time-series aware masking (random contiguous segments)
    #     # -------------------------------------------------
    #     # For each interpolation i in [0..n_samples-1] and each batch item b,
    #     # randomly pick:
    #     #  - seg_len in [1, T] (or some smaller range if desired)
    #     #  - a start index t_start
    #     # Then fix that segment [t_start : t_start+seg_len] to the real input.
    #     #
    #     # We'll define time_mask[i, b, t, d] = 1 => use interpolation,
    #     #                                        = 0 => fix to real input.

    #     time_mask = torch.ones_like(noisy_inputs)  # shape [n_samples, B, T, D]

    #     for i in range(n_samples):
    #         for b in range(B):
    #             seg_len = torch.randint(low=1, high=T + 1, size=(1,)).item()
    #             # e.g., random segment length from 1..T
    #             t_start = torch.randint(low=0, high=T - seg_len + 1, size=(1,)).item()
    #             t_end = t_start + seg_len
    #             # Fix that time slice => set mask=0 for that region
    #             time_mask[i, b, t_start:t_end, :] = 0

    #     # Combine masked (fixed) portion with interpolated portion
    #     fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

    #     # Turn on gradient for only the interpolation portion
    #     masked_inputs.requires_grad = True

    #     # -------------------------------------------------
    #     # 3) Forward pass & gather target predictions
    #     # -------------------------------------------------
    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     # Reshape to [n_samples, B, num_classes]
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     # Gather the logit for the target class
    #     gathered = predictions.gather(
    #         dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(
    #         -1
    #     )  # shape [n_samples, B]

    #     # Sum over samples and batch items
    #     total_for_target = gathered.sum()

    #     # -------------------------------------------------
    #     # 4) Compute gradients w.r.t. the masked_inputs
    #     # -------------------------------------------------
    #     grad = torch.autograd.grad(
    #         outputs=total_for_target,
    #         inputs=masked_inputs,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[
    #         0
    #     ]  # shape [n_samples, B, T, D]

    #     # Average gradient over n_samples
    #     grads = grad.sum(dim=0)  # [B, T, D]

    #     # -------------------------------------------------
    #     # 5) Build final attributions
    #     # -------------------------------------------------
    #     # For positions we left "interpolating,"
    #     # attributions = grads * (inputs - baselines).
    #     # For positions that were forced to real input,
    #     # we want zero attributions.
    #     # One simple approach is to also average the mask
    #     # across the n_samples.
    #     # Alternatively, if you want *every sample* to fix
    #     # that position, you can incorporate a different rule.

    #     # If you want each example's final attributions to
    #     # reflect the fraction of times it was “interpolated”:
    #     mask_avg = time_mask.sum(dim=0)  # shape [B, T, D]
    #     raw_attr = grads * (inputs - baselines)
    #     # Zero out contributions in proportion to how often
    #     # they were "fixed".
    #     final_attributions = raw_attr / mask_avg

    #     return final_attributions

    # def attribute_random_time_segments_one_dim(
    #     self,
    #     inputs: torch.Tensor,  # [B, T, D]
    #     baselines: torch.Tensor,  # [B, T, D]
    #     targets: torch.Tensor,  # [B]
    #     additional_forward_args,
    #     n_samples: int = 50,
    #     num_segments: int = 3,  # how many time segments to fix (single dimension) per sample
    #     max_seg_len: int = None,  # optional max length for each time segment
    # ):
    #     """
    #     This version picks multiple random contiguous time segments per sample,
    #     but for exactly ONE random dimension in each segment.

    #     Steps:
    #     1) Build interpolation from baselines -> inputs (the standard IG approach).
    #     2) For each sample i, each batch item b, and each of num_segments:
    #         - pick a random dimension d
    #         - pick a random contiguous time slice [t_start : t_end)
    #         - fix that slice & dimension to real inputs => no attributions there
    #     3) Forward pass, gather target logit, compute gradients
    #     4) Multiply by (inputs - baselines)
    #     """
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, D = inputs.shape
    #     device = inputs.device

    #     # -------------------------------------------------------
    #     # 1) Build interpolation from baseline -> inputs
    #     # -------------------------------------------------------
    #     alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
    #     start_pos = baselines
    #     expanded_inputs = inputs.unsqueeze(0)  # [1, B, T, D]
    #     expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]

    #     # Interpolate
    #     noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

    #     # Add small noise
    #     noise = torch.randn_like(noisy_inputs) * 1e-4
    #     noisy_inputs = noisy_inputs + noise

    #     # -------------------------------------------------------
    #     # 2) Create random time-segment mask (single dimension)
    #     # -------------------------------------------------------
    #     # We'll build a mask of shape [n_samples, B, T, D]:
    #     #    mask=1 => use the interpolated value
    #     #    mask=0 => fix to the real input
    #     time_mask = torch.ones_like(noisy_inputs)  # [n_samples, B, T, D]

    #     if max_seg_len is None:
    #         max_seg_len = T  # up to the entire T by default

    #     for i in range(n_samples):
    #         for b in range(B):
    #             for seg_id in range(num_segments):
    #                 # Pick a random dimension:
    #                 dim_chosen = random.randint(0, D - 1)
    #                 # Pick a random segment length & start
    #                 seg_len = random.randint(1, max_seg_len)
    #                 t_start = random.randint(0, T - seg_len)
    #                 t_end = t_start + seg_len
    #                 # Fix time slice [t_start : t_end] for that single dimension
    #                 time_mask[i, b, t_start:t_end, dim_chosen] = 0

    #     # Combine: masked_inputs = time_mask * interpolated + (1-time_mask)* real_input
    #     fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

    #     masked_inputs.requires_grad = True

    #     # -------------------------------------------------------
    #     # 3) Forward pass & gather target logits
    #     # -------------------------------------------------------
    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     # Ensure shape => [n_samples, B, num_classes]
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     # Gather only the target logit for each example
    #     gathered = predictions.gather(
    #         dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(
    #         -1
    #     )  # [n_samples, B]

    #     total_for_target = gathered.sum()

    #     # -------------------------------------------------------
    #     # 4) Compute gradients w.r.t. masked_inputs
    #     # -------------------------------------------------------
    #     grad = torch.autograd.grad(
    #         outputs=total_for_target,
    #         inputs=masked_inputs,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[
    #         0
    #     ]  # shape => [n_samples, B, T, D]

    #     grads = grad.sum(dim=0)  # [B, T, D]

    #     # -------------------------------------------------
    #     # 5) Build final attributions
    #     # -------------------------------------------------
    #     # For positions we left "interpolating,"
    #     # attributions = grads * (inputs - baselines).
    #     # For positions that were forced to real input,
    #     # we want zero attributions.
    #     # One simple approach is to also average the mask
    #     # across the n_samples.
    #     # Alternatively, if you want *every sample* to fix
    #     # that position, you can incorporate a different rule.

    #     # If you want each example's final attributions to
    #     # reflect the fraction of times it was “interpolated”:
    #     mask_avg = time_mask.sum(dim=0)  # shape [B, T, D]
    #     raw_attr = grads * (inputs - baselines)
    #     # Zero out contributions in proportion to how often
    #     # they were "fixed".
    #     final_attributions = raw_attr / mask_avg

    #     return final_attributions

    # def find_segments(self,
    #         temporal_diffs: torch.Tensor, window_size: int, min_segments: int) -> torch.Tensor:
    #     """Find segment boundaries using local maxima of temporal differences."""
    #     B = temporal_diffs.size(0)
    #     device = temporal_diffs.device
        
    #     # Identify change points using difference threshold
    #     mean_diff = temporal_diffs.mean()
    #     std_diff = temporal_diffs.std()
    #     change_points = temporal_diffs > (mean_diff + std_diff)
        
    #     # Ensure minimum spacing between segments
    #     min_spacing = window_size
    #     spaced_points = torch.zeros_like(change_points)
        
    #     for b in range(B):
    #         last_point = 0
    #         for t in range(change_points.size(1)):
    #             if change_points[b, t] and (t - last_point) >= min_spacing:
    #                 spaced_points[b, t] = 1
    #                 last_point = t
        
    #     # Add start and end points
    #     boundary_scores = torch.cat([
    #         torch.ones(B, 1, device=device),
    #         spaced_points.float(),
    #         torch.ones(B, 1, device=device)
    #     ], dim=1)
        
    #     return boundary_scores

    # def attribute_random_time_segments_one_dim_same_for_batch(
    #     self,
    #     inputs: torch.Tensor,
    #     baselines: torch.Tensor,
    #     targets: torch.Tensor,
    #     additional_forward_args,
    #     n_samples: int = 50,
    #     num_segments: int = 3,
    #     max_seg_len: int = None,
    #     min_seg_len: int = None,
    # ):
    #     B, T, D = inputs.shape
    #     device = inputs.device
        
    #     max_seg_len = min(max_seg_len or T, T)
    #     min_seg_len = min_seg_len or 1
    #     actual_segments = min(num_segments, T-1)

    #     # Interpolation sampling
    #     beta_dist = torch.distributions.Beta(0.3, 1)
    #     alphas = beta_dist.sample((n_samples, B)).to(device)
    #     alphas = torch.sort(alphas, dim=0).values.view(n_samples, B, 1, 1)

    #     expanded_inputs = inputs.unsqueeze(0)
    #     expanded_baselines = baselines.unsqueeze(0)
    #     noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
    #     noisy_inputs = noisy_inputs + torch.randn_like(noisy_inputs) * 1e-4

    #     # Calculate temporal differences for sampling weights
    #     temporal_diffs = torch.abs(inputs[:, 1:] - inputs[:, :-1]).mean(dim=-1)
    #     diff_weights = F.softmax(temporal_diffs / 0.1, dim=1)  # Temperature parameter
        
    #     # Sample segment boundaries stochastically
    #     time_indices = torch.arange(T-1, device=device)
    #     boundaries = []
    #     for b in range(B):
    #         sample_points = []
    #         available_points = time_indices.clone()
    #         weights = diff_weights[b].clone()
            
    #         for _ in range(actual_segments - 1):
    #             if len(available_points) > 0:
    #                 idx = torch.multinomial(weights, 1)
    #                 point = available_points[idx]
    #                 sample_points.append(point.item())
                    
    #                 # Remove nearby points
    #                 mask = torch.abs(available_points - point) >= min_seg_len
    #                 available_points = available_points[mask]
    #                 weights = weights[mask]
    #                 if len(weights) > 0:
    #                     weights = weights / weights.sum()
                        
    #         sample_points = sorted(sample_points)
    #         boundaries.append([0] + sample_points + [T-1])
    #     boundaries = torch.tensor(boundaries, device=device)

    #     # Create masks
    #     dims = torch.randint(0, D, (n_samples, B, actual_segments), device=device)
    #     time_mask = torch.ones_like(noisy_inputs)
    #     batch_indices = torch.arange(B, device=device)
    #     sample_indices = torch.arange(n_samples, device=device)

    #     for s in range(actual_segments):
    #         start = boundaries[:, s]
    #         end = boundaries[:, s+1]
    #         seg_len = (end - start).max().item()
            
    #         indices = start.unsqueeze(0).unsqueeze(-1) + torch.arange(seg_len, device=device).unsqueeze(0).unsqueeze(0)
    #         valid_indices = indices < end.unsqueeze(-1)
    #         time_mask[
    #             sample_indices.view(-1,1,1),
    #             batch_indices.view(1,-1,1),
    #             indices * valid_indices,
    #             dims[:,:,s].unsqueeze(-1)
    #         ] = 0

    #     # Attribution calculation
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * expanded_inputs.detach()
    #     masked_inputs.requires_grad = True

    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     gathered = predictions.gather(
    #         dim=2,
    #         index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)
        
    #     gathered_per_sample = gathered.unbind(0)
    #     grad_tensors = [torch.ones_like(g) for g in gathered_per_sample]

    #     grads = torch.autograd.grad(
    #         outputs=gathered_per_sample,
    #         inputs=masked_inputs,
    #         grad_outputs=grad_tensors,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]
    #     grads[time_mask == 0] = 0

    #     raw_attr = grads * (inputs - baselines).unsqueeze(0)
    #     mask_avg = time_mask.sum(dim=0)
    #     final_attr = raw_attr / mask_avg

    #     return final_attr.mean(dim=0)

    # def attribute_random_time_segments_one_dim_same_for_batch(
    #         self,
    #         inputs: torch.Tensor,  # [B, T, D]
    #         baselines: torch.Tensor,  # [B, T, D]
    #         targets: torch.Tensor,  # [B]
    #         additional_forward_args,
    #         n_samples: int = 50,
    #         num_segments: int = 3,
    #         max_seg_len: int = None,
    #         min_seg_len: int = None,
    #         diversity_penalty: float = 0.5,  # Factor to discourage overlapping
    #         semantic_importance: torch.Tensor = None,  # Optional [T] importance weights
    #     ):
    #     B, T, D = inputs.shape
    #     device = inputs.device
    #     max_seg_len = min(max_seg_len or T, T)
    #     min_seg_len = min_seg_len or 1

    #     # Beta distribution sampling
    #     beta_dist = torch.distributions.Beta(0.1, 1.5)
    #     alphas = beta_dist.sample((n_samples, B)).to(device)
    #     alphas = torch.sort(alphas, dim=0).values.view(n_samples, B, 1, 1)

    #     expanded_inputs = inputs.unsqueeze(0)
    #     expanded_baselines = baselines.unsqueeze(0)
    #     noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
    #     noisy_inputs = noisy_inputs + torch.randn_like(noisy_inputs) * 1e-4

    #     # Calculate temporal differences for importance sampling
    #     diffs = torch.abs(inputs[0, 1:] - inputs[0, :-1]).mean(-1)  # [T-1]
    #     if semantic_importance is not None:
    #         diffs = diffs * semantic_importance[:-1]  # Apply semantic importance

    #     temp_scores = diffs + diffs.mean()  # Add mean to avoid zeros
    #     temp_scores = F.softmax(temp_scores, dim=0)  # Normalize for sampling

    #     # Sample diverse segments
    #     dims = torch.randint(0, D, (n_samples, num_segments), device=device)
    #     t_starts = torch.zeros((n_samples, num_segments), dtype=torch.long, device=device)
    #     t_ends = torch.zeros((n_samples, num_segments), dtype=torch.long, device=device)
        
    #     for i in range(n_samples):
    #         used_ranges = []  # Track used ranges to ensure diversity
    #         for j in range(num_segments):
    #             # Adjust start sampling to avoid overlap
    #             valid_start = torch.ones(T - max_seg_len + 1, device=device)
    #             for start, end in used_ranges:
    #                 overlap_penalty = torch.arange(start, min(end, T - max_seg_len + 1), device=device)
    #                 valid_start[overlap_penalty] *= diversity_penalty  # Penalize overlap

    #             start_scores = valid_start * temp_scores[:T - max_seg_len + 1]
    #             t_start = torch.multinomial(start_scores, 1).item()
    #             t_starts[i, j] = t_start

    #             # Sample end based on t_start
    #             min_end = t_start + min_seg_len
    #             max_end = min(t_start + max_seg_len, T)
    #             end_scores = temp_scores[min_end:max_end]
    #             if len(end_scores) > 0:
    #                 end_offset = torch.multinomial(end_scores, 1).item()
    #                 t_ends[i, j] = min_end + end_offset
    #             else:
    #                 t_ends[i, j] = min_end

    #             used_ranges.append((t_start, t_ends[i, j]))  # Update used ranges

    #     # Create masks
    #     time_mask = torch.ones_like(noisy_inputs)
    #     batch_indices = torch.arange(B, device=device)
    #     sample_indices = torch.arange(n_samples, device=device)

    #     for s in range(num_segments):
    #         seg_len = (t_ends[:, s] - t_starts[:, s]).max().item()
    #         indices = t_starts[:, s].unsqueeze(-1).unsqueeze(-1) + \
    #                     torch.arange(seg_len, device=device).unsqueeze(0).unsqueeze(0)
    #         valid_indices = indices < t_ends[:, s].unsqueeze(-1).unsqueeze(-1)
    #         time_mask[
    #             sample_indices.view(-1,1,1),
    #             batch_indices.view(1,-1,1),
    #             indices * valid_indices,
    #             dims[:,s].view(-1,1,1)
    #         ] = 0

    #     # Attribution calculation
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * expanded_inputs.detach()
    #     masked_inputs.requires_grad = True

    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     gathered = predictions.gather(
    #         dim=2,
    #         index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)

    #     grads = torch.autograd.grad(
    #         gathered.sum(),
    #         masked_inputs,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]
    #     grads[time_mask == 0] = 0

    #     raw_attr = grads * (inputs - baselines).unsqueeze(0)
    #     mask_avg = time_mask.sum(dim=0)
    #     final_attr = raw_attr / mask_avg

    #     return final_attr.mean(dim=0)

    # def attribute_random_time_segments_one_dim_same_for_batch(
    #     self,
    #     inputs: torch.Tensor,  # [B, T, D]
    #     baselines: torch.Tensor,  # [B, T, D]
    #     targets: torch.Tensor,  # [B]
    #     additional_forward_args,
    #     n_samples: int = 50,
    #     num_segments: int = 3,
    #     max_seg_len: int = None,
    #     min_seg_len: int = None,
    #     diversity_penalty: float = 0.5,  # Factor to discourage overlapping
    #     semantic_importance: torch.Tensor = None,  # Optional [B, T] importance weights
    # ):
    #     B, T, D = inputs.shape
    #     device = inputs.device
    #     max_seg_len = min(max_seg_len or T, T)
    #     min_seg_len = min_seg_len or 1

    #     # Beta distribution sampling
    #     beta_dist = torch.distributions.Beta(0.1, 1.5)
    #     alphas = beta_dist.sample((n_samples, B)).to(device)
    #     alphas = torch.sort(alphas, dim=0).values.view(n_samples, B, 1, 1)

    #     expanded_inputs = inputs.unsqueeze(0)
    #     expanded_baselines = baselines.unsqueeze(0)
    #     noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
    #     noisy_inputs = noisy_inputs + torch.randn_like(noisy_inputs) * 1e-4

    #     # Calculate temporal changes at multiple scales
    #     diffs_short = torch.abs(inputs[:, 1:] - inputs[:, :-1]).mean(-1)  # Local changes [B, T-1]
    #     kernel_med = 5
    #     kernel_long = 10
    #     stride = 1

    #     pad_med = kernel_med // 2
    #     pad_long = kernel_long // 2

    #     diffs_med = F.avg_pool1d(
    #         F.pad(diffs_short.unsqueeze(1), (pad_med, pad_med - 1), mode='reflect'),
    #         kernel_size=kernel_med,
    #         stride=stride
    #     ).squeeze(1)

    #     diffs_long = F.avg_pool1d(
    #         F.pad(diffs_short.unsqueeze(1), (pad_long, pad_long - 1), mode='reflect'),
    #         kernel_size=kernel_long,
    #         stride=stride
    #     ).squeeze(1)

    #     # Truncate to ensure matching sizes
    #     min_length = min(diffs_short.size(1), diffs_med.size(1), diffs_long.size(1))
    #     diffs_short = diffs_short[:, :min_length]
    #     diffs_med = diffs_med[:, :min_length]
    #     diffs_long = diffs_long[:, :min_length]

    #     change_scores = (diffs_short + diffs_med + diffs_long) / 3
    #     if semantic_importance is not None:
    #         change_scores *= semantic_importance[:, :change_scores.size(1)]

    #     temp_scores = F.softmax(change_scores, dim=1)  # [B, T-1]

    #     # Sample diverse segments
    #     t_starts = torch.zeros((B, num_segments), dtype=torch.long, device=device)
    #     t_ends = torch.zeros((B, num_segments), dtype=torch.long, device=device)

    #     for i in range(num_segments):
    #         mask = torch.ones_like(temp_scores)  # temp_scores: [B, T-1]
    #         if i > 0:  # Avoid overlap with previous segments
    #             for prev_start in t_starts[:, :i].unbind(dim=1):
    #                 for b in range(B):  # Apply mask batch-wise
    #                     start_range = torch.arange(
    #                         max(0, prev_start[b] - max_seg_len),
    #                         min(T - 1, prev_start[b] + max_seg_len),
    #                         device=device
    #                     )
    #                     mask[b, start_range] = 0  # Penalize overlap

    #         masked_scores = temp_scores * mask  # [B, T-1]

    #         for batch_idx in range(B):
    #             if masked_scores[batch_idx].sum() > 0:  # Ensure valid probabilities
    #                 sampled_start = torch.multinomial(masked_scores[batch_idx], 1).item()
    #             else:
    #                 # Fallback: Assign a random index if all probabilities are zero
    #                 sampled_start = torch.randint(0, T - 1, (1,), device=device).item()
    #                 print(f"Fallback triggered for batch {batch_idx} at segment {i}.")
    #             t_starts[batch_idx, i] = sampled_start

    #             # Calculate t_ends
    #             min_end = sampled_start + min_seg_len
    #             max_end = min(sampled_start + max_seg_len, T)
    #             if max_end > min_end:
    #                 end_scores = temp_scores[batch_idx, min_end:max_end]
    #                 if end_scores.sum() > 0:  # Ensure valid probabilities for end_scores
    #                     end_offset = torch.multinomial(end_scores, 1).item()
    #                     t_ends[batch_idx, i] = min_end + end_offset
    #                 else:
    #                     t_ends[batch_idx, i] = min_end  # Default to minimum if no valid scores
    #             else:
    #                 t_ends[batch_idx, i] = min_end  # Default if segment range is too small

    #     # Create masks
    #     time_mask = torch.ones_like(noisy_inputs)
    #     for s in range(num_segments):
    #         seg_len = (t_ends[:, s] - t_starts[:, s]).max().item()
    #         indices = t_starts[:, s].unsqueeze(-1).unsqueeze(-1) + \
    #                 torch.arange(seg_len, device=device).unsqueeze(0).unsqueeze(0)
    #         valid_indices = indices < t_ends[:, s].unsqueeze(-1).unsqueeze(-1)
    #         time_mask[
    #             torch.arange(n_samples, device=device).view(-1, 1, 1),
    #             torch.arange(B, device=device).view(1, -1, 1),
    #             indices * valid_indices,
    #         ] = 0

    #     # Attribution calculation
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * expanded_inputs.detach()
    #     masked_inputs.requires_grad = True

    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     gathered = predictions.gather(
    #         dim=2,
    #         index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)

    #     grads = torch.autograd.grad(
    #         gathered.sum(),
    #         masked_inputs,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]
    #     grads[time_mask == 0] = 0

    #     raw_attr = grads * (inputs - baselines).unsqueeze(0)
    #     mask_avg = time_mask.sum(dim=0)
    #     final_attr = raw_attr / mask_avg

    #     return final_attr.mean(dim=0)

    # def attribute_random_time_segments_one_dim_same_for_batch(
    #     self,
    #     inputs: torch.Tensor,  # [B, T, D]
    #     baselines: torch.Tensor,  # [B, T, D]
    #     targets: torch.Tensor,  # [B]
    #     additional_forward_args,
    #     n_samples: int = 50,
    #     num_segments: int = 3,
    #     max_seg_len: int = None,
    #     min_seg_len: int = 1,
    #     noise_scale: float = 0.1,
    # ):
    #     B, T, D = inputs.shape
    #     device = inputs.device
        
    #     max_seg_len = min(max_seg_len or T, T)
        
    #     beta_dist = torch.distributions.Beta(0.3, 1)
    #     alphas = beta_dist.sample((n_samples, B)).to(device)
    #     alphas = torch.sort(alphas, dim=0).values.view(n_samples, B, 1, 1)
        
    #     expanded_inputs = inputs.unsqueeze(0) + torch.randn(n_samples, B, T, D, device=device) * noise_scale
    #     expanded_baselines = baselines.unsqueeze(0)
    #     noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
        
    #     dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
    #     t_starts = torch.randint(0, T-max_seg_len+1, (n_samples, B, num_segments), device=device)
    #     seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        
    #     time_mask = torch.ones_like(noisy_inputs)
    #     batch_indices = torch.arange(B, device=device)
    #     sample_indices = torch.arange(n_samples, device=device)
        
    #     for s in range(num_segments):
    #         indices = t_starts[:,:,s].unsqueeze(-1) + torch.arange(seg_lens[:,:,s].max(), device=device).unsqueeze(0).unsqueeze(0)
    #         valid_indices = indices < T
    #         time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0
    
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * expanded_inputs.detach()
    #     masked_inputs.requires_grad = True
        
    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
        
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)
        
    #     gathered = predictions.gather(
    #         dim=2,
    #         index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)
        
    #     grads = torch.autograd.grad(
    #         gathered.sum(),
    #         masked_inputs,
    #         create_graph=True
    #     )[0]
        
    #     grads[time_mask == 0] = 0
    #     raw_attr = grads * (expanded_inputs - expanded_baselines)
        
    #     attributions = (raw_attr[:-1] + raw_attr[1:]) / 2 * (alphas[1:] - alphas[:-1]).view(-1, 1, 1, 1)
    #     final_attr = attributions.sum(dim=0) / time_mask[:-1].sum(dim=0).clamp(min=1)
        
    #     return final_attr

    def attribute_random_time_segments_one_dim_same_for_batch(
        self,
        inputs: torch.Tensor,  # [B, T, D]
        baselines: torch.Tensor,  # [B, T, D]
        targets: torch.Tensor,  # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,  # how many time segments (one dimension each) to fix per sample
        max_seg_len: int = None,  # optional maximum length for each time segment
        min_seg_len: int = None,
    ):
        """
        Generates random contiguous time segments (each segment picks ONE random dimension).
        BUT crucially, each sample i uses the SAME random segments for the *entire batch*.

        Steps:
        1) Interpolate from baselines -> inputs using n_samples alpha steps
        2) For each sample i (i.e. alpha step), create `num_segments` random slices
            - each slice picks a single dimension, plus time range [t_start : t_end)
            - fix that dimension/time range for ALL batch items
        3) Forward pass & gather target logit => sum => compute gradients
        4) Multiply by (inputs - baselines), optionally scale by how often (t,d) was free
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        data_mask = additional_forward_args[0]

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=device).view(-1, 1, 1, 1)
        
        expanded_inputs = inputs.unsqueeze(0)
        expanded_baselines = baselines.unsqueeze(0)
        # Interpolate with batch-specific alphas
        noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise
        
        if max_seg_len is None:
            max_seg_len = T

        if min_seg_len is None:
            min_seg_len = 1

        # Generate batch-specific masks
        dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        # t_starts = torch.randint(0, T-max_seg_len+1, (n_samples, B, num_segments), device=device)
        t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

        # Initialize mask
        time_mask = torch.ones_like(noisy_inputs)

        # Create indices tensor
        batch_indices = torch.arange(B, device=device)
        sample_indices = torch.arange(n_samples, device=device)

        # Create mask via scatter
        for s in range(num_segments):
            # indices = t_starts[:,:,s].unsqueeze(-1) + torch.arange(seg_lens[:,:,s].max(), device=device).unsqueeze(0).unsqueeze(0)
            # valid_indices = indices < T
            max_len = seg_lens[:,:,s].max()
            # 2) base_range = [0, 1, 2, ..., max_len-1], shape [max_len]
            base_range = torch.arange(max_len, device=device)
            base_range = base_range.unsqueeze(0).unsqueeze(0)
            
            indices = t_starts[:,:,s].unsqueeze(-1) + base_range

            end_points = t_starts[:,:,s] + seg_lens[:,:,s]  # shape [n_samples, B]
            end_points = end_points.unsqueeze(-1)           # shape [n_samples, B, 1]

            valid_indices = (indices < end_points) & (indices < T)
            time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0

        # Combine masked inputs
        fixed_inputs = expanded_inputs.detach()
        masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs
        masked_inputs.requires_grad = True

        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)

        total_for_target = gathered.sum()
        
        grad = torch.autograd.grad(outputs=total_for_target, inputs=masked_inputs, retain_graph=True)[0]
        grad[time_mask == 0] = 0

        grads = grad.sum(dim=0)  # Proper Riemann sum
        final_attr = grads * (inputs - baselines) / time_mask.sum(dim=0)
            
        return final_attr

    # def attribute_random_time_segments_one_dim_same_for_batch_v2(
    #     self,
    #     inputs: torch.Tensor,  # [B, T, D]
    #     baselines: torch.Tensor,  # [B, T, D]
    #     targets: torch.Tensor,  # [B]
    #     additional_forward_args,
    #     n_samples: int = 50,
    #     num_segments: int = 3,  # how many time segments (one dimension each) to fix per sample
    #     max_seg_len: int = None,  # optional maximum length for each time segment
    #     min_seg_len: int = None,
    # ):
    #     """
    #     Generates random contiguous time segments (each segment picks ONE random dimension).
    #     BUT crucially, each sample i uses the SAME random segments for the *entire batch*.

    #     Steps:
    #     1) Interpolate from baselines -> inputs using n_samples alpha steps
    #     2) For each sample i (i.e. alpha step), create `num_segments` random slices
    #         - each slice picks a single dimension, plus time range [t_start : t_end)
    #         - fix that dimension/time range for ALL batch items
    #     3) Forward pass & gather target logit => sum => compute gradients
    #     4) Multiply by (inputs - baselines), optionally scale by how often (t,d) was free
    #     """
    #     '''
    #     # # if inputs.shape != baselines.shape:
    #     # #     raise ValueError("Inputs and baselines must have the same shape.")

    #     # # B, T, D = inputs.shape
    #     # # device = inputs.device

    #     # # data_mask = additional_forward_args[0]

    #     # # # -------------------------------------------------------
    #     # # # 1) Build interpolation from baseline -> inputs
    #     # # # -------------------------------------------------------
    #     # # alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
        
    #     # # expanded_inputs = inputs.unsqueeze(0)
    #     # # expanded_baselines = baselines.unsqueeze(0)
    #     # # # Interpolate with batch-specific alphas
    #     # # noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
    #     # # noise = torch.randn_like(noisy_inputs) * 1e-4
    #     # # noisy_inputs = noisy_inputs + noise
        
    #     # # if max_seg_len is None:
    #     # #     max_seg_len = T

    #     # # if min_seg_len is None:
    #     # #     min_seg_len = 1

    #     # # # Generate batch-specific masks
    #     # # dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
    #     # # seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
    #     # # # t_starts = torch.randint(0, T-max_seg_len+1, (n_samples, B, num_segments), device=device)
    #     # # t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

    #     # # # Initialize mask
    #     # # time_mask = torch.ones_like(noisy_inputs)

    #     # # # Create indices tensor
    #     # # batch_indices = torch.arange(B, device=device)
    #     # # sample_indices = torch.arange(n_samples, device=device)

    #     # # # Create mask via scatter
    #     # # for s in range(num_segments):
    #     # #     indices = t_starts[:,:,s].unsqueeze(-1) + torch.arange(seg_lens[:,:,s].max(), device=device).unsqueeze(0).unsqueeze(0)
    #     # #     valid_indices = indices < T
    #     # #     time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0

    #     # if inputs.shape != baselines.shape:
    #     #     raise ValueError("Inputs and baselines must have the same shape.")
    #     # B, T, D = inputs.shape
    #     # device = inputs.device
    #     # # -------------------------------------------------------
    #     # # 1) Build interpolation from baseline -> inputs
    #     # # -------------------------------------------------------
    #     # alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
    #     # start_pos = baselines
    #     # expanded_inputs = inputs.unsqueeze(0)    # shape [1, B, T, D]
    #     # expanded_start  = start_pos.unsqueeze(0) # shape [1, B, T, D]
    #     # # Interpolate
    #     # noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)
    #     # noise = torch.randn_like(noisy_inputs) * 1e-4
    #     # noisy_inputs = noisy_inputs + noise
    #     # # # 1. Create a Beta distribution
    #     # # beta_dist = Beta(0.5, 0.5)
    #     # # # 2. Sample n_samples alphas from Beta distribution and move to device
    #     # # #    alpha_i ~ Beta(alpha_param, beta_param)
    #     # # alphas = beta_dist.sample((n_samples,)).to(device)
    #     # # # 3. Sort the alphas in ascending order (optional but often desirable)
    #     # # alphas = torch.sort(alphas).values
    #     # # # 4. Reshape alphas to be broadcastable: [n_samples, 1, 1, 1]
    #     # # #    so they can multiply [1, B, T, D] properly
    #     # # alphas = alphas.view(-1, 1, 1, 1)
    #     # # 5. Expand your inputs and baselines to [1, B, T, D]
    #     # start_pos = baselines
    #     # expanded_inputs = inputs.unsqueeze(0)     # [1, B, T, D]
    #     # expanded_start  = start_pos.unsqueeze(0)  # [1, B, T, D]
    #     # # 6. Interpolate between start_pos and inputs using alpha
    #     # #    result shape: [n_samples, B, T, D]
    #     # noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)
    #     # # 7. Optionally add a small random noise for numerical stability (or for exploration)
    #     # noise = torch.randn_like(noisy_inputs) * 1e-4
    #     # noisy_inputs = noisy_inputs + noise
    #     # # -------------------------------------------------------
    #     # # 2) Create a random mask for each sample i,
    #     # #    apply it to all B in the batch
    #     # # -------------------------------------------------------
    #     # # time_mask: [n_samples, B, T, D],  1 => use interpolation, 0 => fix
    #     # time_mask = torch.ones_like(noisy_inputs)  # [n_samples, B, T, D]
    #     # if max_seg_len is None:
    #     #     max_seg_len = T
    #     # if min_seg_len is None:
    #     #     min_seg_len = 1
    #     # # For each sample i, we pick num_segments random (time-range, dimension).
    #     # # We do NOT loop over b in [0..B-1]. Instead, we just set the same mask for all b.
    #     # for b in range(B):
    #     #     for i in range(n_samples):
    #     #     # for i in range(1, n_samples-1):
    #     #         for seg_id in range(num_segments):
    #     #             dim_chosen = random.randint(0, D - 1)
    #     #         # for dim_chosen in range(31):
    #     #             seg_len = random.randint(min_seg_len, max_seg_len)
    #     #             t_start = random.randint(0, T - seg_len)
    #     #             t_end   = t_start + seg_len
                    
    #     #             time_mask[i, b, t_start:t_end, dim_chosen] = 0

    #     # # Combine masked inputs
    #     # fixed_inputs = expanded_inputs.detach()
    #     # masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs
    #     # masked_inputs.requires_grad = True
    #     '''
        
    #     if inputs.shape != baselines.shape:
    #         raise ValueError("Inputs and baselines must have the same shape.")

    #     B, T, D = inputs.shape
    #     device = inputs.device

    #     data_mask = additional_forward_args[0]

    #     # -------------------------------------------------------
    #     # 1) Build interpolation from baseline -> inputs
    #     # -------------------------------------------------------
    #     alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
        
    #     expanded_inputs = inputs.unsqueeze(0)
    #     expanded_baselines = baselines.unsqueeze(0)
    #     # Interpolate with batch-specific alphas
    #     noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
    #     noise = torch.randn_like(noisy_inputs) * 1e-4
    #     noisy_inputs = noisy_inputs + noise
        
    #     if max_seg_len is None:
    #         max_seg_len = T

    #     if min_seg_len is None:
    #         min_seg_len = 1

    #     # Generate batch-specific masks
    #     dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
    #     seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
    #     # t_starts = torch.randint(0, T-max_seg_len+1, (n_samples, B, num_segments), device=device)
    #     t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

    #     # Initialize mask
    #     time_mask = torch.ones_like(noisy_inputs)

    #     # Create indices tensor
    #     batch_indices = torch.arange(B, device=device)
    #     sample_indices = torch.arange(n_samples, device=device)

    #     # Create mask via scatter
    #     for s in range(num_segments):
    #         # indices = t_starts[:,:,s].unsqueeze(-1) + torch.arange(seg_lens[:,:,s].max(), device=device).unsqueeze(0).unsqueeze(0)
    #         # valid_indices = indices < T
    #         max_len = seg_lens[:,:,s].max()
    #         # 2) base_range = [0, 1, 2, ..., max_len-1], shape [max_len]
    #         base_range = torch.arange(max_len, device=device)
    #         base_range = base_range.unsqueeze(0).unsqueeze(0)
            
    #         indices = t_starts[:,:,s].unsqueeze(-1) + base_range

    #         end_points = t_starts[:,:,s] + seg_lens[:,:,s]  # shape [n_samples, B]
    #         end_points = end_points.unsqueeze(-1)           # shape [n_samples, B, 1]

    #         valid_indices = (indices < end_points) & (indices < T)
    #         time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0

    #     # Combine masked inputs
    #     fixed_inputs = expanded_inputs.detach()
    #     masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs
    #     masked_inputs.requires_grad = True

    #     # -------------------------------------------------------
    #     # 3) Forward pass & gather target logits
    #     # -------------------------------------------------------
    #     predictions = self.model(
    #         masked_inputs.view(-1, T, D),
    #         mask=None,
    #         timesteps=None,
    #         return_all=additional_forward_args[2],
    #     )
    #     # Ensure shape => [n_samples, B, num_classes]
    #     if predictions.dim() == 1:
    #         predictions = predictions.unsqueeze(-1)
    #     predictions = predictions.view(n_samples, B, -1)

    #     # Gather only the target logit for each example
    #     gathered = predictions.gather(
    #         dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
    #     ).squeeze(-1)

    #     total_for_target = gathered.sum()

    #     # -------------------------------------------------------
    #     # 4) Compute gradients
    #     # -------------------------------------------------------
    #     grad = torch.autograd.grad(
    #         outputs=total_for_target,
    #         inputs=masked_inputs,
    #         # inputs=noisy_inputs,
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[
    #         0
    #     ]  # shape => [n_samples, B, T, D]

    #     # # Average over n_samples dimension
    #     # grad[time_mask == 0] = 0
    #     # grads = grad.sum(dim=0)  # shape => [B, T, D]

    #     # # -------------------------------------------------------
    #     # # 5) Compute final attributions
    #     # # -------------------------------------------------------
    #     # # Standard gradient * (inputs - baselines)
    #     # raw_attr = grads * (inputs - baselines)

    #     # # If you want to reduce attributions where positions
    #     # # got masked out frequently, multiply by mask_avg:
    #     # mask_avg = time_mask.sum(dim=0)  # shape => [B, T, D]

    #     # final_attr = raw_attr / mask_avg
        
    #     grad = torch.autograd.grad(outputs=total_for_target, inputs=masked_inputs, retain_graph=True)[0]
    #     grad[time_mask == 0] = 0
    #     # Integration following IG formula
    #     alpha_diffs = alphas[1:] - alphas[:-1]
    #     grads = (grad[:-1] * alpha_diffs).sum(dim=0)  # Proper Riemann sum
    #     final_attr = grads * (inputs - baselines)

    #     return final_attr

    def naive_attribute(
        self, inputs, baselines, targets, additional_forward_args, n_samples=50
    ):
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")
        # mask = additional_forward_args[0]
        # if mask is None:
        #     mask = torch.ones_like(inputs)

        # 1. Create a Beta distribution
        # beta_dist = Beta(0.5, 0.5)

        # # 2. Sample n_samples alphas from Beta distribution and move to device
        # #    alpha_i ~ Beta(alpha_param, beta_param)
        # alphas = beta_dist.sample((n_samples,)).to(inputs.device)

        # # 3. Sort the alphas in ascending order (optional but often desirable)
        # alphas = torch.sort(alphas).values

        # # 4. Reshape alphas to be broadcastable: [n_samples, 1, 1, 1]
        # #    so they can multiply [1, B, T, D] properly
        # alphas = alphas.view(-1, 1, 1, 1)
        # Add noise to interpolate between baseline and input
        alphas = torch.linspace(0, 1, n_samples).view(-1, 1, 1, 1).to(inputs.device)
        start_pos = baselines
        # noisy_inputs = baselines.unsqueeze(0) + alphas * (inputs.unsqueeze(0) - baselines.unsqueeze(0))
        noisy_inputs = start_pos.unsqueeze(0) + alphas * (
            inputs.unsqueeze(0) - start_pos.unsqueeze(0)
        )

        # Add noise
        noise = torch.randn_like(noisy_inputs) * 0.0001
        noisy_inputs = noisy_inputs + noise

        # Compute gradients for each noisy input
        noisy_inputs.requires_grad = True

        # ### only last time point prediction
        # mask = mask.unsqueeze(0).repeat(n_samples, 1, 1, 1)
        # mask = mask.view(-1, inputs.shape[1], inputs.shape[2])

        predictions = self.model(
            noisy_inputs.view(-1, inputs.shape[1], inputs.shape[2]),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )
        # print(predictions.shape)

        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(1)

        predictions = predictions.view(n_samples, inputs.shape[0], -1)
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0)
            .unsqueeze(-1)
            .expand(n_samples, inputs.shape[0], 1),
        ).squeeze(-1)

        # print(predictions)
        # print(gathered)
        # print(targets)
        total_for_target = gathered.sum()
        # print(total_for_target.shape)

        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=noisy_inputs,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grads = grad.mean(dim=0)

        # Multiply by (inputs - baseline) to get final attributions
        attributions = grads * (inputs - baselines)
        # print(attributions.shape)
        # raise RuntimeError

        return attributions
