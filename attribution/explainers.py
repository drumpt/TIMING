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

import torch
import random

from utils.tensor_manipulation import normalize as normal

# Perturbation methods:


class FO:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = Occlusion(forward_func=self.f)
        baseline = torch.mean(X, dim=0, keepdim=True)  # The baseline is chosen to be the average value for each feature
        attr = explainer.attribute(X, sliding_window_shapes=(1,), baselines=baseline)
        if normalize:
            attr = normal(torch.abs(attr))  # The absolute value of the FO attribution gives the feature importance
        return attr


class FP:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = FeaturePermutation(forward_func=self.f)
        attr = explainer.attribute(X)
        if normalize:
            attr = normal(torch.abs(attr))  # The absolute value of the FP attribution gives the feature importance
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
            attr = normal(torch.abs(attr))  # The absolute value of the IG attribution gives the feature importance
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
            attr = normal(torch.abs(attr))  # The absolute value of the SVS attribution gives the feature importance
        return attr


class OUR:
    def __init__(self, model):
        self.model = model
        
    def attribute(self, inputs, baselines, targets, additional_forward_args, n_samples=50):
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
        expanded_inputs = inputs.unsqueeze(0)      # [1, B, T, D]
        expanded_start  = start_pos.unsqueeze(0)   # [1, B, T, D]

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
            return_all=additional_forward_args[2]
        )

        # Make sure predictions has shape [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        predictions = predictions.view(n_samples, inputs.shape[0], -1)

        # Gather the logit of the correct class for each sample
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, inputs.shape[0], 1)
        ).squeeze(-1)  # shape [n_samples, B]

        # Sum across all n_samples and batch for gradient
        total_for_target = gathered.sum()

        # -------------------------------------------------
        # 4) Compute gradients wrt `noisy_inputs`
        # -------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=noisy_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]  # shape: [n_samples, B, T, D]

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
    
    def attribute_time_aware(self, inputs, baselines, targets, additional_forward_args, n_samples=50):
        """
        inputs:  [B, T, D]  (batch, time, features)
        baselines: [B, T, D]  (same shape)
        targets: [B]  (integer target indices)
        """

        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        # Create alpha for interpolation (shape [n_samples, 1, 1, 1])
        alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)

        # Interpolation from start_pos to inputs + noise
        start_pos = baselines
        expanded_inputs = inputs.unsqueeze(0)     # [1, B, T, D]
        expanded_start  = start_pos.unsqueeze(0)  # [1, B, T, D]
        noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise
        
        # -------------------------------------------------
        # Time-series aware masking: fix a random time slice
        # -------------------------------------------------
        # For each sample i (0..n_samples-1) and each batch b (0..B-1),
        # pick a random segment of length seg_len and fix it to real input.

        seg_len = max(1, T // 3)  # e.g., fix ~1/3 of the time steps
        # We'll create a mask of shape [n_samples, B, T, D]
        # 1 => use the interpolation, 0 => fix to real input
        time_mask = torch.ones_like(noisy_inputs)  # start with all ones

        for i in range(n_samples):
            for b in range(B):
                t_start = torch.randint(low=0, high=T - seg_len + 1, size=(1,)).item()
                t_end = t_start + seg_len
                # set [t_start : t_end] to 0 => fix them to real input
                time_mask[i, b, t_start:t_end, :] = 0

        fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
        # Broadcast if needed
        # Combine
        masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

        masked_inputs.requires_grad = True

        # Forward pass
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2]
        )
        # shape => [n_samples*B, num_classes] or [n_samples*B] if 1D output
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)  # [n_samples*B, 1]

        predictions = predictions.view(n_samples, B, -1)
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)
        total_for_target = gathered.sum()

        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=masked_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]  # [n_samples, B, T, D]

        # Average gradient over n_samples
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

        return final_attributions
    
    def attribute_time_aware_random_segments(
        self,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        targets: torch.Tensor,
        additional_forward_args,
        n_samples: int = 50
    ):
        """
        inputs:  shape [B, T, D]
        baselines:  shape [B, T, D]
        targets:  shape [B]  (one target class per batch item)
        additional_forward_args:  used only for e.g., return_all
        n_samples:  number of alpha interpolation steps
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        # -------------------------------------------------
        # 1) Build interpolation from baseline --> inputs
        # -------------------------------------------------
        alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
        # Start from "start_pos" so alpha=0 means baselines,
        # alpha=1 means inputs (plus small noise).
        start_pos = baselines
        expanded_inputs = inputs.unsqueeze(0)    # [1, B, T, D]
        expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]
        noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

        # Add a little random noise
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise

        # -------------------------------------------------
        # 2) Time-series aware masking (random contiguous segments)
        # -------------------------------------------------
        # For each interpolation i in [0..n_samples-1] and each batch item b,
        # randomly pick:
        #  - seg_len in [1, T] (or some smaller range if desired)
        #  - a start index t_start
        # Then fix that segment [t_start : t_start+seg_len] to the real input.
        #
        # We'll define time_mask[i, b, t, d] = 1 => use interpolation,
        #                                        = 0 => fix to real input.

        time_mask = torch.ones_like(noisy_inputs)  # shape [n_samples, B, T, D]

        for i in range(n_samples):
            for b in range(B):
                seg_len = torch.randint(low=1, high=T + 1, size=(1,)).item()  
                # e.g., random segment length from 1..T
                t_start = torch.randint(low=0, high=T - seg_len + 1, size=(1,)).item()
                t_end = t_start + seg_len
                # Fix that time slice => set mask=0 for that region
                time_mask[i, b, t_start:t_end, :] = 0

        # Combine masked (fixed) portion with interpolated portion
        fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
        masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

        # Turn on gradient for only the interpolation portion
        masked_inputs.requires_grad = True

        # -------------------------------------------------
        # 3) Forward pass & gather target predictions
        # -------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2]
        )
        # Reshape to [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather the logit for the target class
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)  # shape [n_samples, B]

        # Sum over samples and batch items
        total_for_target = gathered.sum()

        # -------------------------------------------------
        # 4) Compute gradients w.r.t. the masked_inputs
        # -------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=masked_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]  # shape [n_samples, B, T, D]

        # Average gradient over n_samples
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

        return final_attributions


    def attribute_random_time_segments_one_dim(
        self,
        inputs: torch.Tensor,      # [B, T, D]
        baselines: torch.Tensor,   # [B, T, D]
        targets: torch.Tensor,     # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,     # how many time segments to fix (single dimension) per sample
        max_seg_len: int = None    # optional max length for each time segment
    ):
        """
        This version picks multiple random contiguous time segments per sample,
        but for exactly ONE random dimension in each segment.

        Steps:
        1) Build interpolation from baselines -> inputs (the standard IG approach).
        2) For each sample i, each batch item b, and each of num_segments:
            - pick a random dimension d
            - pick a random contiguous time slice [t_start : t_end)
            - fix that slice & dimension to real inputs => no attributions there
        3) Forward pass, gather target logit, compute gradients
        4) Multiply by (inputs - baselines)
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
        start_pos = baselines
        expanded_inputs = inputs.unsqueeze(0)     # [1, B, T, D]
        expanded_start  = start_pos.unsqueeze(0)  # [1, B, T, D]

        # Interpolate
        noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

        # Add small noise
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise

        # -------------------------------------------------------
        # 2) Create random time-segment mask (single dimension)
        # -------------------------------------------------------
        # We'll build a mask of shape [n_samples, B, T, D]:
        #    mask=1 => use the interpolated value
        #    mask=0 => fix to the real input
        time_mask = torch.ones_like(noisy_inputs)  # [n_samples, B, T, D]

        if max_seg_len is None:
            max_seg_len = T  # up to the entire T by default

        for i in range(n_samples):
            for b in range(B):
                for seg_id in range(num_segments):
                    # Pick a random dimension:
                    dim_chosen = random.randint(0, D - 1)
                    # Pick a random segment length & start
                    seg_len = random.randint(1, max_seg_len)
                    t_start = random.randint(0, T - seg_len)
                    t_end   = t_start + seg_len
                    # Fix time slice [t_start : t_end] for that single dimension
                    time_mask[i, b, t_start:t_end, dim_chosen] = 0

        # Combine: masked_inputs = time_mask * interpolated + (1-time_mask)* real_input
        fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
        masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs

        masked_inputs.requires_grad = True

        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2]
        )
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)  # [n_samples, B]

        total_for_target = gathered.sum()

        # -------------------------------------------------------
        # 4) Compute gradients w.r.t. masked_inputs
        # -------------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=masked_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]  # shape => [n_samples, B, T, D]

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

        return final_attr

    def attribute_random_time_segments_one_dim_same_for_batch(
        self,
        inputs: torch.Tensor,      # [B, T, D]
        baselines: torch.Tensor,   # [B, T, D]
        targets: torch.Tensor,     # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,     # how many time segments (one dimension each) to fix per sample
        max_seg_len: int = None,    # optional maximum length for each time segment
        min_seg_len: int = None
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

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
        start_pos = baselines
        expanded_inputs = inputs.unsqueeze(0)    # shape [1, B, T, D]
        expanded_start  = start_pos.unsqueeze(0) # shape [1, B, T, D]

        # Interpolate
        noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise
        
        # # 1. Create a Beta distribution
        # beta_dist = Beta(0.5, 0.5)

        # # 2. Sample n_samples alphas from Beta distribution and move to device
        # #    alpha_i ~ Beta(alpha_param, beta_param)
        # alphas = beta_dist.sample((n_samples,)).to(device)

        # # 3. Sort the alphas in ascending order (optional but often desirable)
        # alphas = torch.sort(alphas).values

        # # 4. Reshape alphas to be broadcastable: [n_samples, 1, 1, 1]
        # #    so they can multiply [1, B, T, D] properly
        # alphas = alphas.view(-1, 1, 1, 1)

        # 5. Expand your inputs and baselines to [1, B, T, D]
        start_pos = baselines
        expanded_inputs = inputs.unsqueeze(0)     # [1, B, T, D]
        expanded_start  = start_pos.unsqueeze(0)  # [1, B, T, D]

        # 6. Interpolate between start_pos and inputs using alpha
        #    result shape: [n_samples, B, T, D]
        noisy_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

        # 7. Optionally add a small random noise for numerical stability (or for exploration)
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise

        # -------------------------------------------------------
        # 2) Create a random mask for each sample i,
        #    apply it to all B in the batch
        # -------------------------------------------------------
        # time_mask: [n_samples, B, T, D],  1 => use interpolation, 0 => fix
        time_mask = torch.ones_like(noisy_inputs)  # [n_samples, B, T, D]

        if max_seg_len is None:
            max_seg_len = T
        
        if min_seg_len is None:
            min_seg_len = 1

        # For each sample i, we pick num_segments random (time-range, dimension).
        # We do NOT loop over b in [0..B-1]. Instead, we just set the same mask for all b.
        for i in range(n_samples):
        # for i in range(1, n_samples-1):
            for seg_id in range(num_segments):
                dim_chosen = random.randint(0, D - 1)
            # for dim_chosen in range(31):
                seg_len = random.randint(min_seg_len, max_seg_len)
                t_start = random.randint(0, T - seg_len)
                t_end   = t_start + seg_len

                # Fix that slice for *all batch items* => set to 0
                time_mask[i, :, t_start:t_end, dim_chosen] = 0
                
                # if t_start > 0:
                #     time_mask[i, :, t_start - 1, dim_chosen] = 0.5
                # if t_end < T :
                #     time_mask[i, :, t_end, dim_chosen] = 0.5
                
        # time_mask = 1 - time_mask

        # Combine masked inputs
        # noisy_inputs.requires_grad = True
        fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
        masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs
        masked_inputs.requires_grad = True
        # masked_inputs.zero_()
        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2]
        )
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)

        total_for_target = gathered.sum()

        # -------------------------------------------------------
        # 4) Compute gradients
        # -------------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=masked_inputs,
            # inputs=noisy_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]  # shape => [n_samples, B, T, D]

        # Average over n_samples dimension
        grad[time_mask == 0] = 0
        grads = grad.sum(dim=0)  # shape => [B, T, D]

        # -------------------------------------------------------
        # 5) Compute final attributions
        # -------------------------------------------------------
        # Standard gradient * (inputs - baselines)
        raw_attr = grads * (inputs - baselines)

        # If you want to reduce attributions where positions
        # got masked out frequently, multiply by mask_avg:
        mask_avg = time_mask.sum(dim=0)  # shape => [B, T, D]
        
        final_attr = raw_attr / mask_avg

        return final_attr


    def attribute_adaptive_time_segments(
        self,
        inputs: torch.Tensor,      # [B, T, D]
        baselines: torch.Tensor,   # [B, T, D]
        targets: torch.Tensor,     # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,
        max_seg_len: int = None
    ):
        """
        Adaptive TimeIG:
        Generates adaptive time segments based on gradient norms for each sample.

        Steps:
        1) Interpolate from baselines -> inputs using n_samples alpha steps.
        2) For each sample, compute gradient norms and select top-k segments.
        3) Mask input segments and compute attributions.

        Parameters:
            - num_segments: Number of segments to adaptively select.
            - max_seg_len: Maximum segment length.
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device
        
        if max_seg_len is None:
            max_seg_len = T

        # 1) Interpolation
        alphas = torch.linspace(0, 1, n_samples, device=device).view(-1, 1, 1, 1)
        start_pos = baselines
        expanded_inputs = inputs.unsqueeze(0)    # [1, B, T, D]
        expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]

        interpolated_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)
        interpolated_inputs.requires_grad = True

        # 2) Gradient Norm Calculation
        # Forward pass and gather target logits
        predictions = self.model(interpolated_inputs.view(-1, T, D), 
                                 mask=None,
                                 timesteps=None,
                                 return_all=additional_forward_args[2])
        predictions = predictions.view(n_samples, B, -1)  # [n_samples, B, num_classes]

        gathered = predictions.gather(dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)).squeeze(-1)
        total_for_target = gathered.sum()

        # Backpropagate to compute gradients
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=interpolated_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]  # [n_samples, B, T, D]

        grad_norms = grad.mean(dim=0).norm(dim=-1)  # [B, T]

        # 3) Adaptive Segment Selection
        masks = torch.ones_like(interpolated_inputs)  # [n_samples, B, T, D]
        for b in range(B):
            top_indices = torch.topk(grad_norms[b], num_segments).indices
            for idx in top_indices:
                t_start = max(0, idx - max_seg_len // 2)
                t_end = min(T, idx + max_seg_len // 2)
                masks[:, b, t_start:t_end, :] = 0

        # Masked Inputs
        masked_inputs = masks * interpolated_inputs + (1 - masks) * inputs.unsqueeze(0)

        # Compute gradients and attributions
        masked_predictions = self.model(masked_inputs.view(-1, T, D),
                                        mask=None,
                                        timesteps=None,
                                        return_all=additional_forward_args[2])
        masked_predictions = masked_predictions.view(n_samples, B, -1)
        masked_gathered = masked_predictions.gather(dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)).squeeze(-1)
        masked_total = masked_gathered.sum()

        masked_grad = torch.autograd.grad(
            outputs=masked_total,
            inputs=interpolated_inputs,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Attribution
        final_attr = masked_grad.mean(dim=0) * (inputs - baselines)

        return final_attr
    
    def naive_attribute(self, inputs, baselines, targets, additional_forward_args, n_samples=50):
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
        noisy_inputs = start_pos.unsqueeze(0) + alphas * (inputs.unsqueeze(0) - start_pos.unsqueeze(0))

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
            return_all=additional_forward_args[2]
        )
        # print(predictions.shape)

        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(1)

        predictions = predictions.view(n_samples, inputs.shape[0], -1)
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, inputs.shape[0], 1)
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
            allow_unused=True
        )[0]
        grads = grad.mean(dim=0)

        # Multiply by (inputs - baseline) to get final attributions
        attributions = grads * (inputs - baselines)
        # print(attributions.shape)
        # raise RuntimeError
        
        return attributions