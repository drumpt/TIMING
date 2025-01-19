from __future__ import annotations

import os
import pathlib
from datetime import datetime
import json
from typing import Dict, List, Set, Optional, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from scipy.stats import gaussian_kde
from tqdm import tqdm
import functools

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributions as dist
import matplotlib.pyplot as plt

class ShapeletExplainer(BaseExplainer):
    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: Optional[DataLoader] = None,
        feature_names: Optional[List[str]] = None,
        p: float = 0.1,
        **kwargs
    ):
        self.device = device
        self.num_features = num_features
        self.data_name = data_name
        self.path = path
        self.feature_names = feature_names
        self.p = p
        self.risk_threshold = p
        
        # Shapelet parameters
        self.min_length = kwargs.get("min_length", 5)
        self.max_length = kwargs.get("max_length", 10)
        
        # Initialize storage
        self.unique_lengths = []
        self.window_patterns = {}
        self._train_loader = train_loader
        
        if self._load_shapelets():
            return
        
        # If loading fails and we have train_loader, discover new shapelets
        if self._train_loader is not None:
            self._discover_shapelets_featurewise(self._train_loader)
            self._train_loader = None
            self._save_shapelets()

    def _get_cache_path(self):
        """Get path for shapelet cache."""
        cache_dir = self.path / "shapelet_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{self.data_name}_multivariate_shapelets.pt"
    
    def _save_shapelets(self):
        """Save discovered shapelets."""
        save_data = {
            'unique_lengths': self.unique_lengths,
            'window_patterns': {
                k: v.cpu() for k, v in self.window_patterns.items()
            }
        }
        torch.save(save_data, self._get_cache_path())
        print(f"Saved shapelets to {self._get_cache_path()}")

    def _load_shapelets(self):
        """Load existing shapelets if available."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        try:
            data = torch.load(cache_path, map_location=self.device)
            self.unique_lengths = data['unique_lengths']
            self.window_patterns = {
                k: v.to(self.device) for k, v in data['window_patterns'].items()
            }
            print(f"Loaded shapelets from {cache_path}")
            return True
        except Exception as e:
            print(f"Error loading shapelets: {e}")
            return False

    def _discover_shapelets_featurewise(self, train_loader: DataLoader):
        """Discover feature-wise motifs using a sliding window approach."""
        print("Discovering shapelets feature-wise...")

        # Collect data
        all_data = []
        with torch.no_grad():
            for batch in train_loader:
                x, _, _ = batch  # (B, N, T)
                x = x.to(self.device)
                all_data.append(x.cpu())

        train_data = torch.cat(all_data, dim=0)  # (B, N, T)

        print(f"Train data shape: {train_data.shape}")  # Debugging

        # Extract shapelets for each feature
        n_samples, n_features, n_timestamps = train_data.shape
        for feature_idx in range(n_features):
            feature_data = train_data[:, feature_idx, :]  # (B, T)

            print(f"Processing feature {feature_idx} with data shape: {feature_data.shape}")  # Debugging

            # Sliding window approach
            for length in range(self.min_length, self.max_length + 1):
                windows = []
                for start in range(0, n_timestamps - length + 1):
                    window = feature_data[:, start:start + length]  # (B, length)
                    windows.append(window)

                if not windows:
                    print(f"No windows extracted for feature {feature_idx} and length {length}")  # Debugging
                    continue

                print(f"Extracted {len(windows)} windows of length {length} for feature {feature_idx}")  # Debugging

                # Aggregate windows and calculate mean as representative motif
                windows_tensor = torch.stack(windows, dim=0)  # (num_windows, B, length)
                mean_window = windows_tensor.mean(dim=0)  # (B, length)

                key = f'shapelet_{length}_feature_{feature_idx}'
                if key not in self.window_patterns:
                    self.window_patterns[key] = []
                self.window_patterns[key].append(mean_window)

                # Update unique_lengths
                if length not in self.unique_lengths:
                    self.unique_lengths.append(length)

        # Consolidate shapelets
        for key in self.window_patterns:
            self.window_patterns[key] = torch.stack(self.window_patterns[key])

        if not self.unique_lengths:
            print("Warning: No shapelets discovered. Ensure data has sufficient variability.")
        else:
            self.unique_lengths.sort()

        print("Feature-wise shapelet discovery completed.")


    def attribute(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Attribution method using only the longest window size from pre-computed training motifs.
        """
        x, mask = x.contiguous(), mask.contiguous()
        device = x.device

        # Check if unique_lengths is empty
        if not self.unique_lengths:
            print("No shapelets discovered. Returning zero attribution.")
            return np.zeros_like(x.cpu().numpy())

        # Use the longest window size
        max_window_size = max(self.unique_lengths)

        # Skip if window size is too large, return zero attribution
        if max_window_size > x.shape[2]:
            return np.zeros_like(x.cpu().numpy())

        # Get predictions for pattern selection
        with torch.no_grad():
            predictions = self.base_model.predict(x, mask=mask, return_all=False)
            pos_mask = predictions >= self.risk_threshold
            neg_mask = ~pos_mask

        # Initialize attribution storage
        attribution = torch.zeros_like(x)
        attribution_counts = torch.zeros_like(x)

        # Create windows for the chosen size
        windows = x.unfold(-1, max_window_size, max_window_size)
        n_windows = windows.shape[2]

        # Process each window position
        for window_idx in range(n_windows):
            start_pos = window_idx * max_window_size
            end_pos = start_pos + max_window_size

            # Get current windows
            curr_windows = windows[:, :, window_idx]  # [batch, features, window_size]

            # Add input as final waypoint
            path_points = [curr_windows]

            # Compute all interpolation points at once
            n_steps = 50

            # Interpolate between zero and the input
            interpolated = torch.linspace(0, 1, n_steps).view(-1, 1, 1, 1).to(device) * curr_windows

            # Compute gradients
            interpolated.requires_grad = True
            predictions = self.base_model.predict(interpolated, mask=mask, return_all=False)
            grads = torch.autograd.grad(predictions.sum(), interpolated, retain_graph=True)[0]

            attribution[:, :, start_pos:end_pos] = grads.mean(dim=0)
            attribution_counts[:, :, start_pos:end_pos] += 1

        # Average attributions
        attribution_counts = torch.clamp(attribution_counts, min=1)
        attribution = attribution / attribution_counts

        return attribution.detach().cpu().numpy()

    def get_name(self):
        return "shapelet_explainer"


class MotifGuidedIG(ShapeletExplainer):
    def __init__(
        self,
        device,
        model,
        num_features,
        data_name,
        path,
        train_loader,
        p=0.1,
        n_motifs=3,
        n_steps=50,
        alpha_param=0.3,
        **kwargs,
    ):
        self.n_motifs = n_motifs
        self.n_steps = n_steps
        self.alpha_param = alpha_param
        self.min_window_size = 6
        
        self.base_model = model


    def _find_nearest_motifs(self, patterns, curr_windows):
        """Find nearest motifs from training set for each current window."""
        nearest_motifs = []
        for feature_idx in range(curr_windows.shape[1]):
            patterns_feature = patterns[:, feature_idx, :]  # (num_patterns, w_size)
            curr_windows_feature = curr_windows[:, feature_idx, :]  # (batch_size, w_size)

            # Calculate squared distances
            distances = (
                (patterns_feature.unsqueeze(1) - curr_windows_feature.unsqueeze(0)) ** 2
            ).sum(dim=-1)  # (num_patterns, batch_size)

            # Find nearest motifs
            nearest_indices = torch.argmin(distances, dim=0)  # (batch_size,)
            nearest_motifs_feature = patterns_feature[nearest_indices]
            nearest_motifs.append(nearest_motifs_feature)

        # Stack motifs for all features
        return torch.stack(nearest_motifs, dim=1)

    def _get_waypoints(self, curr_windows, pos_mask, neg_mask, window_size):
        """Generate waypoints for integration path using beta-distributed interpolation."""
        batch_size, n_features, w_size = curr_windows.shape
        device = curr_windows.device

        # Initialize with zero baseline as the first waypoint
        waypoints = [torch.zeros_like(curr_windows)]

        # Define the beta distribution for alpha sampling
        beta_dist = torch.distributions.Beta(self.alpha_param, self.alpha_param)

        # Process positive samples
        if pos_mask.any():
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            pattern_key = f"low_risk_{window_size}"
            if pattern_key in self.window_patterns:
                patterns = self.window_patterns[pattern_key]

                for _ in range(self.n_motifs):
                    # Sample alpha from beta distribution
                    alpha = beta_dist.sample((1,)).item()  # Single alpha value
                    print(f"{alpha=}")

                    # Find nearest motifs for the current interpolated windows
                    interpolated = (
                        alpha * torch.zeros_like(curr_windows[pos_indices])
                        + (1 - alpha) * curr_windows[pos_indices]
                    )
                    nearest_motifs = self._find_nearest_motifs(patterns, interpolated)

                    # Update interpolated with motifs influence
                    interpolated = (
                        alpha * nearest_motifs + (1 - alpha) * curr_windows[pos_indices]
                    )
                    waypoints.append(interpolated)

        # Process negative samples
        if neg_mask.any():
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            pattern_key = f"high_risk_{window_size}"
            if pattern_key in self.window_patterns:
                patterns = self.window_patterns[pattern_key]

                for _ in range(self.n_motifs):
                    # Sample alpha from beta distribution
                    alpha = beta_dist.sample((1,)).item()  # Single alpha value

                    # Find nearest motifs for the current interpolated windows
                    interpolated = (
                        alpha * torch.zeros_like(curr_windows[neg_indices])
                        + (1 - alpha) * curr_windows[neg_indices]
                    )
                    nearest_motifs = self._find_nearest_motifs(patterns, interpolated)

                    # Update interpolated with motifs influence
                    interpolated = (
                        alpha * nearest_motifs + (1 - alpha) * curr_windows[neg_indices]
                    )
                    waypoints.append(interpolated)

        # Add the original windows as the final waypoint
        waypoints.append(curr_windows)

        # Stack all waypoints into a single tensor
        return torch.stack(waypoints)


    def _generate_interpolations(self, start, end, n_points):
        """Generate interpolations between start and end."""
        return torch.lerp(start.unsqueeze(0), end.unsqueeze(0), torch.linspace(0, 1, n_points).to(start.device).view(-1, 1, 1, 1))

    def _integrated_gradients_path(self, x, mask, waypoints, start_pos, end_pos, n_points):
        device = x.device
        total_ig = torch.zeros_like(x).to(device)

        for i in range(len(waypoints) - 1):
            start_point, end_point = waypoints[i], waypoints[i + 1]
            interpolations = self._generate_interpolations(start_point, end_point, n_points)

            full_inputs = x.clone().unsqueeze(0).repeat(n_points, 1, 1, 1)
            full_inputs[:, :, :, start_pos:end_pos] = interpolations

            full_inputs.requires_grad_(True)
            predictions = self.base_model.predict(full_inputs.view(-1, *x.shape[1:]), mask=mask.repeat(n_points, 1, 1), return_all=False)
            grads = torch.autograd.grad(predictions.sum(), full_inputs, retain_graph=True)[0]
            diffs = (interpolations[1:] - interpolations[:-1]) / n_points
            total_ig[:, :, start_pos:end_pos] += (grads[:-1][:, :, :, start_pos:end_pos] * diffs).sum(dim=0).detach()

        return total_ig[:, :, start_pos:end_pos]

    def attribute(self, x, mask=None):
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x, mask = x.contiguous(), mask.contiguous()
        predictions = self.base_model.predict(x, mask=mask, return_all=False)
        pos_mask = predictions >= self.p
        neg_mask = ~pos_mask

        final_attribution = torch.zeros_like(x).cpu()
        attribution_counts = torch.zeros_like(x).cpu()

        window_size = self.min_window_size
        windows = x.unfold(-1, window_size, window_size)
        if windows.shape[2] > 0:
            for window_idx in range(windows.shape[2]):
                start_pos = window_idx * window_size
                end_pos = start_pos + window_size
                curr_windows = windows[:, :, window_idx]

                waypoints = self._get_waypoints(curr_windows, pos_mask, neg_mask, window_size)
                window_attr = self._integrated_gradients_path(x, mask, waypoints, start_pos, end_pos, self.n_steps).cpu()
                final_attribution[:, :, start_pos:end_pos] += window_attr
                attribution_counts[:, :, start_pos:end_pos] += 1

        # attribution_counts = torch.clamp(attribution_counts, min=1)
        final_attribution = final_attribution / attribution_counts

        if self.p == -1.0:
            final_attribution = torch.abs(final_attribution)
        else:
            mask = (predictions < self.p).view(-1, 1, 1)
            final_attribution[mask.expand_as(final_attribution).cpu()] *= -1

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return final_attribution.detach().cpu().numpy()

    def get_name(self):
        return "motif_guided_ig"