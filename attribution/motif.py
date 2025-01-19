from __future__ import annotations

import os
import pathlib
import json
from typing import Dict, List, Set, Optional
import pathlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from captum._utils.common import _run_forward
# from winit.explainer.explainers import BaseExplainer


class IrregularShapeletAnalyzer:
    def __init__(
        self,
        output_dir: str = "shapelet_results",
        shapelet_lengths: dict = {5: 5, 8: 5, 10: 5},
        max_iter: int = 2000,
        batch_size: int = 16,
        weight_regularizer: float = 0.001,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save configuration for later use
        self.config = {
            "shapelet_lengths": shapelet_lengths,
            "max_iter": max_iter,
            "batch_size": batch_size,
            "weight_regularizer": weight_regularizer,
        }

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def create_binary_mask(self, timestamps, features, time_grid):
        n_samples = len(timestamps)
        n_timesteps = len(time_grid)
        n_features = features[0].shape[1] if len(features) > 0 else 0

        mask = np.zeros((n_samples, n_timesteps, n_features))
        for i, (ts, feat) in enumerate(zip(timestamps, features)):
            for j in range(n_features):
                for t in ts:
                    idx = np.abs(time_grid - t).argmin()
                    mask[i, idx, j] = 1
        return mask

    def prepare_data(
        self,
        timestamps,
        features,
        labels,
        time_grid=None,
        test_size=0.2,
        random_state=42,
    ):
        if time_grid is None:
            all_times = np.concatenate(timestamps)
            time_grid = np.linspace(all_times.min(), all_times.max(), num=1000)

        mask = self.create_binary_mask(timestamps, features, time_grid)

        # Save data stats
        stats = {
            "n_samples": len(timestamps),
            "n_features": features[0].shape[1],
            "n_timesteps": len(time_grid),
            "n_classes": len(np.unique(labels)),
        }

        with open(os.path.join(self.run_dir, "data_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

        return mask, time_grid


class MotifDiscovery:
    def __init__(self, shapelet_analyzer: IrregularShapeletAnalyzer):
        self.shapelet_analyzer = shapelet_analyzer

    def discover_motifs(
        self,
        timestamps: List[np.ndarray],
        features: List[np.ndarray],
        labels: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
    ) -> Dict:
        mask, time_grid = self.shapelet_analyzer.prepare_data(
            timestamps, features, labels, time_grid
        )

        # Save results
        results = {
            "run_dir": self.shapelet_analyzer.run_dir,
            "n_samples": len(timestamps),
            "n_features": features[0].shape[1],
            "time_points": len(time_grid),
        }

        return results


class MotifVisualizer:
    def __init__(self, save_dir: pathlib.Path):
        """
        Initialize MotifVisualizer

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_motifs(self, motifs: Dict, feature_names: Optional[List[str]] = None):
        """Save motifs with visualizations and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / timestamp
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "n_motifs": len(motifs),
        }

        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Save visualizations for each motif
        for i, motif_data in enumerate(motifs):
            motif_dir = save_path / f"motif_{i}"
            motif_dir.mkdir(exist_ok=True)

            # Create visualization
            plt.figure(figsize=(10, 6))
            self._plot_motif(motif_data, feature_names)
            plt.savefig(motif_dir / "visualization.png")
            plt.close()

            # Save motif data
            with open(motif_dir / "data.json", "w") as f:
                json.dump(self._motif_to_dict(motif_data), f, indent=4)

    def _plot_motif(self, motif_data: Dict, feature_names: Optional[List[str]] = None):
        """Create visualization for a motif"""
        if "pattern" in motif_data:
            plt.subplot(2, 1, 1)
            pattern = motif_data["pattern"]
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.numpy()
            plt.plot(pattern)
            plt.title("Motif Pattern")
            plt.xlabel("Time")
            plt.ylabel("Value")

        if "matches" in motif_data:
            plt.subplot(2, 1, 2)
            plt.hist([m["start"] for m in motif_data["matches"]], bins=20)
            plt.title("Motif Locations")
            plt.xlabel("Time Index")
            plt.ylabel("Count")

        plt.tight_layout()

    def _motif_to_dict(self, motif_data: Dict) -> Dict:
        """Convert motif data to JSON-serializable format"""
        serializable = {}
        for k, v in motif_data.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                serializable[k] = v.tolist()
            elif isinstance(v, list):
                serializable[k] = v
            else:
                serializable[k] = (
                    float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                )
        return serializable

# class ShapeletExplainer:
#     def __init__(
#         self,
#         model,
#         base_explainer,
#         device,
#         num_features: int,
#         data_name: str,
#         path: pathlib.Path,
#         train_loader: Optional[DataLoader] = None,
#         feature_names: Optional[List[str]] = None,
#         p: float = 0.1,
#         risk_threshold: float = 0.5,
#         **kwargs,
#     ):
#         self.device = device
#         self.num_features = num_features
#         self.data_name = data_name
#         self.path = path
#         self.feature_names = feature_names
#         self.p = p
#         self.risk_threshold = risk_threshold
        
#         # Enhanced shapelet parameters
#         self.min_length = kwargs.get("min_length", 5)
#         self.max_length = kwargs.get("max_length", 10)
#         self.n_shapelets = kwargs.get("n_shapelets", 5)
#         self.distance_threshold = kwargs.get("distance_threshold", 0.1)
#         self.batch_size = kwargs.get("batch_size", 32)
        
#         # Initialize shapelets with precomputed indices
#         self.high_risk_shapelets = []
#         self.low_risk_shapelets = []
#         self.pattern_indices = {}  # Cache for pattern indices
#         self._train_loader = train_loader
        
#         self.base_model = model
#         if base_explainer == "ig":
#             from captum.attr import IntegratedGradients
#             self.explainer = IntegratedGradients(self.base_model)
            
#     # @torch.cuda.amp.autocast()
#     def _create_motif_aware_baseline(
#             self, x: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor
#         ) -> torch.Tensor:
#             """Create baseline by removing temporal motifs and applying carry-forward."""
#             batch_size, n_features, seq_length = x.shape  # B x N x T format
#             baseline = x.clone()
#             device = x.device
            
#             # Step 1: Remove motifs
#             high_risk = pred >= self.risk_threshold
            
#             # Process in batches
#             for i in range(batch_size):
#                 # Select shapelets based on risk
#                 shapelets = self.high_risk_shapelets if high_risk[i] else self.low_risk_shapelets
#                 if not shapelets:
#                     continue
                    
#                 # Group patterns by length
#                 patterns_by_length = {}
#                 for shapelet in shapelets:
#                     length = shapelet["length"]
#                     if length not in patterns_by_length:
#                         patterns_by_length[length] = []
#                     patterns_by_length[length].append(
#                         shapelet["pattern"].to(device)
#                     )
                
#                 # Process each length group separately
#                 for length, patterns in patterns_by_length.items():
#                     # Stack patterns of the same length
#                     pattern_tensor = torch.stack(patterns)
                    
#                     # Process each feature
#                     for feat in range(n_features):
#                         if length >= seq_length:
#                             continue
                            
#                         # Create windows for the current sequence
#                         windows = baseline[i, feat].unfold(0, length, 1)
#                         mask_windows = mask[i, feat].unfold(0, length, 1)
                        
#                         # Calculate distances to all patterns at once
#                         distances = torch.cdist(
#                             windows.view(-1, length),
#                             pattern_tensor
#                         )
                        
#                         # Check quality and mask conditions
#                         quality = torch.exp(-distances)
#                         valid_mask = mask_windows.sum(dim=1) >= length * 0.5
#                         matches = (quality > 0.3).any(dim=1) & valid_mask
                        
#                         # Set matching regions to NaN
#                         for window_idx in matches.nonzero().squeeze(-1):
#                             baseline[i, feat, window_idx:window_idx + length] = float('nan')
            
#             # Step 2: Apply carry-forward to fill NaN values
#             nan_mask = torch.isnan(baseline)
#             if nan_mask.any():
#                 # Initialize with zeros
#                 filled = torch.zeros_like(baseline)
                
#                 # For each batch and feature
#                 for i in range(batch_size):
#                     for f in range(n_features):
#                         valid_mask = ~nan_mask[i, f]
                        
#                         # If no valid values exist, use zeros
#                         if not valid_mask.any():
#                             filled[i, f] = torch.zeros(seq_length, device=device)
#                             continue
                            
#                         # Get valid values and their indices
#                         valid_vals = baseline[i, f][valid_mask]
#                         valid_indices = valid_mask.nonzero().squeeze()
                        
#                         if valid_indices.dim() == 0:  # Single valid index
#                             valid_indices = valid_indices.unsqueeze(0)
                        
#                         # Handle forward fill
#                         current_val = 0.0
#                         for t in range(seq_length):
#                             if valid_mask[t]:
#                                 current_val = baseline[i, f, t]
#                             filled[i, f, t] = current_val
                            
#                 baseline = filled
                
#             return baseline

#     def attribute(self, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
#         """Optimized attribution with mixed precision and efficient batching."""
#         with torch.no_grad():
#             pred = self.base_model.predict(x, mask=mask, return_all=False)
        
#         # Get baseline
#         baseline = self._create_motif_aware_baseline(x, mask, pred)
        
#         # Process in optimized chunks
#         batch_size = x.shape[0]
#         n_gpus = torch.cuda.device_count()
#         chunk_size = self.batch_size * max(1, n_gpus)
#         results = []
        
#         for start_idx in range(0, batch_size, chunk_size):
#             end_idx = min(start_idx + chunk_size, batch_size)
            
#             # Process chunk
#             chunk_results = self._process_chunk(
#                 x[start_idx:end_idx],
#                 mask[start_idx:end_idx],
#                 baseline[start_idx:end_idx],
#                 pred[start_idx:end_idx]
#             )
#             results.append(chunk_results)
        
#         return np.concatenate(results, axis=0)

#     def _process_chunk(self, x_chunk, mask_chunk, baseline_chunk, pred_chunk):
#         """Process a data chunk efficiently."""
#         n_gpus = torch.cuda.device_count()
#         if n_gpus > 0:
#             # Split across available GPUs
#             split_size = len(x_chunk) // n_gpus
#             futures = []
            
#             with ThreadPoolExecutor(max_workers=n_gpus) as executor:
#                 for i in range(n_gpus):
#                     start = i * split_size
#                     end = start + split_size if i < n_gpus - 1 else len(x_chunk)
                    
#                     futures.append(executor.submit(
#                         self._process_gpu_chunk,
#                         x_chunk[start:end],
#                         mask_chunk[start:end],
#                         baseline_chunk[start:end],
#                         pred_chunk[start:end],
#                         f'cuda:{i}'
#                     ))
                
#                 results = [f.result() for f in futures]
#             return np.concatenate(results, axis=0)
#         else:
#             return self._process_gpu_chunk(x_chunk, mask_chunk, baseline_chunk, pred_chunk, 'cpu')

#     # @torch.cuda.amp.autocast()
#     def _process_gpu_chunk(self, x, mask, baseline, pred, device):
#         """Process a chunk on a specific GPU."""
#         x = x.to(device)
#         mask = mask.to(device)
#         baseline = baseline.to(device)
        
#         # with torch.cuda.amp.autocast():
#         return self.explainer.attribute(x, mask, baseline=baseline)

#     def _find_centroids(self, subsequences: List[Dict], length: int) -> List[Dict]:
#         """Optimized centroid finding with GPU acceleration."""
#         if not subsequences:
#             return []
            
#         # Move computation to GPU if available
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         sequences = torch.stack([s["sequence"] for s in subsequences]).to(device)
        
#         # Use k-means++ initialization for better centroids
#         centroids = []
#         remaining_mask = torch.ones(len(subsequences), dtype=bool, device=device)
        
#         while len(centroids) < self.n_shapelets and remaining_mask.any():
#             if len(centroids) == 0:
#                 # Initialize first centroid using k-means++ initialization
#                 weights = torch.ones(sequences.shape[0], device=device)
#                 centroid_idx = torch.multinomial(weights, 1)[0]
#             else:
#                 # Choose next centroid using k-means++ initialization
#                 distances = torch.cdist(sequences, torch.stack([c["pattern"] for c in centroids]))
#                 min_distances = distances.min(dim=1)[0]
#                 weights = min_distances * remaining_mask
#                 centroid_idx = torch.multinomial(weights, 1)[0]
            
#             centroid = sequences[centroid_idx]
            
#             # Compute distances efficiently
#             distances = torch.sum((sequences - centroid.unsqueeze(0)) ** 2, dim=1)
#             matches = (distances < self.distance_threshold) & remaining_mask
            
#             if matches.sum() >= self.min_length:
#                 match_indices = matches.nonzero().squeeze(1)
#                 centroids.append({
#                     "pattern": centroid.cpu(),
#                     "length": length,
#                     "matches": [subsequences[i.item()] for i in match_indices.cpu().numpy()]
#                 })
            
#             remaining_mask &= ~matches
            
#         return centroids

#     def _discover_shapelets(self, train_loader: DataLoader):
#         """Discover shapelets from training data considering B x N x T format."""
#         from tqdm import tqdm

#         if self._load_shapelets():
#             return

#         print("No cached shapelets found. Discovering shapelets...")

#         # Get model predictions
#         all_data = []
#         all_masks = []
#         all_preds = []

#         with torch.no_grad():
#             for batch in tqdm(train_loader, desc="Getting predictions"):
#                 x, _, mask = batch  # x shape: (B, N, T)
#                 x = x.to(self.device)
#                 mask = mask.to(self.device)
#                 pred = self.base_model.predict(x, mask=mask, return_all=False)
#                 all_data.append(x.cpu())
#                 all_masks.append(mask.cpu())
#                 all_preds.append(pred.cpu())

#         # Concatenate all data
#         train_data = torch.cat(all_data, dim=0)  # (B, N, T)
#         train_masks = torch.cat(all_masks, dim=0)  # (B, N, T)
#         predictions = torch.cat(all_preds, dim=0).squeeze()  # (B,)

#         # Create boolean masks for high/low risk
#         high_risk_idx = predictions >= self.risk_threshold
#         low_risk_idx = ~high_risk_idx

#         # Split data based on predictions
#         # Expand index to match batch dimension only, features and time will be selected intact
#         high_risk_data = train_data[high_risk_idx]  # Will maintain (B', N, T) format
#         high_risk_masks = train_masks[high_risk_idx]  # Will maintain (B', N, T) format
#         low_risk_data = train_data[low_risk_idx]  # Will maintain (B', N, T) format
#         low_risk_masks = train_masks[low_risk_idx]  # Will maintain (B', N, T) format

#         print(
#             f"Found {high_risk_data.shape[0]} high-risk and {low_risk_data.shape[0]} low-risk samples"
#         )

#         # Process each feature separately
#         for f in tqdm(range(self.num_features), desc="Processing high-risk features"):
#             feature_shapelets = self._find_feature_shapelets(
#                 high_risk_data[:, f, :],  # (B', T) for feature f
#                 high_risk_masks[:, f, :],  # (B', T) for feature f
#             )
#             self.high_risk_shapelets.extend(feature_shapelets)

#         for f in tqdm(range(self.num_features), desc="Processing low-risk features"):
#             feature_shapelets = self._find_feature_shapelets(
#                 low_risk_data[:, f, :],  # (B', T) for feature f
#                 low_risk_masks[:, f, :],  # (B', T) for feature f
#             )
#             self.low_risk_shapelets.extend(feature_shapelets)

#         print(
#             f"Discovered {len(self.high_risk_shapelets)} high-risk and {len(self.low_risk_shapelets)} low-risk shapelets"
#         )
#         self._save_shapelets()

#     def _get_cache_path(self) -> pathlib.Path:
#         """Generate cache path for conditional shapelets."""
#         params = f"{self.data_name}_l{self.min_length}-{self.max_length}_n{self.n_shapelets}_d{self.distance_threshold}_t{self.risk_threshold}"
#         cache_dir = self.path / "shapelet_cache"
#         cache_dir.mkdir(parents=True, exist_ok=True)
#         return cache_dir / f"{params}.pt"

#     def _save_shapelets(self):
#         """Save conditional shapelets."""
#         cache_path = self._get_cache_path()
#         save_data = {
#             "high_risk": self._serialize_shapelets(self.high_risk_shapelets),
#             "low_risk": self._serialize_shapelets(self.low_risk_shapelets),
#         }
#         torch.save(save_data, cache_path)
#         print(f"Saved shapelets to {cache_path}")

#     def _load_shapelets(self) -> bool:
#         """Load shapelets if available."""
#         cache_path = self._get_cache_path()
#         if not cache_path.exists():
#             return False

#         try:
#             data = torch.load(cache_path)
#             self.high_risk_shapelets = self._deserialize_shapelets(data["high_risk"])
#             self.low_risk_shapelets = self._deserialize_shapelets(data["low_risk"])
#             print(
#                 f"Loaded {len(self.high_risk_shapelets)} high-risk and {len(self.low_risk_shapelets)} low-risk shapelets"
#             )
#             return True
#         except Exception as e:
#             print(f"Error loading shapelets: {e}")
#             return False

#     def _serialize_shapelets(self, shapelets):
#         return [
#             {
#                 "pattern": shapelet["pattern"].cpu().numpy(),
#                 "length": shapelet["length"],
#                 "matches": [
#                     {
#                         "sequence": match["sequence"].cpu().numpy(),
#                         "mask": match["mask"].cpu().numpy(),
#                         "start": match["start"],
#                         "series": match["series"],
#                     }
#                     for match in shapelet["matches"]
#                 ],
#             }
#             for shapelet in shapelets
#         ]

#     def _deserialize_shapelets(self, data):
#         return [
#             {
#                 "pattern": torch.from_numpy(shapelet["pattern"]),
#                 "length": shapelet["length"],
#                 "matches": [
#                     {
#                         "sequence": torch.from_numpy(match["sequence"]),
#                         "mask": torch.from_numpy(match["mask"]),
#                         "start": match["start"],
#                         "series": match["series"],
#                     }
#                     for match in shapelet["matches"]
#                 ],
#             }
#             for shapelet in data
#         ]

#     def get_name(self):
#         return "shapelet_motif_attribution"

# import pathlib
# import torch
# import numpy as np
# from typing import List, Dict, Optional
# from torch.utils.data import DataLoader
# from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------



import pathlib
import torch
import numpy as np
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


class ShapeletExplainer:
    def __init__(
        self,
        model,
        base_explainer,
        device,
        num_features: int,
        num_classes: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: Optional[DataLoader] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        :param model: your trained model
        :param base_explainer: e.g. 'ig'
        :param device: torch.device
        :param num_features: number of features (N)
        :param data_name: name of the dataset
        :param path: path to store shapelets
        :param train_loader: optional DataLoader for discovering shapelets
        :param feature_names: optional feature names
        :param p: parameter for something in your pipeline
        :param kwargs: extra parameters for shapelet extraction
        """
        self.device = device
        self.num_features = num_features
        self.data_name = data_name
        self.path = path
        self.feature_names = feature_names
        self.num_classes = num_classes

        # Shapelet discovery parameters
        self.min_length = kwargs.get("min_length", 5)
        self.max_length = kwargs.get("max_length", 10)
        self.n_shapelets = kwargs.get("n_shapelets", 5)
        self.distance_threshold = kwargs.get("distance_threshold", 0.1)
        self.batch_size = kwargs.get("batch_size", 32)

        # Store shapelets by class instead of “high-risk” vs. “low-risk”
        self.class_shapelets = {}

        # Cache for pattern indices, if needed
        self.pattern_indices = {}
        self._train_loader = train_loader

        self.base_model = model
        if base_explainer == "ig":
            from captum.attr import IntegratedGradients
            self.explainer = IntegratedGradients(self.base_model)

    # -------------------------------------------------------------------------
    # Baseline creation with shape (B, T, N)
    # -------------------------------------------------------------------------
    def _create_motif_aware_baseline(
        self, x: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Create baseline by removing temporal motifs (shapelets) associated with
        each sample's predicted class, then applying a forward fill.
        
        x shape: (B, T, N)
        mask shape: (B, T, N)
        pred shape: (B, C) or (B,)  [multi-class or single-label]
        """
        # Now, x.shape = (batch_size, seq_length, n_features)
        batch_size, seq_length, n_features = x.shape
        baseline = x.clone()

        # If pred is (B, C), pick the class with max logit/prob
        # Otherwise, if pred is already (B,), assume it's the final label

        predicted_label = torch.argmax(pred, dim=1)  # shape = (B,)


        for i in range(batch_size):
            cls_id = predicted_label[i].item()
            if cls_id not in self.class_shapelets or not self.class_shapelets[cls_id]:
                continue

            shapelets = self.class_shapelets[cls_id]

            # Group shapelets by length to reduce repeated computations
            patterns_by_length = {}
            for shapelet in shapelets:
                length = shapelet["length"]
                if length not in patterns_by_length:
                    patterns_by_length[length] = []
                patterns_by_length[length].append(shapelet["pattern"].to(x.device))

            # Remove shapelet matches from the baseline
            for length, patterns in patterns_by_length.items():
                pattern_tensor = torch.stack(patterns)  # (#patterns, length)

                # For each feature dimension
                for feat in range(n_features):
                    # We now unfold along the time dimension (dim=1 is time, but in slicing it's 0 after we pick sample i).
                    # shape -> (seq_length,)
                    windows = baseline[i, :, feat].unfold(0, length, 1)         # (#windows, length)
                    mask_windows = mask[i, :, feat].unfold(0, length, 1)        # (#windows, length)

                    if windows.numel() == 0:
                        continue

                    # Compute distances to shapelet patterns
                    distances = torch.cdist(windows, pattern_tensor)            # (#windows, #patterns)
                    quality = torch.exp(-distances)                             # higher => more similar
                    valid_mask = (mask_windows.sum(dim=1) >= length * 0.5)
                    matches = (quality > 0.3).any(dim=1) & valid_mask

                    # Where we have matches, set baseline to NaN
                    for window_idx in matches.nonzero().squeeze(-1):
                        baseline[i, window_idx : window_idx + length, feat] = float('nan')

        # Forward fill NaN
        # nan_mask = torch.isnan(baseline)
        # if nan_mask.any():
        #     filled = torch.zeros_like(baseline)
        #     for i in range(batch_size):
        #         for feat in range(n_features):
        #             valid_mask = ~nan_mask[i, :, feat]
        #             if not valid_mask.any():
        #                 # If all are NaN, fill with zeros
        #                 filled[i, :, feat] = 0.0
        #                 continue
        #             current_val = 0.0
        #             for t in range(seq_length):
        #                 if valid_mask[t]:
        #                     current_val = baseline[i, t, feat]
        #                 filled[i, t, feat] = current_val
        #     baseline = filled
        
        baseline[torch.isnan(baseline)] = 0

        return baseline

    # -------------------------------------------------------------------------
    # Main attribution entry point
    # -------------------------------------------------------------------------
    def attribute(self, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """
        Compute attributions with shape (B, T, N). 
        We first generate a baseline with _create_motif_aware_baseline, 
        then run the explainer in batches.
        """
        with torch.no_grad():
            # For multi-class, we might get shape (B, C).
            # If single-label, (B,).
            pred = self.base_model(x, mask=mask, return_all=False)

        # Build the baseline
        baseline = self._create_motif_aware_baseline(x, mask, pred)

        # Batch processing
        batch_size = x.shape[0]
        n_gpus = torch.cuda.device_count()
        chunk_size = self.batch_size * max(1, n_gpus)
        results = []

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            chunk_results = self._process_chunk(
                x[start_idx:end_idx],
                mask[start_idx:end_idx],
                baseline[start_idx:end_idx],
                pred[start_idx:end_idx],
            )
            results.append(chunk_results)

        return torch.cat(results, dim=0)
    # def attribute(self, x: torch.Tensor, mask: torch.Tensor):
    #     """
    #     1) Predict classes for all samples in x => group by class.
    #     2) For each class c, and each shapelet s in class_shapelets[c]:
    #        - Remove only that shapelet s for the sub-batch => baseline
    #        - Run IG => store
    #     3) Return a structure that contains IG for each (sample, shapelet).
        
    #     x: (N, T, F)
    #     mask: (N, T, F)
        
    #     Returns:
    #         A structure that holds IG for each shapelet. For instance, 
    #         a dict: {
    #            c -> {
    #               "shapelet_idx_0": (B_c, T, F),
    #               "shapelet_idx_1": (B_c, T, F),
    #               ...
    #            }
    #         }
    #         or a big tensor with shape (N, max_shapelets_c, T, F).
    #     """
    #     x = x.to(self.device)
    #     mask = mask.to(self.device)
    #     N, T, F = x.shape

    #     # 1) Predict classes for all N samples
    #     with torch.no_grad():
    #         pred = self.base_model(x, mask=mask, return_all=False)  # shape (N,C) or (N,)

    #     pred_labels = torch.argmax(pred, dim=1)  # shape (N,)


    #     # 2) Group sample indices by predicted class
    #     class_to_indices = defaultdict(list)
    #     for i, lbl in enumerate(pred_labels):
    #         class_to_indices[lbl.item()].append(i)

    #     # We'll store results in a dictionary for clarity:
    #     # ig_results[c][shapelet_index] = tensor of shape (B_c, T, F)
    #     # where B_c is the # of samples predicted as class c
    #     ig_results = {}

    #     # 3) For each class c
    #     for c, idx_list in class_to_indices.items():
    #         idx_tensor = torch.tensor(idx_list, device=self.device)  # shape (B_c,)
    #         B_c = len(idx_list)

    #         x_c = x[idx_tensor]         # shape (B_c, T, F)
    #         mask_c = mask[idx_tensor]   # shape (B_c, T, F)

    #         shapelets_for_c = self.class_shapelets.get(c, [])
    #         if not shapelets_for_c:
    #             continue

    #         ig_results[c] = {}

    #         # 4) For each shapelet in that class
    #         for s_idx, shapelet in enumerate(shapelets_for_c):
    #             # create baseline: remove only this shapelet from x_c
    #             baseline_c = self._create_single_shapelet_baseline(
    #                 x_c, mask_c, shapelet
    #             )  # (B_c, T, F)

    #             # run IG from baseline_c -> x_c in one pass
    #             ig_c_s = self.explainer.attribute(
    #                 x_c,
    #                 baselines=baseline_c,
    #                 target=c,  # or partial_targets if you prefer
    #                 additional_forward_args=(mask_c, None, False),
    #             ).detach().cpu()
    #             # shape (B_c, T, F)

    #             ig_results[c][f"shapelet_{s_idx}"] = ig_c_s

    #     return ig_results

    # # ----------------------------------------------------------------------
    # # Removes *one* shapelet from (B_c, T, F) sub-batch in a vectorized way
    # # ----------------------------------------------------------------------
    # def _create_single_shapelet_baseline(
    #     self,
    #     x_c: torch.Tensor,      # (B_c, T, F)
    #     mask_c: torch.Tensor,   # (B_c, T, F)
    #     shapelet: dict
    # ) -> torch.Tensor:
    #     """
    #     Remove only 'shapelet' from the sub-batch x_c => baseline.
    #     """
    #     baseline = x_c.clone()
    #     B_c, T, F = baseline.shape

    #     length = shapelet["length"]
    #     pattern = shapelet["pattern"].to(self.device)  # shape (length, ) 
    #                                                    # or possibly (length, Ffeat?)

    #     # Example: single-feature shapelet
    #     for b_idx in range(B_c):
    #         for f_idx in range(F):
    #             seq_1d = baseline[b_idx, :, f_idx]  # shape (T,)
    #             msk_1d = mask_c[b_idx, :, f_idx]    # shape (T,)

    #             if seq_1d.size(0) < length:
    #                 continue

    #             # slide window
    #             windows = seq_1d.unfold(0, length, 1)       # (#windows, length)
    #             mask_windows = msk_1d.unfold(0, length, 1)  # (#windows, length)

    #             # distance to shapelet pattern
    #             distances = torch.cdist(
    #                 windows.unsqueeze(1),
    #                 pattern.unsqueeze(0).unsqueeze(0),
    #                 p=2
    #             ).squeeze(-1).squeeze(-1)  # shape (#windows,)

    #             # threshold
    #             matches = (distances < self.distance_threshold)
    #             valid_mask = (mask_windows.sum(dim=1) >= length * 0.5)
    #             final_matches = matches & valid_mask

    #             for w_idx in final_matches.nonzero().squeeze():
    #                 baseline[b_idx, w_idx : w_idx + length, f_idx] = float('nan')

    #     baseline[torch.isnan(baseline)] = 0.0
    #     return baseline

    def _process_chunk(self, x_chunk, mask_chunk, baseline_chunk, pred_chunk):
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            split_size = len(x_chunk) // n_gpus
            futures = []
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                for i in range(n_gpus):
                    start = i * split_size
                    end = start + split_size if i < n_gpus - 1 else len(x_chunk)
                    futures.append(
                        executor.submit(
                            self._process_gpu_chunk,
                            x_chunk[start:end],
                            mask_chunk[start:end],
                            baseline_chunk[start:end],
                            pred_chunk[start:end],
                            f"cuda:{i}",
                        )
                    )
                results = [f.result() for f in futures]
            return torch.cat(results, dim=0)
        else:
            return self._process_gpu_chunk(
                x_chunk, mask_chunk, baseline_chunk, pred_chunk, "cpu"
            )

    def _process_gpu_chunk(self, x, mask, baseline, pred, device):
        x = x.to(device)
        mask = mask.to(device)
        baseline = baseline.to(device)
        
        with torch.autograd.set_grad_enabled(False):
            partial_targets = _run_forward(
                self.base_model,
                x,
                additional_forward_args=(mask, None, False),
            )
        partial_targets = torch.argmax(partial_targets, -1)
        
        return self.explainer.attribute(x, 
                                        baselines=baseline,
                                        target=partial_targets,
                                        additional_forward_args=(mask, None, False)).detach().cpu()

    # -------------------------------------------------------------------------
    # Shapelet discovery (B, T, N) version
    # -------------------------------------------------------------------------
    def _discover_shapelets(self, train_loader: DataLoader):
        """
        Discover shapelets for each predicted class, for data shaped (B, T, N).
        """
        from tqdm import tqdm

        if self._load_shapelets():
            return

        print("No cached shapelets found. Discovering shapelets...")

        all_data = []
        all_masks = []
        all_preds = []

        # Collect all data
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Getting predictions"):
                x, _, mask = batch  # shapes: x -> (B, T, N), mask -> (B, T, N)
                x = x.to(self.device)
                mask = mask.to(self.device)

                # For multi-class, we get shape (B, C)
                # For single-label, shape (B,)
                pred = self.base_model(x, mask=mask, return_all=False)

                all_data.append(x.cpu())
                all_masks.append(mask.cpu())
                all_preds.append(pred.cpu())

        train_data = torch.cat(all_data, dim=0)   # shape (B, T, N)
        train_masks = torch.cat(all_masks, dim=0) # shape (B, T, N)
        predictions = torch.cat(all_preds, dim=0) # shape (B, C) or (B,)

        # If multi-class (B, C), we pick the predicted label
        # If single-class (B,), skip argmax
        
        # print(predictions.shape)
        # raise RuntimeError
        predicted_labels = torch.argmax(predictions, dim=1)  # shape (B,)


        # Group indices by predicted label
        class_indices = {}
        for c in range(self.num_classes):
            class_indices[c] = (predicted_labels == c).nonzero().squeeze()

        # Build shapelets for each class
        for c in range(self.num_classes):
            idxs = class_indices[c]
            if idxs.numel() == 0:
                print(f"No samples predicted as class {c}, skipping.")
                self.class_shapelets[c] = []
                continue

            # Data for class c: shape (B_c, T, N)
            class_data = train_data[idxs]   # (B_c, T, N)
            class_masks = train_masks[idxs] # (B_c, T, N)

            print(f"Class {c}: found {class_data.shape[0]} samples.")

            # Extract shapelets by feature
            shapelets_for_this_class = []
            for f in tqdm(range(self.num_features), desc=f"Class {c} features"):
                # For feature f, slice: shape (B_c, T)
                feature_data = class_data[:, :, f]
                feature_mask = class_masks[:, :, f]

                # Extract shapelets for this single feature across time
                feature_shapelets = self._find_feature_shapelets(feature_data, feature_mask)
                shapelets_for_this_class.extend(feature_shapelets)

            self.class_shapelets[c] = shapelets_for_this_class

        total_shapes = sum(len(v) for v in self.class_shapelets.values())
        print(f"Discovered a total of {total_shapes} shapelets across {self.num_classes} classes.")

        self._save_shapelets()

    def _find_feature_shapelets(self, feature_data: torch.Tensor, feature_mask: torch.Tensor):
        """
        For a single feature across samples, discover shapelets by extracting
        subsequences in the time dimension. 
        feature_data shape: (B_c, T)
        feature_mask shape: (B_c, T)
        """
        discovered_shapelets = []
        B, T = feature_data.shape

        # For each length in [min_length ... max_length], collect subsequences
        subseqs_by_length = {l: [] for l in range(self.min_length, self.max_length + 1)}

        for i in range(B):
            seq = feature_data[i]  # shape (T,)
            msk = feature_mask[i]  # shape (T,)

            for length in range(self.min_length, self.max_length + 1):
                if length > T:
                    continue
                windows = seq.unfold(0, length, 1)      # shape (#windows, length)
                mask_windows = msk.unfold(0, length, 1) # shape (#windows, length)

                for w_idx, (window, mw) in enumerate(zip(windows, mask_windows)):
                    if mw.sum().item() < length * 0.5:
                        continue
                    subseqs_by_length[length].append(
                        {
                            "sequence": window.clone(),
                            "mask": mw.clone(),
                            "start": w_idx,
                            "series": i,
                        }
                    )

        # Cluster subsequences for each length
        for length, subsequences in subseqs_by_length.items():
            centroids = self._find_centroids(subsequences, length)
            discovered_shapelets.extend(centroids)

        return discovered_shapelets

    def _find_centroids(self, subsequences: List[Dict], length: int) -> List[Dict]:
        """
        Similar to k-means++ style centroid finding for shapelets.
        subsequences[i]["sequence"] -> shape (length,)
        """
        if not subsequences:
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequences = torch.stack([s["sequence"] for s in subsequences]).to(device)  # (num_subseqs, length)

        centroids = []
        remaining_mask = torch.ones(len(subsequences), dtype=bool, device=device)

        while len(centroids) < self.n_shapelets and remaining_mask.any():
            if len(centroids) == 0:
                # k-means++ initialization for the first centroid
                weights = torch.ones(sequences.shape[0], device=device)
                centroid_idx = torch.multinomial(weights, 1)[0]
            else:
                # k-means++ for subsequent centroids
                existing = torch.stack([c["pattern"] for c in centroids]).to(device)  # (#centroids, length)
                distances = torch.cdist(sequences, existing)                          # (num_subseqs, #centroids)
                min_dists = distances.min(dim=1)[0]
                weights = min_dists * remaining_mask
                centroid_idx = torch.multinomial(weights, 1)[0]

            centroid = sequences[centroid_idx]
            # Euclidean distance to centroid
            dist_to_centroid = torch.sum((sequences - centroid.unsqueeze(0)) ** 2, dim=1)
            matches = (dist_to_centroid < self.distance_threshold) & remaining_mask

            if matches.sum() >= self.min_length:
                match_indices = matches.nonzero().squeeze(1)
                centroids.append(
                    {
                        "pattern": centroid.cpu(),
                        "length": length,
                        "matches": [subsequences[j.item()] for j in match_indices.cpu().numpy()],
                    }
                )
            remaining_mask &= ~matches

        return centroids

    # -------------------------------------------------------------------------
    # Caching shapelets
    # -------------------------------------------------------------------------
    def _get_cache_path(self) -> pathlib.Path:
        """Generate a path for caching discovered shapelets."""
        # We can store the number of classes if known
        nc = getattr(self, "num_classes", "X")
        params = f"{self.data_name}_l{self.min_length}-{self.max_length}_n{self.n_shapelets}_d{self.distance_threshold}_nc{nc}"
        cache_dir = self.path / "shapelet_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{params}.pt"

    def _save_shapelets(self):
        """Save shapelets as a dictionary keyed by class."""
        cache_path = self._get_cache_path()
        save_data = {}
        for c, shapelets in self.class_shapelets.items():
            save_data[c] = self._serialize_shapelets(shapelets)
        torch.save(save_data, cache_path)
        print(f"Saved shapelets to {cache_path}")

    def _load_shapelets(self) -> bool:
        """Attempt to load cached shapelets."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False
        try:
            data = torch.load(cache_path)
            # Rebuild self.class_shapelets
            self.class_shapelets = {}
            for c, shapelets in data.items():
                c_int = int(c)  # In case keys were saved as strings
                self.class_shapelets[c_int] = self._deserialize_shapelets(shapelets)
            print(
                f"Loaded shapelets for classes: {list(self.class_shapelets.keys())}"
            )
            return True
        except Exception as e:
            print(f"Error loading shapelets: {e}")
            return False

    def _serialize_shapelets(self, shapelets):
        """Serialize shapelet patterns and matches to CPU numpy arrays."""
        return [
            {
                "pattern": sh["pattern"].cpu().numpy(),
                "length": sh["length"],
                "matches": [
                    {
                        "sequence": m["sequence"].cpu().numpy(),
                        "mask": m["mask"].cpu().numpy(),
                        "start": m["start"],
                        "series": m["series"],
                    }
                    for m in sh["matches"]
                ],
            }
            for sh in shapelets
        ]

    def _deserialize_shapelets(self, data):
        """Rebuild shapelets from serialized numpy arrays."""
        return [
            {
                "pattern": torch.from_numpy(sh["pattern"]),
                "length": sh["length"],
                "matches": [
                    {
                        "sequence": torch.from_numpy(m["sequence"]),
                        "mask": torch.from_numpy(m["mask"]),
                        "start": m["start"],
                        "series": m["series"],
                    }
                    for m in sh["matches"]
                ],
            }
            for sh in data
        ]

    def get_name(self):
        return "shapelet_motif_attribution"
