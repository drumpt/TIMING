from __future__ import annotations

from typing import Dict, List, Optional
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from winit.explainer.motif import TemporalMotif, MotifDiscovery
from winit.explainer.generator.explainers import GradientShapEnsembleExplainer


class TemporalMotif:
    def __init__(
        self,
        pattern: torch.Tensor,
        mask: torch.Tensor,
        frequency: int,
        importance_score: float,
    ):
        self.pattern = pattern
        self.mask = mask
        self.frequency = frequency
        self.importance_score = importance_score
        self.instances = []  # Store locations where this motif appears


class MotifDiscovery:
    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 10,
        distance_threshold: float = 0.1,
        min_frequency: int = 5,
        temporal_weight: float = 0.3,
        batch_size: int = 1000,  # Add batch processing
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.distance_threshold = distance_threshold
        self.min_frequency = min_frequency
        self.temporal_weight = temporal_weight
        self.batch_size = batch_size

    def _compute_representative_pattern(self, cluster_seqs, cluster_masks):
        """
        Compute a representative pattern for a cluster of subsequences

        Args:
            cluster_seqs (List[torch.Tensor]): List of sequences in the cluster
            cluster_masks (List[torch.Tensor]): Corresponding masks for the sequences

        Returns:
            torch.Tensor: Representative pattern for the cluster
        """
        # Convert to tensor for easier processing
        seqs_tensor = torch.stack(cluster_seqs)
        masks_tensor = torch.stack(cluster_masks)

        # Compute completeness of each sequence (proportion of valid observations)
        sequence_completeness = masks_tensor.float().mean(dim=1)

        # Normalize weights so they sum to 1
        weights = sequence_completeness / sequence_completeness.sum()

        # Compute weighted average of sequences
        weighted_seqs = seqs_tensor * masks_tensor.float() * weights[:, None, None]

        # Compute the representative pattern
        rep_pattern = weighted_seqs.sum(dim=0)

        return rep_pattern

    # (Rest of the existing methods from the previous implementation)
    def _cluster_subsequences(self, subsequences):
        """Memory-efficient clustering using batch processing"""
        n_subsequences = len(subsequences)
        labels = np.full(n_subsequences, -1)
        current_label = 0

        # Process subsequences in batches
        for i in range(0, n_subsequences, self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, n_subsequences)
            batch = subsequences[batch_start:batch_end]

            # Find neighbors for current batch
            neighbors = self._find_neighbors_batch(
                batch, subsequences, self.distance_threshold
            )

            # Assign cluster labels
            for j, neighs in enumerate(neighbors):
                if labels[batch_start + j] == -1 and len(neighs) >= self.min_frequency:
                    current_label = self._expand_cluster(
                        batch_start + j, neighs, labels, current_label
                    )

        return labels

    def _find_neighbors_batch(self, batch, all_subsequences, eps):
        """Find neighbors for a batch of subsequences"""
        neighbors = []

        for seq in batch:
            # Find neighbors using distance threshold
            seq_neighbors = []
            for j, other_seq in enumerate(all_subsequences):
                if (
                    self.compute_sequence_distance(
                        seq["sequence"],
                        other_seq["sequence"],
                        seq["mask"],
                        other_seq["mask"],
                    )
                    < eps
                ):
                    seq_neighbors.append(j)
            neighbors.append(seq_neighbors)

        return neighbors

    def _expand_cluster(self, point_idx, neighbors, labels, current_label):
        """Expand cluster from seed point"""
        labels[point_idx] = current_label

        # Process all neighbors
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = current_label
            i += 1

        return current_label + 1

    def discover_motifs(self, train_data, train_masks):
        """Main motif discovery process"""
        motifs = {}

        for length in range(self.min_length, self.max_length + 1):
            # Extract subsequences for current length
            subsequences = self._extract_subsequences_batch(
                train_data, train_masks, length
            )

            if len(subsequences) == 0:
                continue

            # Cluster subsequences efficiently
            cluster_labels = self._cluster_subsequences(subsequences)

            # Extract motifs from clusters
            length_motifs = self._extract_motifs(cluster_labels, subsequences)

            if length_motifs:
                motifs[length] = length_motifs

        return motifs

    def _extract_motifs(
        self, clusters: np.ndarray, subsequences: List[Dict]
    ) -> List[TemporalMotif]:
        """Extract representative motifs from clusters"""
        motifs = []

        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points
                continue

            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) < self.min_frequency:
                continue

            # Compute representative pattern
            cluster_seqs = [subsequences[i]["sequence"] for i in cluster_indices]
            cluster_masks = [subsequences[i]["mask"] for i in cluster_indices]

            rep_pattern = self._compute_representative_pattern(
                cluster_seqs, cluster_masks
            )

            importance = self._compute_motif_importance(
                cluster_seqs, cluster_masks, cluster_indices, subsequences
            )

            motif = TemporalMotif(
                rep_pattern,
                cluster_masks[0],  # Use first mask as representative
                len(cluster_indices),
                importance,
            )

            # Store instances
            for idx in cluster_indices:
                motif.instances.append(
                    {
                        "batch": subsequences[idx]["batch"],
                        "feature": subsequences[idx]["feature"],
                        "start": subsequences[idx]["start"],
                    }
                )

            motifs.append(motif)

        return motifs

    def _extract_subsequences_batch(self, train_data, train_masks, length):
        """Extract subsequences in a memory-efficient way"""
        subsequences = []
        batch_size, n_features, seq_length = train_data.shape

        for b in range(batch_size):
            for f in range(n_features):
                valid_start_indices = []

                # Find all valid starting points
                for start in range(seq_length - length + 1):
                    submask = train_masks[b, f, start : start + length]
                    if submask.sum() >= self.min_length:
                        valid_start_indices.append(start)

                # Process valid subsequences
                for start in valid_start_indices:
                    subsequences.append(
                        {
                            "sequence": train_data[b, f, start : start + length],
                            "mask": train_masks[b, f, start : start + length],
                            "batch": b,
                            "feature": f,
                            "start": start,
                        }
                    )

                # If subsequences list gets too large, yield batch
                if len(subsequences) >= self.batch_size:
                    subsequences_batch = subsequences
                    subsequences = []
                    return subsequences_batch

        return subsequences

    def compute_temporal_distance(self, mask1, mask2):
        """
        Compute distance between temporal patterns of two masks

        Args:
            mask1 (torch.Tensor): First mask
            mask2 (torch.Tensor): Second mask

        Returns:
            float: Temporal distance between masks
        """
        # Get gaps between observations
        gaps1 = self._compute_gaps(mask1)
        gaps2 = self._compute_gaps(mask2)

        if len(gaps1) == 0 or len(gaps2) == 0:
            return float("inf")

        # Pad gaps to the same length if needed
        max_length = max(len(gaps1), len(gaps2))

        padded_gaps1 = torch.full((max_length,), 0, dtype=torch.float32)
        padded_gaps2 = torch.full((max_length,), 0, dtype=torch.float32)

        padded_gaps1[: len(gaps1)] = gaps1.float()
        padded_gaps2[: len(gaps2)] = gaps2.float()

        # Compute mean absolute difference
        return torch.mean(torch.abs(padded_gaps1 - padded_gaps2)).item()

    def compute_sequence_distance(self, seq1, seq2, mask1, mask2):
        """
        Compute distance between two sequences considering both values and temporal gaps

        Args:
            seq1 (torch.Tensor): First sequence
            seq2 (torch.Tensor): Second sequence
            mask1 (torch.Tensor): Mask for first sequence
            mask2 (torch.Tensor): Mask for second sequence

        Returns:
            float: Distance between sequences
        """
        # Find valid observations for both sequences
        valid_indices1 = torch.where(mask1)[0]
        valid_indices2 = torch.where(mask2)[0]

        valid_seq1 = seq1[mask1.bool()]
        valid_seq2 = seq2[mask2.bool()]

        # If no valid observations, return infinite distance
        if len(valid_seq1) == 0 or len(valid_seq2) == 0:
            return float("inf")

        # Pad the shorter sequence to match the longer one
        max_length = max(len(valid_seq1), len(valid_seq2))

        # Pad sequences to the same length, using mean of valid values for padding
        seq1_mean = valid_seq1.mean().item() if len(valid_seq1) > 0 else 0
        seq2_mean = valid_seq2.mean().item() if len(valid_seq2) > 0 else 0

        padded_seq1 = torch.full((max_length,), seq1_mean, dtype=torch.float32)
        padded_seq2 = torch.full((max_length,), seq2_mean, dtype=torch.float32)

        padded_seq1[: len(valid_seq1)] = valid_seq1
        padded_seq2[: len(valid_seq2)] = valid_seq2

        # Compute MSE distance for values
        value_dist = torch.nn.functional.mse_loss(padded_seq1, padded_seq2).item()

        # Compute temporal distance
        temporal_dist = self.compute_temporal_distance(mask1, mask2)

        # Combine distances
        return value_dist + self.temporal_weight * temporal_dist

    def _compute_gaps(self, mask):
        """Compute temporal gaps between observations"""
        # Find positions of valid observations
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) <= 1:
            return torch.tensor([])

        # Compute gaps between consecutive observations
        return valid_indices[1:] - valid_indices[:-1]

    def _compute_motif_importance(
        self, cluster_seqs, cluster_masks, cluster_indices, subsequences
    ):
        """
        Compute the importance score for a motif

        This is a placeholder implementation that can be customized based on specific requirements.
        Currently, it returns a basic importance metric based on frequency and sequence completeness.

        Args:
            cluster_seqs (List[torch.Tensor]): Sequences in the cluster
            cluster_masks (List[torch.Tensor]): Masks for the sequences
            cluster_indices (np.ndarray): Indices of sequences in the cluster
            subsequences (List[Dict]): Original subsequence data

        Returns:
            float: Importance score of the motif
        """
        # Compute average sequence completeness
        completeness = torch.stack(cluster_masks).float().mean()

        # Use cluster size as a measure of frequency
        frequency_score = len(cluster_indices)

        # Combine completeness and frequency (you can adjust this formula)
        importance_score = completeness * frequency_score

        return importance_score.item()


class MotifExplainer(BaseExplainer):
    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        super().__init__(device)
        self.device = device
        self.num_features = num_features
        self.motif_discoverer = MotifDiscovery()
        self.data_name = data_name
        self.path = path
        self.train_loader = train_loader
        self.kwargs = kwargs
        self.explainer = None

        self.discovered_motifs = {}
        if train_loader is not None:
            self._discover_training_motifs(train_loader)

    def set_model(self, model, set_eval=True):
        super().set_model(model, set_eval=set_eval)
        self.explainer = GradientShapEnsembleExplainer(
            self.device,
            self.num_features,
            self.data_name,
            path=self.path,
            train_loader=self.train_loader,
            **self.kwargs,
        )
        self.explainer.set_model(model, set_eval=set_eval)

    def _discover_training_motifs(self, train_loader: DataLoader):
        """Discover motifs from training data"""
        train_data = torch.stack([x[0] for x in train_loader.dataset])
        train_masks = torch.stack([x[2] for x in train_loader.dataset])
        self.discovered_motifs = self.motif_discoverer.discover_motifs(
            train_data, train_masks
        )

    def attribute(self, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """Compute enhanced attribution scores"""
        # Get base attribution
        base_attribution = self.explainer.attribute(x, mask)

        # Enhance with motif-based attribution
        enhanced_attribution = self._enhance_attribution(x, mask, base_attribution)

        return enhanced_attribution

    def _enhance_attribution(
        self, x: torch.Tensor, mask: torch.Tensor, base_attribution: np.ndarray
    ) -> np.ndarray:
        """Enhance base attribution with motif information"""
        enhanced_attribution = base_attribution.copy()

        for length, motifs in self.discovered_motifs.items():
            for motif in motifs:
                # Find matches in current sequence
                matches = self._find_motif_matches(x, mask, motif)

                # Enhance attribution based on matches
                for match in matches:
                    start_idx = match["start"]
                    feature_idx = match["feature"]

                    # Enhance attribution scores for matched region
                    enhancement = motif.importance_score * self._compute_match_quality(
                        x[:, feature_idx, start_idx : start_idx + length],
                        mask[:, feature_idx, start_idx : start_idx + length],
                        motif,
                    )

                    enhanced_attribution[
                        :, feature_idx, start_idx : start_idx + length
                    ] += enhancement

        return enhanced_attribution

    def _find_motif_matches(
        self, x: torch.Tensor, mask: torch.Tensor, motif: TemporalMotif
    ) -> List[Dict]:
        """Find matches of a motif in the current sequence"""
        matches = []
        batch_size, n_features, seq_length = x.shape
        motif_length = len(motif.pattern)

        for b in range(batch_size):
            for f in range(n_features):
                for start in range(seq_length - motif_length + 1):
                    subseq = x[b, f, start : start + motif_length]
                    submask = mask[b, f, start : start + motif_length]

                    if submask.sum() >= self.motif_discoverer.min_length:
                        distance = self.motif_discoverer.compute_sequence_distance(
                            subseq, motif.pattern, submask, motif.mask
                        )

                        if distance < self.motif_discoverer.distance_threshold:
                            matches.append(
                                {
                                    "batch": b,
                                    "feature": f,
                                    "start": start,
                                    "distance": distance,
                                }
                            )

        return matches

    def get_name(self):
        return "motif_enhanced_attribution"
