from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from winit.utils import aggregate_scores


class Masker:
    """
    A class used to compute performance drop using different masking methods.
    """

    def __init__(
        self,
        mask_method: str,
        top: int | float,
        balanced: bool,
        seed: int,
        absolutize: bool,
        aggregate_method: str = "mean",
    ):
        """
        Constructor

        Args:
            mask_method:
                The masking method. Must be "end", "std" or "end_fit". Note that "end" will
                mask the all subsequent observations if the observation is deemed important. "std"
                will mask all subsequent observations until the value changed by 1 STD of the
                feature distribution. "end_fit" is like "end", but with the code of the original
                FIT repository and started masking when t >= 10.
            top:
               If int, it will mask the top `top` observations of each time series. If float, it
               will mask the top `top*100` percent of the observations for all time series.
            balanced:
               If True, the number of masked will be balanced std. It is used for the STD-BAL
               masking method in the paper.
            seed:
               The random seed.
            absolutize:
               Indicate whether we should absolutize the feature importance for determining the
               top features.
            aggregate_method:
               For features that contain a window size, i.e. WinIT, this describes the aggregation
               method. It can be "absmax", "max", "mean".
        """
        if mask_method not in ["end", "std", "end_fit", "mam"]:
            raise NotImplementedError(f"Mask method {mask_method} unrecognized")
        self.mask_method = mask_method
        self.top = top
        self.balanced = balanced
        self.seed = seed
        self.local = isinstance(top, int)
        min_time_dict = {"std": 1, "end": 1, "end_fit": 10, "mam": 1}
        self.min_time = min_time_dict[self.mask_method]
        self.importance_threshold = -1000
        self.absolutize = absolutize
        self.aggregate_method = aggregate_method
        # assert not balanced or self.local and mask_method in ["std", "end"]

        self.start_masked_count = None
        self.all_masked_count = None
        self.feature_masked = None

    def get_name(self):
        return self.mask_method

    def mask(
        self,
        x_test: np.ndarray,
        mask_test: np.ndarray,
        importance_scores: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Perform masking.

        Args:
            x_test:
                The original input of shape (num_samples, num_features, num_times)
            mask_test:
                The original mask of shape (num_samples, num_features, num_times)
            importance_scores:
                The importance score dictionary from CV to numpy arrays.

        Returns:
            A tuple of three dictionaries:
            - new_xs: CV to numpy arrays of masked data
            - new_masks: CV to numpy arrays of updated masks
            - maskeds: CV to numpy arrays of boolean masked indicators
        """
        if self.mask_method == "mam":
            return self.missing_aware_mask(x_test, mask_test, importance_scores)

        new_xs = {}
        new_masks = {}
        maskeds = {}
        start_masked_count = {}
        all_masked_count = {}
        feature_masked = {}
        num_samples, num_features, num_times = x_test.shape
        
        for cv, importance_score in importance_scores.items():
            importance_score = aggregate_scores(importance_score, self.aggregate_method)
            if self.absolutize:
                importance_score = np.abs(importance_score)
            coordinate_list = self._generate_arg_sort(importance_score, randomize_ties=True)

            new_x = x_test.copy()
            new_mask = mask_test.copy()
            masked = np.zeros_like(new_x, dtype=bool)
            start_masked = np.zeros_like(new_x, dtype=bool)
            
            if self.mask_method in ["std", "end"]:
                if not self.local:
                    num_feature_time_drop = int(len(coordinate_list) * self.top)
                    coordinate_list = coordinate_list[:num_feature_time_drop]
                    for coordinate in coordinate_list:
                        sample_id, feature_id, time_id = coordinate
                        if importance_score[sample_id, feature_id, time_id] <= self.importance_threshold:
                            break
                        if masked[sample_id, feature_id, time_id]:
                            continue
                        end_time, new_x[sample_id, feature_id, :] = self._carry_forward(
                            time_id, new_x[sample_id, feature_id, :], self.mask_method
                        )
                        masked[sample_id, feature_id, time_id:end_time] = True
                        start_masked[sample_id, feature_id, time_id] = True
                        new_mask[sample_id, feature_id, time_id:end_time] = 1
                else:
                    num_feature_time = new_x.shape[1] * (new_x.shape[2] - self.min_time)
                    for sample_id in range(new_x.shape[0]):
                        start = sample_id * num_feature_time
                        num_masked_total = 0
                        num_masked = 0
                        for cur in range(num_feature_time):
                            sample_index, feature_id, time_id = coordinate_list[start + cur]
                            if sample_index != sample_id:
                                raise RuntimeError("Failed sanity check!")

                            balance_condition = self.balanced and num_masked_total >= self.top
                            num_drop_condition = num_masked >= self.top
                            threshold_condition = (
                                importance_score[sample_id, feature_id, time_id]
                                <= self.importance_threshold
                            )
                            if threshold_condition or balance_condition or num_drop_condition:
                                break
                            if masked[sample_index, feature_id, time_id]:
                                continue
                            end_ts, new_x[sample_id, feature_id, :] = self._carry_forward(
                                time_id,
                                new_x[sample_id, feature_id, :],
                                self.mask_method,
                            )
                            masked[sample_id, feature_id, time_id:end_ts] = True
                            start_masked[sample_id, feature_id, time_id] = True
                            new_mask[sample_id, feature_id, time_id:end_ts] = 1
                            num_masked_total += end_ts - time_id
                            num_masked += 1
                            
            elif self.mask_method == "end_fit":
                for i, x in enumerate(new_x):
                    if not self.local:
                        q = np.percentile(importance_score[:, :, self.min_time:],
                                        100 - self.top * 100)
                        min_t_feat = [
                            np.min(np.where(importance_score[i, f, self.min_time:] >= q)[0]) if
                            len(np.where(importance_score[i, f, self.min_time:] >= q)[0]) > 0 else
                            x.shape[-1] - self.min_time - 1 for f in range(num_features)
                        ]
                        for f in range(importance_score[i].shape[0]):
                            x[f, min_t_feat[f] + self.min_time:] = x[f, min_t_feat[f] + self.min_time - 1]
                            masked[i, f, min_t_feat[f] + self.min_time:] = True
                            start_masked[i, f, min_t_feat[f] + self.min_time] = True
                            new_mask[i, f, min_t_feat[f] + self.min_time:] = 1
                    else:
                        for _ in range(self.top):
                            imp = np.unravel_index(importance_score[i, :, self.min_time:].argmax(),
                                                importance_score[i, :, self.min_time:].shape)
                            importance_score[i, imp[0], imp[1] + self.min_time:] = -1
                            x[imp[0], imp[1] + self.min_time:] = x[imp[0], imp[1] + self.min_time - 1]
                            masked[i, imp[0], imp[1] + self.min_time:] = True
                            start_masked[i, imp[0], imp[1] + self.min_time] = True
                            new_mask[i, imp[0], imp[1] + self.min_time:] = 1

            start_masked_count[cv] = np.sum(start_masked, axis=0)
            all_masked_count[cv] = np.sum(masked, axis=0)
            feature_masked[cv] = np.sum(np.sum(start_masked, axis=1) > 0, axis=0)

            new_xs[cv] = new_x
            new_masks[cv] = new_mask
            maskeds[cv] = masked

        self.start_masked_count = start_masked_count
        self.all_masked_count = all_masked_count
        self.feature_masked = feature_masked

        return new_xs, new_masks, maskeds

    def missing_aware_mask(
        self,
        x_test,
        mask_test,
        importance_scores
    ):
        """
        Missing-aware masking implementation with carry forward.
        """
        new_xs = {}
        new_masks = {}
        maskeds = {}
        start_masked_count = {}
        all_masked_count = {}
        feature_masked = {}
        batch_size, num_features, num_times = x_test.shape

        for cv, importance_score in importance_scores.items():
            importance_score = aggregate_scores(importance_score, self.aggregate_method)
            if self.absolutize:
                importance_score = np.abs(importance_score)

            new_x = x_test.copy()
            new_mask = mask_test.copy()
            masked = np.zeros_like(new_x, dtype=bool)
            start_masked = np.zeros_like(new_x, dtype=bool)

            for b in range(batch_size):
                all_groups = []
                num_masked_total = 0
                num_masked = 0

                # First, identify all groups and their last real values
                for f in range(num_features):
                    groups = []
                    current_group = []
                    last_real_value = None

                    for t in range(num_times):
                        if mask_test[b, f, t] == 0:  # Real value
                            if current_group:
                                if last_real_value is not None:  # Store last real value with group
                                    groups.append((current_group, last_real_value))
                                else:
                                    groups.append((current_group, None))  # For first group
                            current_group = [t]
                            last_real_value = t
                        elif last_real_value is not None:
                            current_group.append(t)

                    if current_group:
                        groups.append((current_group, last_real_value))

                    for group, last_real in groups:
                        if last_real is not None:  # Only consider groups with a previous real value
                            group_score = np.max(importance_score[b, f, group])
                            all_groups.append((group_score, f, group, last_real))

                all_groups.sort(reverse=True)

                if isinstance(self.top, float):
                    num_groups_to_mask = int(len(all_groups) * self.top)
                else:
                    num_groups_to_mask = min(
                        int(self.top * len(all_groups) / (num_features * num_times)),
                        len(all_groups)
                    )

                # Mask groups with carry forward
                for i in range(num_groups_to_mask):
                    if i < len(all_groups):
                        _, feature_idx, group, last_real = all_groups[i]
                        start_time = group[0]
                        end_time = group[-1] + 1

                        if start_time > 0:
                            # Get the value to carry forward
                            carry_value = new_x[b, feature_idx, last_real]
                            new_x[b, feature_idx, start_time:end_time] = carry_value
                            
                            # Update masks
                            new_mask[b, feature_idx, start_time:end_time] = 1
                            masked[b, feature_idx, start_time:end_time] = True
                            start_masked[b, feature_idx, start_time] = True
                            num_masked_total += end_time - start_time
                            num_masked += 1

            # Update counters
            start_masked_count[cv] = np.sum(start_masked, axis=0)
            all_masked_count[cv] = np.sum(masked, axis=0)
            feature_masked[cv] = np.sum(np.sum(start_masked, axis=1) > 0, axis=0)

            new_xs[cv] = new_x
            new_masks[cv] = new_mask
            maskeds[cv] = masked

        self.start_masked_count = start_masked_count
        self.all_masked_count = all_masked_count
        self.feature_masked = feature_masked

        return new_xs, new_masks, maskeds

    def _generate_arg_sort(self, scores, randomize_ties=True):
        """
        Returns a list of coordinates that is the argument of the sorting. In descending order.
        If local is True, the list of coordinates will be sorted within each sample. i.e.,
        the first (num_features * num_times) coordinates would always correspond to the first
        sample, the next (num_features * num_times) coordinates would correspond to the second
        sample, etc.

        Args:
            scores:
               Importance scores of shape (num_samples, num_features, num_times)
            min_time:
               The minimum timesteps to sort.
            local:
               Indicates whether the sorting is local or global. i.e. along all axes, or just the
               feature and time axes.
            randomize_ties:
               If there is a tie in the scores (which happens very often in Dynamask), we randomly
               permute the coordinates across the tie.

        Returns:
            A array of shape (all_coordinate_length, 3), for each row is the coordinate.
        """
        truncated_scores = scores[:, :, self.min_time :]
        if self.local:
            flattened_scores = truncated_scores.reshape(scores.shape[0], -1)
            argsorted_ravel_local = np.argsort(flattened_scores)[:, ::-1]
            if randomize_ties:
                self._shuffle_ties(argsorted_ravel_local, flattened_scores)
            feature_index = argsorted_ravel_local // truncated_scores.shape[2]
            time_index = (
                argsorted_ravel_local % truncated_scores.shape[2]
            ) + self.min_time
            arange = (
                np.arange(scores.shape[0])
                .reshape(-1, 1)
                .repeat(feature_index.shape[1], 1)
            )
            coordinate_list = np.stack([arange, feature_index, time_index]).reshape(
                3, -1
            )  # (3, all_coordinate_length)
            return coordinate_list.transpose()
        else:
            flattened_scores = truncated_scores.ravel()
            argsorted_ravel_global = np.argsort(flattened_scores)[::-1]
            if randomize_ties:
                self._shuffle_ties_global(argsorted_ravel_global, flattened_scores)
            coordinate_list = np.stack(
                np.unravel_index(argsorted_ravel_global, truncated_scores.shape)
            )  # (3, all_coordinate_length)
            coordinate_list[2, :] += self.min_time
            return coordinate_list.transpose()

    def _shuffle_ties_global(self, argsorted_ravel_global, flattened_scores):
        sorted_scores = flattened_scores[argsorted_ravel_global]
        repeated = np.r_[False, sorted_scores[1:] == sorted_scores[:-1]]
        indices = np.where(np.diff(repeated))[0]
        if len(indices) % 2 == 1:
            indices = np.r_[indices, len(sorted_scores) - 1]
        indices = indices.reshape(-1, 2)
        indices[:, 1] += 1
        rng = np.random.default_rng(self.seed)
        for repeated_index in indices:
            from_index, to_index = repeated_index
            rng.shuffle(argsorted_ravel_global[from_index:to_index])

    def _shuffle_ties(self, argsorted_ravel_local, flattened_scores):
        sorted_scores = np.take_along_axis(
            flattened_scores, argsorted_ravel_local, axis=1
        )
        repeated = np.concatenate(
            [
                np.zeros((len(sorted_scores), 1), dtype=bool),
                sorted_scores[:, 1:] == sorted_scores[:, :-1],
            ],
            axis=1,
        )
        indices = np.where(np.diff(repeated, axis=1))
        previous_sample_id = -1
        left = -1
        rng = np.random.default_rng(self.seed)
        for sample_id, x in zip(*indices):
            if sample_id != previous_sample_id:
                # new sample seen
                if left != -1:
                    # still have right bound left.
                    right = sorted_scores.shape[1]
                    rng.shuffle(argsorted_ravel_local[previous_sample_id, left:right])
                left = x
            else:
                # same sample
                if left != -1:
                    rng.shuffle(argsorted_ravel_local[previous_sample_id, left : x + 1])
                    left = -1
                else:
                    left = x
            previous_sample_id = sample_id
        if left != -1:
            right = sorted_scores.shape[1]
            rng.shuffle(argsorted_ravel_local[previous_sample_id, left:right])

    def _carry_forward(self, timestep, time_series, mask):
        assert len(time_series.shape) == 1
        ts_length = time_series.shape[0]

        assert timestep != 0, "Carry forward is not defined at index 0"

        if mask == "std":
            threshold = np.std(time_series)
            old = time_series[timestep]
            segment = np.abs(time_series[timestep:] - old) > threshold
            over = np.where(segment)[0]
            new_timestep = ts_length if len(over) == 0 else over[0] + timestep
        elif mask == "end":
            new_timestep = ts_length
        else:
            raise NotImplementedError(
                f"Mask method {mask} not recognized for carry forward"
            )

        time_series[timestep:new_timestep] = time_series[timestep - 1]
        return new_timestep, time_series

    def get_name(self) -> str:
        if self.local:
            if self.balanced:
                return f"bal{int(self.top)}_{self.mask_method}_{self.aggregate_method}"
            return f"top{int(self.top)}_{self.mask_method}_{self.aggregate_method}"
        return (
            f"globaltop{int(self.top * 100)}_{self.mask_method}_{self.aggregate_method}"
        )
