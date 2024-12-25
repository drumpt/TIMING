from __future__ import annotations

import os
import gc
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from winit.dataloader import WinITDataset
from winit.utils import aggregate_scores


class BoxPlotter:
    """
    A class for plotting various box plots as plotExampleBox in FIT repo.
    """

    def __init__(
        self,
        dataset: WinITDataset,
        base_plot_path: pathlib.Path,
        num_to_plot: int,
        explainer_name: str,
    ):
        """
        Constructor.

        Args:
            dataset:
                The dataset.
            base_plot_path:
                The base plot path for the plots
            num_to_plot:
                The number of samples to plot.
            explainer_name:
                The name of the explainer.
        """
        self.plot_path = base_plot_path / dataset.get_name()
        self.num_to_plot = num_to_plot
        self.explainer_name = explainer_name
        testset = list(dataset.test_loader.dataset)
        self.x_test = torch.stack([x[0] for x in testset]).cpu().numpy()
        self.y_test = torch.stack([x[1] for x in testset]).cpu().numpy()
        self.mask_test = (
            torch.stack([x[2] for x in testset]).cpu().numpy()
        )  # Added mask storage
        self.plot_path.mkdir(parents=True, exist_ok=True)

    def plot_combined_visualization(
            self,
            importances: Dict[int, np.ndarray],
            aggregate_method: str,
            x_test: np.ndarray,
            mask_test: np.ndarray,
            new_xs: Dict[int, np.ndarray],
            new_masks: Dict[int, np.ndarray],
            importance_masks: Dict[int, np.ndarray],
            orig_preds: Dict[int, np.ndarray],
            new_preds: Dict[int, np.ndarray],
            y_test: np.ndarray,
            mask_name: str,
        ) -> None:
            # Set memory-efficient backend
            plt.switch_backend("Agg")

            # Set global font settings
            plt.rcParams["font.family"] = ["DejaVu Serif"]
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["figure.max_open_warning"] = False

            # Define font sizes
            TITLE_SIZE = 16
            SUBTITLE_SIZE = 14
            LABEL_SIZE = 12

            for cv, importance_unaggregated in importances.items():
                importance_scores = aggregate_scores(
                    importance_unaggregated, aggregate_method
                )

                for i in range(self.num_to_plot):
                    try:
                        # Extract scalar values for title
                        label_val = float(y_test[i])
                        orig_pred_val = float(orig_preds[cv][i])
                        new_pred_val = float(new_preds[cv][i])

                        # Create figure with 3x2 subplots
                        fig, axes = plt.subplots(3, 2, figsize=(15, 14))

                        # Main title
                        title_text = f"Sample {i} (Label: {label_val:.0f} Original Pred: {orig_pred_val:.3f} Masked Pred: {new_pred_val:.3f})"
                        fig.suptitle(title_text, fontsize=TITLE_SIZE, y=0.95)

                        # Function to add calibrated indices
                        def add_calibrated_indices(ax, data):
                            num_rows, num_cols = data.shape
                            # Add row indices on the left with smaller font
                            ax.set_yticks(range(num_rows))
                            ax.set_yticklabels(range(num_rows), fontsize=8)
                            # Add column indices on the bottom with smaller font and rotation
                            ax.set_xticks(range(num_cols))
                            ax.set_xticklabels(range(num_cols), fontsize=8, rotation=90)
                            ax.tick_params(axis='both', which='major', length=0)

                        # Plot configurations with modified settings for input visualization
                        plot_configs = [
                            {
                                "data": x_test[i],
                                "title": "Original Input",
                                "cmap": "RdBu_r",  # Red-Blue diverging colormap
                                "pos": (0, 0),
                                "symmetric": True,  # Enable symmetric scaling
                            },
                            {
                                "data": new_xs[cv][i],
                                "title": "Masked Input",
                                "cmap": "RdBu_r",  # Red-Blue diverging colormap
                                "pos": (0, 1),
                                "symmetric": True,  # Enable symmetric scaling
                            },
                            {
                                "data": mask_test[i],
                                "title": "Original Mask",
                                "cmap": "Reds",
                                "pos": (1, 0),
                                "symmetric": False,
                            },
                            {
                                "data": new_masks[cv][i],
                                "title": "New Mask",
                                "cmap": "Reds",
                                "pos": (1, 1),
                                "symmetric": False,
                            },
                            {
                                "data": np.abs(importance_scores[i]),
                                "title": "Feature Importance",
                                "cmap": "Blues",
                                "pos": (2, 0),
                                "symmetric": False,
                            },
                            {
                                "data": importance_masks[cv][i],
                                "title": "Selected Mask",
                                "cmap": "gray",
                                "pos": (2, 1),
                                "symmetric": False,
                            },
                        ]

                        # Create all subplots
                        for config in plot_configs:
                            ax = axes[config["pos"]]
                            data = config["data"]
                            
                            # Set symmetric vmin/vmax for input visualization
                            if config["symmetric"]:
                                abs_max = np.max(np.abs(data))
                                vmin, vmax = -abs_max, abs_max
                            else:
                                vmin, vmax = None, None

                            im = ax.imshow(
                                data,
                                cmap=config["cmap"],
                                interpolation="nearest",
                                aspect="auto",
                                vmin=vmin,
                                vmax=vmax,
                            )

                            # Add calibrated indices
                            add_calibrated_indices(ax, data)

                            # Add grid for cells
                            num_rows, num_cols = data.shape
                            ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
                            ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
                            ax.grid(
                                which="minor",
                                color="gray",
                                linestyle="-",
                                linewidth=0.5,
                                alpha=0.3,
                            )

                            ax.set_title(config["title"], fontsize=SUBTITLE_SIZE, pad=10)

                            cbar = plt.colorbar(im, ax=ax)
                            cbar.ax.tick_params(labelsize=LABEL_SIZE)

                            for spine in ax.spines.values():
                                spine.set_visible(True)
                                spine.set_linewidth(1.5)

                        plt.tight_layout(rect=[0, 0, 1, 0.95])

                        # Save figure
                        save_path = self._get_plot_file_name(
                            i, f"{self.explainer_name}/{aggregate_method}/{mask_name}/cv_{cv}"
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        # Save with lower DPI and compression
                        plt.savefig(
                            save_path,
                            bbox_inches="tight",
                            dpi=200,  # Reduced DPI
                        )

                    except Exception as e:
                        print(f"Error plotting sample {i}: {str(e)}")

                    finally:
                        plt.close("all")  # Ensure all figures are closed

                    # Clear memory
                    if i % 5 == 0:
                        gc.collect()

    def plot_importances(
        self, importances: Dict[int, np.ndarray], aggregate_method: str
    ) -> None:
        """
        Plot the importance for all cv and save it to files.

        Args:
            importances:
                A dictionary from CV to feature importances.
            aggregate_method:
                The aggregation method for WinIT.
        """
        for cv, importance_unaggregated in importances.items():
            importance_scores = aggregate_scores(
                importance_unaggregated, aggregate_method
            )
            for i in range(self.num_to_plot):
                prefix = (
                    f"{self.explainer_name}_{aggregate_method}_cv_{cv}_attributions"
                )
                self._plot_example_box(
                    np.abs(importance_scores[i]), self._get_plot_file_name(i, prefix)
                )

    def plot_ground_truth_importances(self, ground_truth_importance: np.ndarray):
        """
        Plot the ground truth importances for all cv and save it to files.

        Args:
            ground_truth_importance:
                The ground truth importance.
        """
        prefix = "ground_truth_attributions"
        for i in range(self.num_to_plot):
            self._plot_example_box(
                ground_truth_importance[i], self._get_plot_file_name(i, prefix)
            )

    def plot_labels(self):
        """
        Plot the labels and save it to files. If the label is one-dimensional, skip the plotting.
        """
        if self.y_test.ndim != 2:
            return
        for i in range(self.num_to_plot):
            self._plot_example_box(
                self.y_test[i : i + 1], self._get_plot_file_name(i, prefix="labels")
            )

    def plot_x_pred(
        self,
        x: np.ndarray | Dict[int, np.ndarray] | None,
        preds: Dict[int, np.ndarray],
        prefix: str = None,
    ):
        """
        Plot the data and the corresponding predictions and save it to files.

        Args:
            x:
                The data. Can be a numpy array of a dictionary of CV to numpy arrays.
            preds:
                The predictions. A dictionary of CV to numpy arrays. (In case of only 1 data,
                the predictions are the predictions of the model of the corresponding CV
                on the same data.
            prefix:
                The prefix of the name of the files to be saved.
        """
        if x is None:
            x = self.x_test

        if isinstance(x, np.ndarray):
            for i in range(self.num_to_plot):
                filename_prefix = prefix if prefix is not None else "data"
                self._plot_example_box(
                    x[i], self._get_plot_file_name(i, prefix=filename_prefix)
                )
        elif isinstance(x, dict):
            for cv, xin in x.items():
                for i in range(self.num_to_plot):
                    filename_prefix = (
                        f"data_cv_{cv}" if prefix is None else f"{prefix}_cv_{cv}"
                    )
                    self._plot_example_box(
                        xin[i], self._get_plot_file_name(i, prefix=filename_prefix)
                    )

        for cv, pred in preds.items():
            for i in range(self.num_to_plot):
                filename_prefix = (
                    f"preds_cv_{cv}" if prefix is None else f"{prefix}_cv_{cv}"
                )
                self._plot_example_box(
                    pred[i].reshape(1, -1), self._get_plot_file_name(i, filename_prefix)
                )

    def _get_plot_file_name(self, i: int, prefix: str) -> pathlib.Path:
        return self.plot_path / f"{prefix}" / f"{i}.png"

    @staticmethod
    def _plot_example_box(input_array, save_location: pathlib.Path):
        fig, ax = plt.subplots()
        plt.axis("off")

        ax.imshow(input_array, interpolation="nearest", cmap="gray")

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(str(save_location), bbox_inches="tight", pad_inches=0)

        plt.close()
