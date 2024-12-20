from __future__ import annotations

import logging
import pathlib
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from winit.explainer.explainers import BaseExplainer
from winit.explainer.generator.generator import (
    FeatureGenerator,
    BaseFeatureGenerator,
    GeneratorTrainingResults,
)
from winit.explainer.generator.jointgenerator import JointFeatureGenerator


class WinITSetAllExplainer(BaseExplainer):
    """
    The explainer for our method WinIT
    """

    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: DataLoader | None = None,
        window_size: int = 10,
        num_samples: int = 1,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        random_state: int | None = None,
        args=None,
        **kwargs,
    ):
        """
        Construtor

        Args:
            device:
                The torch device.
            num_features:
                The number of features.
            data_name:
                The name of the data.
            path:
                The path indicating where the generator to be saved.
            train_loader:
                The train loader if we are using the data distribution instead of a generator
                for generating counterfactual. Default=None.
            window_size:
                The window size for the WinIT
            num_samples:
                The number of Monte-Carlo samples for generating counterfactuals.
            conditional:
                Indicate whether the individual feature generator we used are conditioned on
                the current features. Default=False
            joint:
                Indicate whether we are using the joint generator.
            metric:
                The metric for the measures of comparison of the two distributions for i(S)_a^b
            random_state:
                The random state.
            **kwargs:
                There should be no additional kwargs.
        """
        super().__init__(device)
        self.window_size = window_size
        self.num_samples = num_samples  # we don't need multiple samples
        assert self.num_samples == 1
        self.num_features = num_features
        self.data_name = data_name
        self.joint = joint
        self.conditional = conditional
        self.metric = metric
        self.args = args

        self.generators: BaseFeatureGenerator | None = None
        self.path = path
        if train_loader is not None:
            self.data_distribution = (
                torch.stack([x[0] for x in train_loader.dataset]).detach().cpu().numpy()
            )
        else:
            self.data_distribution = None
        self.rng = np.random.default_rng(random_state)

        self.log = logging.getLogger(WinITSetAllExplainer.__name__)
        if len(kwargs):
            self.log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def _model_predict(self, x, mask=None, timesteps=None):
        # print(f"{x=}")
        # print(f"in _model_predict {x.shape=}")
        """
        Run predict on base model. If the output is binary, i.e. num_class = 1, we will make it
        into a probability distribution by append (p, 1-p) to it.
        """
        p = self.base_model.predict(x, mask, timesteps, return_all=False)
        if self.base_model.num_states == 1:
            # Create a 'probability distribution' (p, 1 - p)
            prob_distribution = torch.cat((p, 1 - p), dim=1)
            return prob_distribution
        return p

    def attribute(self, x, mask=None):
        """
        Compute the WinIT attribution only for the final timestep prediction.

        Args:
            x: The input Tensor of shape (batch_size, num_features, num_timesteps)
            mask: Mask tensor indicating invalid positions

        Returns:
            The attribution Tensor of shape (batch_size, num_features, num_timesteps)
            The (i, j, k)-entry is the importance of observation (i, j, k) to the final prediction
        """
        self.base_model.eval()
        self.base_model.zero_grad()

        with torch.no_grad():
            tic = time()
            batch_size, num_features, num_timesteps = x.shape

            # Get prediction for full sequence
            p_y = self._model_predict(x, mask)
            timesteps = (
                torch.linspace(0, 1, num_timesteps, device=x.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

            # Initialize scores array
            scores = np.zeros((batch_size, num_features, num_timesteps), dtype=float)

            # For each timestep in the past
            for time_past in range(num_timesteps):
                # Generate counterfactuals
                counterfactuals = torch.zeros(
                    self.num_samples,  # number of Monte-Carlo samples
                    batch_size,  # batch size
                    num_features,  # feature dimension
                    num_timesteps - time_past,  # time forward
                    device=x.device,  # same device as input
                )

                # For each feature
                for f in range(num_features):
                    # Repeat input for num samples
                    x_hat_in = x.unsqueeze(0).repeat(
                        self.num_samples, 1, 1, 1
                    )  # (ns, bs, f, time)

                    # Replace values with counterfactuals
                    x_hat_in[:, :, :, time_past:num_timesteps] = counterfactuals
                    assert torch.count_nonzero(x_hat_in[:, :, :, time_past:num_timesteps]) == 0
                    
                    mask_hat_in = mask.unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
                    mask_hat_in[:, :, :, time_past:num_timesteps] = 1  # Doesn't exist
                    time_hat_in = timesteps.unsqueeze(0).repeat(self.num_samples, 1, 1)

                    # Get predictions for counterfactuals
                    p_y_hat = self._model_predict(
                        x_hat_in.reshape(
                            self.num_samples * batch_size, num_features, num_timesteps
                        ),
                        mask_hat_in.reshape(
                            self.num_samples * batch_size, num_features, num_timesteps
                        ),
                        time_hat_in.reshape(
                            self.num_samples * batch_size, num_timesteps
                        ),
                    )

                    # Expand original predictions
                    p_y_exp = (
                        p_y.unsqueeze(0)
                        .repeat(self.num_samples, 1, 1)
                        .reshape(self.num_samples * batch_size, p_y.shape[-1])
                    )

                    # Compute importance scores
                    iSab_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                        self.num_samples, batch_size
                    )
                    iSab = torch.mean(iSab_sample, dim=0).detach().cpu().numpy()

                    # Clip for numerical stability
                    iSab = np.clip(iSab, -1e6, 1e6)
                    scores[:, f, time_past] = iSab

            # Compute the differences for consecutive timesteps
            scores[:, :, 1:] = scores[:, :, 1:] - scores[:, :, :-1]
            self.log.info(f"Attribution completed: Time elapsed: {(time() - tic):.4f}")
            return scores

    def forward_fill_time(self, array):
        """
        Forward fills NaN values along the time dimension of a 4D numpy array.
        Input shape: (batch, channel, time, feature)
        """
        import pandas as pd

        shape = array.shape

        # Reshape to combine batch and channel dimensions for processing
        reshaped = array.reshape(-1, shape[2], shape[3])

        # Process each feature independently
        filled = np.zeros_like(reshaped)
        for i in range(reshaped.shape[0]):  # iterate over combined batch*channel
            for j in range(reshaped.shape[2]):  # iterate over features
                filled[i, :, j] = (
                    pd.Series(reshaped[i, :, j]).fillna(method="ffill").values
                )

        # Restore original 4D shape
        return filled.reshape(shape)

    def _compute_metric(
        self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the metric for comparisons of two distributions.

        Args:
            p_y_exp:
                The current expected distribution. Shape = (batch_size, num_states)
            p_y_hat:
                The modified (counterfactual) distribution. Shape = (batch_size, num_states)

        Returns:
            The result Tensor of shape (batch_size).

        """
        if self.metric == "kl":
            return torch.sum(
                torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp), -1
            )
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            diff = torch.abs(p_y_hat - p_y_exp)
            return torch.sum(diff, -1)
        raise Exception(f"unknown metric. {self.metric}")

    def _init_generators(self):
        if self.joint:
            gen_path = self.path / "joint_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = JointFeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=self.num_features * 3,
                prediction_size=self.window_size,
                data=self.data_name,
            )
        else:
            gen_path = self.path / "feature_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = FeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=50,
                prediction_size=self.window_size,
                conditional=self.conditional,
                data=self.data_name,
            )

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults:
        self._init_generators()
        return self.generators.train_generator(train_loader, valid_loader, num_epochs)

    def test_generators(self, test_loader) -> float:
        test_loss = self.generators.test_generator(test_loader)
        self.log.info(f"Generator Test MSE Loss: {test_loss}")
        return test_loss

    def load_generators(self) -> None:
        self._init_generators()
        self.generators.load_generator()

    def get_name(self):
        builder = ["winit", "window", str(self.window_size)]
        if self.num_samples != 3:
            builder.extend(["samples", str(self.num_samples)])
        if self.conditional:
            builder.append("cond")
        if self.joint:
            builder.append("joint")
        builder.append(self.metric)
        if self.data_distribution is not None:
            builder.append("usedatadist")
        return "_".join(builder)
