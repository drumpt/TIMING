from __future__ import annotations

import logging
import pathlib

import numpy as np
import torch

from winit.explainer.explainers import BaseExplainer
from winit.explainer.generator.generator import GeneratorTrainingResults
from winit.explainer.generator.jointgenerator import JointFeatureGenerator


class FITSetCFExplainer(BaseExplainer):
    """
    The explainer for FIT. The implementation is modified from the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(
        self,
        device,
        feature_size: int,
        data_name: str,
        path: pathlib.Path,
        num_samples: int = 1,
        **kwargs,
    ):
        """
        Constructor.

        Args:
            device:
                The torch device.
            feature_size:
                The total number of features.
            data_name:
                The name of the data.
            path:
                The path where the generator state dict are saved.
            num_samples:
                The number of samples for counterfactual generations.
            **kwargs:
                There should be no additional kwargs.
        """
        super().__init__(device)
        self.generator: JointFeatureGenerator | None = None
        self.feature_size = feature_size
        self.n_samples = num_samples
        assert num_samples == 1
        self.data_name = data_name
        self.path = path
        self.log = logging.getLogger(FITSetCFExplainer.__name__)
        if len(kwargs) > 0:
            self.log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def _model_predict(self, x, mask=None, timesteps=None):
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

    def _init_generators(self):
        gen_path = self.path / "joint_generator"
        gen_path.mkdir(parents=True, exist_ok=True)
        self.generator = JointFeatureGenerator(
            self.feature_size,
            self.device,
            gen_path,
            hidden_size=self.feature_size * 3,
            data=self.data_name,
        )

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults:
        self._init_generators()
        return self.generator.train_generator(
            train_loader, valid_loader, num_epochs, lr=0.001, weight_decay=0
        )

    def test_generators(self, test_loader) -> float:
        test_loss = self.generator.test_generator(test_loader)
        self.log.info(f"Joint Generator Test MSE Loss: {test_loss}")
        return test_loss

    def load_generators(self):
        self._init_generators()
        self.generator.load_generator()

    def attribute(self, x, mask=None):
        """FIT attribution using carry-forward for counterfactuals"""
        x = x.to(self.device)
        batch_size, n_features, t_len = x.shape
        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        score = np.zeros(list(x.shape))
        for t in range(1, t_len):
            p_y_t = self._model_predict(
                x[:, :, : t + 1],
                mask[:, :, : t + 1],
                timesteps[:, : t + 1],
            )
            p_tm1 = self._model_predict(
                x[:, :, 0:t],
                mask[:, :, :t],
                timesteps[:, :t],
            )

            for i in range(n_features):
                # Instead of using generator, carry forward the previous value
                x_hat = x[:, :, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1, 1)
                prev_value = x_hat[:, :, i, t-1:t]  # Get previous value
                x_hat[:, :, i, t] = prev_value.squeeze(-1)  # Carry forward
                x_hat = x_hat.reshape(-1, n_features, t + 1)

                mask_hat = mask[:, :, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1, 1)
                mask_hat[:, :, :, t] = 0  # Values exist
                mask_hat = mask_hat.reshape(-1, n_features, t + 1)

                time_hat = timesteps[:, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1)
                time_hat = time_hat.reshape(-1, t + 1)

                y_hat_t = self._model_predict(x_hat, mask_hat, time_hat)
                y_hat_t = y_hat_t.reshape(self.n_samples, -1, y_hat_t.shape[-1])

                first_term = torch.sum(
                    torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1
                )
                p_y_t_expanded = p_y_t.unsqueeze(0).expand(self.n_samples, -1, -1)
                second_term = torch.sum(
                    torch.nn.KLDivLoss(reduction="none")(
                        torch.log(y_hat_t), p_y_t_expanded
                    ),
                    -1,
                )
                div = first_term.unsqueeze(0) - second_term
                E_div = torch.mean(div, dim=0).detach().cpu().numpy()

                score[:, i, t] = 2.0 / (1 + np.exp(-5 * E_div)) - 1

        self.log.info("attribute_carry_forward done")
        return score

    def get_name(self):
        if self.n_samples == 10:
            return "fit"

        return f"fit_sample_{self.n_samples}"
