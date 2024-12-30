from __future__ import annotations

import abc
import logging
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, DeepLift, GradientShap

from winit.explainer.generator.generator import GeneratorTrainingResults
from winit.models import TorchModel
from winit.utils import resolve_device
from winit.explainer.generator.generator import (
    FeatureGenerator,
    BaseFeatureGenerator,
    GeneratorTrainingResults,
)
from winit.explainer.generator.jointgenerator import JointFeatureGenerator


class BaseExplainer(abc.ABC):
    """
    A base class for explainer.
    """

    def __init__(self, device=None, args=None):
        """
        Constructor.

        Args:
            device:
               The torch device.
        """
        self.base_model: TorchModel | None = None
        self.device = resolve_device(device)
        self.args = args

    @abc.abstractmethod
    def attribute(self, x, mask):
        """
        The attribution method that the explainer will give.
        Args:
            x:
                The input tensor.

        Returns:
            The attribution with respect to x. The shape should be the same as x, or it could
            be one dimension greater than x if there is aggregation needed.

        """

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults | None:
        """
        If the explainer or attribution method needs a generator, this will train the generator.

        Args:
            train_loader:
                The dataloader for training
            valid_loader:
                The dataloader for validation.
            num_epochs:
                The number of epochs.

        Returns:
            The training results for the generator, if applicable. This includes the
            training curves.

        """
        return None

    def test_generators(self, test_loader) -> float | None:
        """
        If the explainer or attribution method needs a generator, this will return the performance
        of the generator on the test set.

        Args:
            test_loader:
                The dataloader for testing.

        Returns:
            The test result (MSE) for the generator, if applicable.

        """
        return None

    def load_generators(self) -> None:
        """
        If the explainer or attribution method needs a generator, this will load the generator from
        the disk.
        """

    def set_model(self, model, set_eval=True) -> None:
        """
        Set the base model the explainer wish to explain.

        Args:
            model:
                The base model.
            set_eval:
                Indicating whether we set to eval mode for the explainer. Note that in some cases
                like Dynamask or FIT, they do not set the model to eval mode.
        """
        self.base_model = model
        if set_eval:
            self.base_model.eval()
        self.base_model.to(self.device)

    @abc.abstractmethod
    def get_name(self):
        """
        Return the name of the explainer.
        """


class MockExplainer(BaseExplainer):
    """
    Class for mock explainer. The mock explainer returns all the attributes to 0.
    """

    def __init__(self):
        super().__init__()

    def attribute(self, x, mask, **kwargs):
        return np.zeros(x.shape)

    def get_name(self):
        return "mock"


class DeepLiftExplainer(BaseExplainer):
    """
    The explainer for the DeepLIFT method using zeros as the baseline and captum for the
    implementation.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model)
        self.explainer = DeepLift(self.base_model)

    def attribute(self, x, mask):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        assert (
            self.base_model.num_states == 1
        ), "TODO: Implement retrospective for > 1 class"
        score = self.explainer.attribute(
            x, baselines=(x * 0), additional_forward_args=(mask, None, False)
        )
        score = abs(score.detach().cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "deeplift"


class IGExplainer(BaseExplainer):
    """
    The explainer for integrated gradients using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model, set_eval=set_eval)
        self.explainer = IntegratedGradients(self.base_model)

    def attribute(self, x, mask):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        score = self.explainer.attribute(
            x, baselines=(x * 0), additional_forward_args=(mask, None, False)
        )
        score = np.abs(score.detach().cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "ig"


class GradientShapExplainer(BaseExplainer):
    """
    The explainer for gradient shap using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model, set_eval=set_eval)
        self.explainer = GradientShap(self.base_model)

    def attribute(self, x, mask):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x = x.to(self.device)
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        score = self.explainer.attribute(
            x,
            n_samples=50,
            stdevs=0.0001,
            baselines=(torch.cat([x * 0, x * 1])),
            additional_forward_args=(mask, None, False),
        )
        score = abs(score.cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "gradientshap"


class FOExplainer(BaseExplainer):
    """
    The explainer for feature occlusion. The implementation is simplified from the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, device, n_samples=10, **kwargs):
        super().__init__(device)
        self.n_samples = n_samples
        if len(kwargs) > 0:
            log = logging.getLogger(FOExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x, mask):
        self.base_model.eval()
        self.base_model.zero_grad()

        x = x.to(self.device)
        batch_size, n_features, t_len = x.shape
        score = np.zeros(x.shape)

        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        for t in range(1, t_len):
            p_y_t = self.base_model.predict(
                x[:, :, : t + 1],
                mask[:, :, : t + 1],
                timesteps[:, : t + 1],
                return_all=False,
            )
            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                mask_hat = mask[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(self.n_samples):
                    x_hat[:, i, t] = torch.Tensor(
                        np.random.uniform(-3, +3, size=(len(x),))
                    )
                    mask_hat[:, i, t] = 1  # Value exists
                    y_hat_t = self.base_model.predict(
                        x_hat,
                        mask_hat,
                        timesteps[:, : t + 1],
                        return_all=False,
                    )
                    kl = torch.abs(y_hat_t - p_y_t)
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl
        return score

    def get_name(self):
        if self.n_samples != 10:
            return f"fo_sample_{self.n_samples}"
        return "fo"


class FOZeroExplainer(BaseExplainer):
    """
    The explainer for feature occlusion. The implementation is simplified from the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, device, n_samples=1, **kwargs):
        super().__init__(device)
        self.n_samples = n_samples
        if len(kwargs) > 0:
            log = logging.getLogger(FOZeroExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x, mask):
        self.base_model.eval()
        self.base_model.zero_grad()

        x = x.to(self.device)
        batch_size, n_features, t_len = x.shape
        score = np.zeros(x.shape)

        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        for t in range(1, t_len):
            p_y_t = self.base_model.predict(
                x[:, :, : t + 1],
                mask[:, :, : t + 1],
                timesteps[:, : t + 1],
                return_all=False,
            )
            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                mask_hat = mask[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(self.n_samples):
                    x_hat[:, i, t] = torch.zeros_like(x_hat[:, i, t])
                    mask_hat[:, i, t] = 0  # Value doesn't exist
                    y_hat_t = self.base_model.predict(
                        x_hat,
                        mask_hat,
                        timesteps[:, : t + 1],
                        return_all=False,
                    )
                    kl = torch.abs(y_hat_t - p_y_t)
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl
        return score

    def get_name(self):
        if self.n_samples != 10:
            return f"fozero_sample_{self.n_samples}"
        return "fozero"


class AFOExplainer(BaseExplainer):
    """
    The explainer for augmented feature occlusion. The implementation is simplified from
    the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, device, train_loader, n_samples=10, **kwargs):
        super().__init__(device)
        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])
        self.n_samples = n_samples
        if len(kwargs) > 0:
            log = logging.getLogger(AFOExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x, mask):
        self.base_model.eval()
        self.base_model.zero_grad()

        x = x.to(self.device)
        batch_size, n_features, t_len = x.shape
        score = np.zeros(x.shape)

        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        for t in range(1, t_len):
            p_y_t = self.base_model.predict(
                x[:, :, : t + 1],
                mask[:, :, : t + 1],
                timesteps[:, : t + 1],
                return_all=False,
            )
            for i in range(n_features):
                feature_dist = np.array(self.data_distribution[:, i, :]).reshape(-1)
                x_hat = x[:, :, 0 : t + 1].clone()
                mask_hat = mask[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(self.n_samples):
                    x_hat[:, i, t] = torch.Tensor(
                        np.random.choice(feature_dist, size=(len(x),))
                    ).to(self.device)
                    mask_hat[:, i, t] = 1  # Value exists
                    y_hat_t = self.base_model.predict(
                        x_hat,
                        mask_hat,
                        timesteps[:, : t + 1],
                        return_all=False,
                    )
                    kl = torch.abs((y_hat_t[:, :]) - (p_y_t[:, :]))
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl
        return score

    def get_name(self):
        if self.n_samples != 10:
            return f"afo_sample_{self.n_samples}"
        return "afo"


class AFOGenExplainer(BaseExplainer):
    """
    The explainer for augmented feature occlusion. The implementation is simplified from
    the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: DataLoader | None = None,
        num_samples: int = 10,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        random_state: int | None = None,
        args=None,
        **kwargs,
    ):
        super().__init__(device)

        self.n_samples = num_samples
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

        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])
        if len(kwargs) > 0:
            log = logging.getLogger(AFOGenExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x, mask):
        self.base_model.eval()
        self.base_model.zero_grad()

        x = x.to(self.device)
        batch_size, n_features, t_len = x.shape
        score = np.zeros(x.shape)

        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        for t in range(1, t_len):
            p_y_t = self.base_model.predict(
                x[:, :, : t + 1],
                mask[:, :, : t + 1],
                timesteps[:, : t + 1],
                return_all=False,
            )

            # Generate counterfactuals for all features at time t
            counterfactuals = self._generate_counterfactuals(
                1,  # time_forward is 1 since we're only replacing current timestep
                x[:, :, :t],  # past data up to t
                x[:, :, t : t + 1],  # current timestep data
            )

            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                mask_hat = mask[:, :, 0 : t + 1].clone()
                kl_all = []

                # Use generated counterfactuals instead of random sampling
                for s in range(self.n_samples):
                    x_hat[:, i, t] = counterfactuals[
                        i, s, :, 0
                    ]  # Use generated counterfactual
                    mask_hat[:, i, t] = 1  # Value exists

                    y_hat_t = self.base_model.predict(
                        x_hat,
                        mask_hat,
                        timesteps[:, : t + 1],
                        return_all=False,
                    )
                    kl = torch.abs((y_hat_t[:, :]) - (p_y_t[:, :]))
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))

                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl
        return score

    def _generate_counterfactuals(
        self, time_forward: int, x_in: torch.Tensor, x_current: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate the counterfactuals.

        Args:
            time_forward:
                Number of timesteps of counterfactuals we wish to generate.
            x_in:
                The past Tensor. Shape = (batch_size, num_features, num_times)
            x_current:
                The current Tensor if a conditional generator is used.
                Shape = (batch_size, num_features, time_forward). If the generator is not
                conditional, x_current is None.

        Returns:
            Counterfactual of shape (num_features, num_samples, batch_size, time_forward)

        """
        # x_in shape (bs, num_feature, num_time)
        # x_current shape (bs, num_feature, time_forward)
        # return counterfactuals shape (num_feature, num_samples, batchsize, time_forward)
        batch_size, _, num_time = x_in.shape
        if self.data_distribution is not None:
            # Random sample instead of using generator
            counterfactuals = torch.zeros(
                (self.num_features, self.n_samples, batch_size, time_forward),
                device=self.device,
            )
            for f in range(self.num_features):
                values = self.data_distribution[:, f, :].reshape(-1)
                counterfactuals[f, :, :, :] = torch.tensor(
                    self.rng.choice(
                        values, size=(self.n_samples, batch_size, time_forward)
                    ),
                    device=self.device,
                )
            return counterfactuals

        if isinstance(self.generators, FeatureGenerator):
            mu, std = self.generators.forward(x_current, x_in, deterministic=True)
            mu = mu[:, :, :time_forward]
            std = std[:, :, :time_forward]  # (bs, f, time_forward)
            counterfactuals = mu.unsqueeze(0) + torch.randn(
                self.n_samples,
                batch_size,
                self.num_features,
                time_forward,
                device=self.device,
            ) * std.unsqueeze(0)
            return counterfactuals.permute(2, 0, 1, 3)

        if isinstance(self.generators, JointFeatureGenerator):
            counterfactuals = torch.zeros(
                (self.num_features, self.n_samples, batch_size, time_forward),
                device=self.device,
            )
            for f in range(self.num_features):
                mu_z, std_z = self.generators.get_z_mu_std(x_in)
                gen_out, _ = (
                    self.generators.forward_conditional_multisample_from_z_mu_std(
                        x_in,
                        x_current,
                        list(set(range(self.num_features)) - {f}),
                        mu_z,
                        std_z,
                        self.n_samples,
                    )
                )
                # gen_out shape (ns, bs, num_feature, time_forward)
                counterfactuals[f, :, :, :] = gen_out[:, :, f, :]
            return counterfactuals

        raise ValueError("Unknown generator or no data distribution provided.")

    def get_name(self):
        if self.n_samples != 10:
            return f"afogen_sample_{self.n_samples}"
        return "afogen"


class AFOEnsembleExplainer(BaseExplainer):
    """
    The explainer for augmented feature occlusion. The implementation is simplified from
    the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: DataLoader | None = None,
        num_samples: int = 10,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        random_state: int | None = None,
        args=None,
        **kwargs,
    ):
        super().__init__(device)

        self.n_samples = num_samples
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

        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])
        if len(kwargs) > 0:
            log = logging.getLogger(AFOGenExplainer.__name__)
            log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def attribute(self, x, mask):
        """
        Compute attributions using ensemble of:
        1. Zero replacement
        2. Previous value copying 
        3. Generator-based counterfactuals
        """
        self.base_model.eval()
        self.base_model.zero_grad()

        x = x.to(self.device)
        batch_size, n_features, t_len = x.shape
        scores = np.zeros((3, batch_size, n_features, t_len)) # Store scores from each method

        timesteps = torch.linspace(0, 1, t_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        for t in range(1, t_len):
            # Get original prediction
            p_y_t = self.base_model.predict(
                x[:, :, : t + 1],
                mask[:, :, : t + 1], 
                timesteps[:, : t + 1],
                return_all=False,
            )

            # Generate counterfactuals for generator-based method
            counterfactuals = self._generate_counterfactuals(
                1,
                x[:, :, :t],
                x[:, :, t : t + 1],
            )

            for i in range(n_features):
                # Method 1: Zero replacement
                x_hat_zero = x[:, :, : t + 1].clone()
                mask_hat_zero = mask[:, :, : t + 1].clone()
                x_hat_zero[:, i, t] = torch.zeros_like(x_hat_zero[:, i, t])
                mask_hat_zero[:, i, t] = 0
                y_hat_zero = self.base_model.predict(
                    x_hat_zero,
                    mask_hat_zero,
                    timesteps[:, : t + 1],
                    return_all=False,
                )
                kl_zero = torch.abs(y_hat_zero - p_y_t)
                scores[0, :, i, t] = np.mean(kl_zero.detach().cpu().numpy(), -1)

                # Method 2: Previous value copying
                x_hat_prev = x[:, :, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1, 1)
                prev_value = x_hat_prev[:, :, i, t - 1 : t]
                x_hat_prev[:, :, i, t] = prev_value.squeeze(-1)
                x_hat_prev = x_hat_prev.reshape(-1, n_features, t + 1)
                
                mask_hat_prev = mask[:, :, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1, 1)
                mask_hat_prev = mask_hat_prev.reshape(-1, n_features, t + 1)
                
                timesteps_prev = timesteps[:, : t + 1].unsqueeze(0).repeat(self.n_samples, 1, 1)
                timesteps_prev = timesteps_prev.reshape(-1, t + 1)
                
                y_hat_prev = self.base_model.predict(
                    x_hat_prev, 
                    mask_hat_prev,
                    timesteps_prev,
                    return_all=False
                )
                y_hat_prev = y_hat_prev.reshape(self.n_samples, -1, y_hat_prev.shape[-1])
                p_y_t_expanded = p_y_t.unsqueeze(0).expand(self.n_samples, -1, -1)
                
                kl_prev = torch.abs(y_hat_prev - p_y_t_expanded)
                scores[1, :, i, t] = torch.mean(kl_prev, dim=0).detach().cpu().numpy().mean(-1)

                # Method 3: Generator-based
                x_hat_gen = x[:, :, : t + 1].clone()
                mask_hat_gen = mask[:, :, : t + 1].clone()
                kl_all_gen = []
                
                for s in range(self.n_samples):
                    x_hat_gen[:, i, t] = counterfactuals[i, s, :, 0]
                    mask_hat_gen[:, i, t] = 1
                    
                    y_hat_gen = self.base_model.predict(
                        x_hat_gen,
                        mask_hat_gen, 
                        timesteps[:, : t + 1],
                        return_all=False,
                    )
                    kl = torch.abs(y_hat_gen - p_y_t)
                    kl_all_gen.append(np.mean(kl.detach().cpu().numpy(), -1))
                    
                scores[2, :, i, t] = np.mean(np.array(kl_all_gen), axis=0)

        # Take maximum scores across methods
        final_scores = np.max(scores, axis=0)
        return final_scores

    def _generate_counterfactuals(
        self, time_forward: int, x_in: torch.Tensor, x_current: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate the counterfactuals.

        Args:
            time_forward:
                Number of timesteps of counterfactuals we wish to generate.
            x_in:
                The past Tensor. Shape = (batch_size, num_features, num_times)
            x_current:
                The current Tensor if a conditional generator is used.
                Shape = (batch_size, num_features, time_forward). If the generator is not
                conditional, x_current is None.

        Returns:
            Counterfactual of shape (num_features, num_samples, batch_size, time_forward)

        """
        # x_in shape (bs, num_feature, num_time)
        # x_current shape (bs, num_feature, time_forward)
        # return counterfactuals shape (num_feature, num_samples, batchsize, time_forward)
        batch_size, _, num_time = x_in.shape
        if self.data_distribution is not None:
            # Random sample instead of using generator
            counterfactuals = torch.zeros(
                (self.num_features, self.n_samples, batch_size, time_forward),
                device=self.device,
            )
            for f in range(self.num_features):
                values = self.data_distribution[:, f, :].reshape(-1)
                counterfactuals[f, :, :, :] = torch.tensor(
                    self.rng.choice(
                        values, size=(self.n_samples, batch_size, time_forward)
                    ),
                    device=self.device,
                )
            return counterfactuals

        if isinstance(self.generators, FeatureGenerator):
            mu, std = self.generators.forward(x_current, x_in, deterministic=True)
            mu = mu[:, :, :time_forward]
            std = std[:, :, :time_forward]  # (bs, f, time_forward)
            counterfactuals = mu.unsqueeze(0) + torch.randn(
                self.n_samples,
                batch_size,
                self.num_features,
                time_forward,
                device=self.device,
            ) * std.unsqueeze(0)
            return counterfactuals.permute(2, 0, 1, 3)

        if isinstance(self.generators, JointFeatureGenerator):
            counterfactuals = torch.zeros(
                (self.num_features, self.n_samples, batch_size, time_forward),
                device=self.device,
            )
            for f in range(self.num_features):
                mu_z, std_z = self.generators.get_z_mu_std(x_in)
                gen_out, _ = (
                    self.generators.forward_conditional_multisample_from_z_mu_std(
                        x_in,
                        x_current,
                        list(set(range(self.num_features)) - {f}),
                        mu_z,
                        std_z,
                        self.n_samples,
                    )
                )
                # gen_out shape (ns, bs, num_feature, time_forward)
                counterfactuals[f, :, :, :] = gen_out[:, :, f, :]
            return counterfactuals

        raise ValueError("Unknown generator or no data distribution provided.")

    def get_name(self):
        if self.n_samples != 10:
            return f"afoensemble_sample_{self.n_samples}"
        return "afoensemble"
