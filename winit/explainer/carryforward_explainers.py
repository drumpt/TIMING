from __future__ import annotations

import abc
import logging

import numpy as np
import torch

from winit.explainer.generator.generator import GeneratorTrainingResults
from winit.models import TorchModel
from winit.utils import resolve_device

from captum.attr import IntegratedGradients, DeepLift



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
    def attribute(self, x, mask=None):
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


def gradient_shap(model, inputs, baselines, n_samples=50):
    """
    Compute GradientSHAP attributions for a time series model.

    Args:
        model (callable): The model function. Should take inputs and return predictions.
        inputs (torch.Tensor): The input tensor for which attributions are calculated (batch_size, features, time_steps).
        baselines (torch.Tensor): The baseline tensor corresponding to the previous values (batch_size, features, time_steps).
        n_samples (int): Number of noisy samples to average over for SHAP calculations.

    Returns:
        attributions (torch.Tensor): The computed attributions (batch_size, features, time_steps).
    """
    if inputs.shape != baselines.shape:
        raise ValueError("Inputs and baselines must have the same shape.")

    # Add noise to interpolate between baseline and input
    alphas = torch.linspace(0, 1, n_samples).view(-1, 1, 1, 1).to(inputs.device)
    noisy_inputs = baselines.unsqueeze(0) + alphas * (inputs.unsqueeze(0) - baselines.unsqueeze(0))

    # Add noise
    noise = torch.randn_like(noisy_inputs) * 0.0001
    noisy_inputs = noisy_inputs + noise

    # Compute gradients for each noisy input
    noisy_inputs.requires_grad = True
    
    ### only last time point prediction
    predictions = model.predict(noisy_inputs.view(-1, inputs.shape[1], inputs.shape[2]), return_all=False)
    
    ### all y_t
    # predictions = model.predict(noisy_inputs.view(-1, inputs.shape[1], inputs.shape[2]))
    # print(predictions.shape)
    # raise RuntimeError
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(1)

    predictions = predictions.view(n_samples, inputs.shape[0], -1)

    grad = torch.autograd.grad(
            outputs=predictions[:, :, :].sum(),
            inputs=noisy_inputs,
            retain_graph=True
        )
    grads = grad[0].mean(dim=0)
    # grads = []
    # for i in range(predictions.size(-1)):   
    #     grad = torch.autograd.grad(
    #         outputs=predictions[:, :, i].sum(),
    #         inputs=noisy_inputs,
    #         retain_graph=True
    #     )
    #     grads.append(torch.abs(grad[0]).detach())

    # grads = torch.stack(grads, dim=-1).mean(dim=-1)  # (n_samples, batch_size, features, time_steps, targets)
    # grads = grads.mean(dim=0)  # Average over noisy samples

    # Compute attributions by multiplying gradients by the difference between input and baseline
    attributions = grads * (inputs - baselines)
    return attributions

class GradientShapCFExplainer(BaseExplainer):
    """
    The explainer for gradient shap using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, device):
        super().__init__(device)
        self.explainer = None

    def set_model(self, model, set_eval=True):
        super().set_model(model, set_eval=set_eval)
        # self.explainer = GradientShap(self.base_model)

    def attribute(self, x, mask=None):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x = x.to(self.device)
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        baselines = torch.zeros_like(x).to(self.device)
        baselines[:, :, 1:] = x[:, :, :-1]
        score = gradient_shap(self.base_model, x, baselines, n_samples=50)
        # score = self.explainer.attribute(
        #     x, n_samples=50, stdevs=0.0001, baselines=baselines, additional_forward_args=(False)
        # )
        
        score = abs(score.cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "gradientshap_carryforward"


class DeepLiftCfExplainer(BaseExplainer):
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

    def attribute(self, x, mask=None):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        assert self.base_model.num_states == 1, "TODO: Implement retrospective for > 1 class"
        baselines = torch.zeros_like(x).to(self.device)
        baselines[:, :, 1:] = x[:, :, :-1]
        score = self.explainer.attribute(x, baselines=baselines, additional_forward_args=(False))
        score = abs(score.detach().cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "deeplift_carryforward"
    
    
class IGCFExplainer(BaseExplainer):
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

    def attribute(self, x, mask=None):
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        baselines = torch.zeros_like(x).to(self.device)
        baselines[:, :, 1:] = x[:, :, :-1]
        score = self.explainer.attribute(x, baselines=baselines, additional_forward_args=(False))
        score = np.abs(score.detach().cpu().numpy())

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score

    def get_name(self):
        return "ig_carryforward"