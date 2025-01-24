import torch as th
import torch.nn as nn

from captum._utils.common import _run_forward

from typing import Callable, Union

from tint.models import Net


class ExtremalMaskNN(nn.Module):
    """
    Extremal Mask NN model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32

    References
        #. `Learning Perturbations to Explain Time Series Predictions <https://arxiv.org/abs/2305.18840>`_
        #. `Understanding Deep Networks via Extremal Perturbations and Smooth Masks <https://arxiv.org/abs/1910.08485>`_
    """

    def __init__(
        self,
        forward_func: Callable,
        model: nn.Module = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "forward_func", forward_func)
        self.model = model
        self.batch_size = batch_size

        self.input_size = None
        self.register_parameter("mask", None)

    def init(self, input_size: tuple, batch_size: int = 32) -> None:
        self.input_size = input_size
        self.batch_size = batch_size

        self.mask = nn.Parameter(th.Tensor(*input_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mask.data.fill_(0.5)

    def forward(
        self,
        x: th.Tensor,
        batch_idx,
        baselines,
        target,
        *additional_forward_args,
    ) -> (th.Tensor, th.Tensor):
        mask = self.mask

        # Subset sample to current batch
        mask = mask[
            self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)
        ]

        # We clamp the mask
        mask = mask.clamp(0, 1)

        # If model is provided, we use it as the baselines
        if self.model is not None:
            baselines = self.model(x - baselines)

        # Mask data according to samples
        # We eventually cut samples up to x time dimension
        # x1 represents inputs with important features masked.
        # x2 represents inputs with unimportant features masked.
        mask = mask[:, : x.shape[1], ...]
        x1 = x * mask + baselines * (1.0 - mask)
        x2 = x * (1.0 - mask) + baselines * mask

        # Return f(perturbed x)
        return (
            _run_forward(
                forward_func=self.forward_func,
                inputs=x1,
                target=target,
                additional_forward_args=additional_forward_args,
            ),
            _run_forward(
                forward_func=self.forward_func,
                inputs=x2,
                target=target,
                additional_forward_args=additional_forward_args,
            ),
        )

    def representation(self):
        return self.mask.detach().cpu().clamp(0, 1)


class ExtremalMaskNet(Net):
    """
    Extremal mask model as a Pytorch Lightning model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        preservation_mode (bool): If ``True``, uses the method in
            preservation mode. Otherwise, uses the deletion mode.
            Default to ``True``
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
        lambda_1 (float): Weighting for the mask loss. Default to 1.
        lambda_2 (float): Weighting for the model output loss. Default to 1.
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    Examples:
        >>> import torch as th
        >>> from tint.attr.models import ExtremalMaskNet
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> mask = ExtremalMaskNet(
        ...     forward_func=mlp,
        ...     optim="adam",
        ...     lr=0.01,
        ... )
    """

    def __init__(
        self,
        forward_func: Callable,
        preservation_mode: bool = True,
        model: nn.Module = None,
        batch_size: int = 32,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        lambda_3: float = 1.0,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        mask = ExtremalMaskNN(
            forward_func=forward_func,
            model=model,
            batch_size=batch_size,
        )

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

        self.preservation_mode = preservation_mode
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)
    
    # def pointwise_contribution(self, batch_idx, x, y_target, baselines, target, additional_forward_args):
    #     probs = self.net.mask[
    #         self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
    #     ].clamp(0, 1)
        
    #     probs = probs / probs.sum(dim=(1, 2), keepdim=True)
        
    #     B, T, D = probs.shape
        
    #     num_samples = 3
        
    #     probs_2d = probs.reshape(B, T * D)
    #     sampled_indices = th.multinomial(probs_2d, num_samples, replacement=True)
        
    #     loss = 0
        

    #     for i in range(num_samples):
    #         each_sample_indices = sampled_indices[:, i] 
    #         ts = each_sample_indices // D
    #         ds = each_sample_indices % D
            
    #         mask = th.ones_like(probs)
    #         # mask = self.net.mask[
    #         #     self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
    #         # ].clamp(0, 1)
            
    #         # mask[th.arange(B), ts, ds] = 0
            
    #         baselines = self.net.model(x - baselines)
    #         x[th.arange(B), ts, ds] = baselines[th.arange(B), ts, ds]
        
    #         x1 = x * mask + baselines * (1.0 - mask)
            
            
    #         y_hat = _run_forward(
    #             forward_func=self.net.forward_func,
    #             inputs=x1,
    #             target=target,
    #             additional_forward_args=additional_forward_args,
    #         )
            
    #         loss -= self.loss(y_hat, y_target)
            
    #     return loss / num_samples
    
    # def pointwise_contribution_y_hat(self, batch_idx, x, y_target, baselines, target, additional_forward_args):
    #     probs = self.net.mask[
    #         self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
    #     ].clamp(0, 1)
        
    #     probs = probs / probs.sum(dim=(1, 2), keepdim=True)
        
    #     B, T, D = probs.shape
        
    #     num_samples = 3
        
    #     probs_2d = probs.reshape(B, T * D)
    #     sampled_indices = th.multinomial(probs_2d, num_samples, replacement=True)
        
    #     loss = 0
        
    #     # baselines = self.net.model(x - baselines)

    #     for i in range(num_samples):
    #         each_sample_indices = sampled_indices[:, i] 
    #         ts = each_sample_indices // D
    #         ds = each_sample_indices % D
            
    #         mask = self.net.mask[
    #             self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
    #         ].clamp(0, 1)
            
    #         # mask[th.arange(B), ts, ds] = 0
            
    #         baselines = self.net.model(x - baselines)
    #         x[th.arange(B), ts, ds] = baselines[th.arange(B), ts, ds]
    #         # baselines[th.arange(B), ts, ds] = x[th.arange(B), ts, ds]
        
    #         x1 = x * mask + baselines * (1.0 - mask)
            
    #         y_hat = _run_forward(
    #             forward_func=self.net.forward_func,
    #             inputs=x1,
    #             target=target,
    #             additional_forward_args=additional_forward_args,
    #         )
            
    #         loss -= self.loss(y_hat, y_target)
            
    #     return loss / num_samples
    
    # def pointwise_contribution_y_hat_reverse(self, batch_idx, x, y_target, baselines, target, additional_forward_args):
    #     probs = self.net.mask[
    #         self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
    #     ].clamp(0, 1)
        
    #     probs = probs / probs.sum(dim=(1, 2), keepdim=True)
        
    #     B, T, D = probs.shape
        
    #     num_samples = 3
        
    #     probs_2d = probs.reshape(B, T * D)
    #     probs_2d = 1 - probs_2d
    #     sampled_indices = th.multinomial(probs_2d, num_samples, replacement=True)
        
    #     loss = 0
        
    #     for i in range(num_samples):
    #         each_sample_indices = sampled_indices[:, i] 
    #         ts = each_sample_indices // D
    #         ds = each_sample_indices % D
            
    #         mask = self.net.mask[
    #             self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
    #         ].clamp(0, 1)
            
    #         # mask[th.arange(B), ts, ds] = 1
    #         baselines = self.net.model(x - baselines)
    #         baselines[th.arange(B), ts, ds] = x[th.arange(B), ts, ds]
        
    #         x1 = x * mask + baselines * (1.0 - mask)
            
    #         y_hat = _run_forward(
    #             forward_func=self.net.forward_func,
    #             inputs=x1,
    #             target=target,
    #             additional_forward_args=additional_forward_args,
    #         )
            
    #         loss += self.loss(y_hat, y_target)
            
    #     return loss / num_samples


    def gradient_mask(self, batch_idx, x, baselines, target, additional_forward_args):
        mask = self.net.mask[
                self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
            ]
        
        mask = mask.clamp(0, 1)
        
        baselines = self.net.model(x - baselines)
        x1 = x * mask + baselines * (1.0 - mask)
            
        y_hat = _run_forward(
            forward_func=self.net.forward_func,
            inputs=x1,
            target=target,
            additional_forward_args=additional_forward_args,
        )     
        
        grad = th.autograd.grad(
            outputs=y_hat.sum(),
            inputs=x1,
            retain_graph=True,
            allow_unused=True
        )[0]  # shape: [B, T, D]
        
        
        return (grad.abs() * (1 - self.net.mask[
                self.net.batch_size * batch_idx : self.net.batch_size * (batch_idx + 1)
            ])).sum().sum()

    def step(self, batch, batch_idx, stage):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, baselines, target, *additional_forward_args = batch

        # If additional_forward_args is only one None,
        # set it to None
        if additional_forward_args == [None]:
            additional_forward_args = None

        # Get perturbed output
        # y_hat1 is computed by masking important features
        # y_hat2 is computed by masking unimportant features
        if additional_forward_args is None:
            y_hat1, y_hat2 = self(x.float(), batch_idx, baselines, target)
        else:
            y_hat1, y_hat2 = self(
                x.float(),
                batch_idx,
                baselines,
                target,
                *additional_forward_args,
            )

        # Get unperturbed output for inputs and baselines
        y_target1 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )
        y_target2 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=th.zeros_like(y) + baselines,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )
                    
        # loss = self.lambda_3 * self.pointwise_contribution(batch_idx, x.float(), y_target1, baselines, target, additional_forward_args)
        # loss = self.lambda_3 * self.pointwise_contribution_y_hat(batch_idx, x.float(), y_hat1, baselines, target, additional_forward_args)
        # loss = self.lambda_3 * self.pointwise_contribution_y_hat_reverse(batch_idx, x.float(), y_hat1, baselines, target, additional_forward_args)
        
        # loss = self.lambda_3 * self.pointwise_contribution_y_hat(batch_idx, x.float(), y_hat1, baselines, target, additional_forward_args)
        # loss += self.lambda_3 * self.pointwise_contribution_y_hat_reverse(batch_idx, x.float(), y_hat1, baselines, target, additional_forward_args)
        loss = self.lambda_3 * self.gradient_mask(batch_idx, x.float(), baselines, target, additional_forward_args)
        
        
        # Add L1 loss
        # if self.preservation_mode:
        #     mask_ = self.lambda_1 * self.net.mask.abs()
        # else:
        #     mask_ = self.lambda_1 * (1.0 - self.net.mask).abs()
        
        if self.preservation_mode:
            mask_ = self.lambda_1 * self.net.mask.abs()
        else:
            mask_ = self.lambda_1 * (1.0 - self.net.mask).abs()

        if self.net.model is not None:
            mask_ = mask_[
                self.net.batch_size
                * batch_idx : self.net.batch_size
                * (batch_idx + 1)
            ]
            mask_ += self.lambda_2 * self.net.model(x - baselines).abs()
        loss += mask_.mean()

        # Add preservation and deletion losses if required
        if self.preservation_mode:
            loss += 0.0001 * self.loss(y_hat1, y_target1)
        else:
            loss += 0.0001 * self.loss(y_hat2, y_target2)


        return loss

    def configure_optimizers(self):
        params = [{"params": self.net.mask}]

        if self.net.model is not None:
            params += [{"params": self.net.model.parameters()}]

        if self._optim == "adam":
            optim = th.optim.Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}