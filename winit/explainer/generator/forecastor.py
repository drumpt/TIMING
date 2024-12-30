from torch.utils.data import DataLoader

import abc
import dataclasses
import logging
import pathlib
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from winit.explainer.generator.generator import (
    BaseFeatureGenerator,
    GeneratorTrainingResults,
)

class LinearForecaster(nn.Module):
    """
    A simple linear layer forecaster.
    Transforms (..., hidden_size) to (..., num_features).
    """
    def __init__(self, hidden_size, num_features):
        super(LinearForecaster, self).__init__()
        self.linear = nn.Linear(hidden_size, num_features)
    
    def forward(self, x):
        return self.linear(x)


class MLPForecaster(nn.Module):
    """
    A multi-layer perceptron (MLP) forecaster.
    Transforms (..., hidden_size) to (..., num_features).
    """
    def __init__(self, hidden_size, num_features, hidden_units=128, num_layers=2):
        super(MLPForecaster, self).__init__()
        layers = []
        input_dim = hidden_size
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.ReLU())
            input_dim = hidden_units
        layers.append(nn.Linear(hidden_units, num_features))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class Forecastor(BaseFeatureGenerator):
    def __init__(
        self,
        base_model,
        forecastor,
        feature_size: int,
        device,
        gen_path: pathlib.Path,
        latent_size: int = 128,
        num_layers: int = 2,
        prediction_size: int = 1,
        data: str = "mimic",
    ):
        super().__init__(feature_size, device, prediction_size, gen_path)
        self.base_model = base_model
        hidden_size = self.base_model.encoded_hidden_size
        self.forecastor_name = forecastor
        if forecastor == "linear":
            self.forecastor = LinearForecaster(hidden_size, feature_size)
        elif forecastor == "mlp":
            self.forecastor = MLPForecaster(hidden_size, feature_size, hidden_units=latent_size, num_layers=num_layers)
        
        self.log = logging.getLogger(Forecastor.__name__)

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        num: int,
        run_train: bool = True,
        optimizer: Optimizer = None,
    ) -> float:
        if run_train:
            self.train()
            if optimizer is None:
                raise ValueError("optimizer is none in train mode.")
        else:
            self.eval()
        epoch_loss = 0
        
        loss_criterion = torch.nn.MSELoss()
            
        for i, batch in enumerate(dataloader):
            signals = torch.Tensor(batch[0].float()).to(self.device)

            masks = (
                torch.Tensor(batch[2].float()).to(self.device)
                if len(batch) > 2
                else None
            )
            
            if run_train:
                optimizer.zero_grad()
            # label: (num_samples, num_features, num_times - 1)
            label = signals.clone()[:, :, 1:]
            output = self.forecastor(self.base_model.encoding(signals, masks, return_all=True))
            # output: (num_samples, num_times, num_features)
            output = output[:, :-1, :].permute(0, 2, 1)
            loss = loss_criterion(output.reshape(output.shape[0], -1), label.reshape(label.shape[0], -1))
            epoch_loss += loss.item()
            if run_train:
                loss.backward()
                optimizer.step()
            
        return float(epoch_loss) / len(dataloader)

    def train_generator(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs,
        **kwargs,
    ) -> GeneratorTrainingResults:
        
        tic = time()
        self.to(self.device)

        # Overwrite default learning parameters if values are passed
        for p in self.base_model.parameters():
            p.requires_grad = False
        default_params = {"lr": 0.001, "weight_decay": 0}
        for k, v in kwargs.items():
            if k in default_params.keys():
                default_params[k] = v

        parameters = self.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=default_params["lr"],
            weight_decay=default_params["weight_decay"],
        )

        best_loss = 1000000
        best_epoch = -1

        train_loss_trends = np.zeros((1, num_epochs + 1))
        valid_loss_trends = np.zeros((1, num_epochs + 1))
        best_epochs = np.zeros(1, dtype=int)

        for epoch in range(num_epochs + 1):
            self._run_one_epoch(train_loader, 3, True, optimizer)
            train_loss = self._run_one_epoch(
                train_loader, 0, False, None
            )
            valid_loss = self._run_one_epoch(
                valid_loader, 0, False, None
            )
            train_loss_trends[0, epoch] = train_loss
            valid_loss_trends[0, epoch] = valid_loss

            if self.verbose and epoch % self.verbose_eval == 0:
                self.log.info(f"\nEpoch {epoch}")
                self.log.info(f"Generator Training Loss   ===> {train_loss}")
                self.log.info(f"Generator Validation Loss ===> {valid_loss}")

            if self.early_stopping:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    torch.save(self.state_dict(), str(self._get_model_file_name()))
                    if self.verbose:
                        self.log.info(f"save ckpt:in epoch {epoch}")
        best_epochs[0] = best_epoch

        self.log.info(
            f"Joint generator test loss = {best_loss:.6f}   Time elapsed: {time() - tic}"
        )
        return GeneratorTrainingResults(
            self.get_name(), train_loss_trends, valid_loss_trends, best_epochs
        )
    
    def test_generator(self, test_loader) -> float:
        """
        Test the generator

        Args:
            test_loader:
                The test data

        Returns:
            The test MSE result for the generator.

        """
        self.to(self.device)
        test_loss = self._run_one_epoch(test_loader, 10, False, None)
        return test_loss

    def _get_model_file_name(self) -> pathlib.Path:
        self.gen_path.mkdir(parents=True, exist_ok=True)
        return self.gen_path / f"{self.forecastor_name}_len_{self.prediction_size}.pt"

    def load_generator(self):
        """
        Load the generator from the disk.
        """
        self.load_state_dict(torch.load(str(self._get_model_file_name())))
        self.to(self.device)
        
    @staticmethod
    def get_name() -> str:
        return "forecast_generator"