import sys
from os import path
from pathlib import Path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))


import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os
from utils.tools import print_results

from attribution.extremal_mask import ExtremalMask
from attribution.extremalmasknn import *
from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from attribution.motif import ShapeletExplainer
from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime, KernelShap, DeepLiftShap
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    DynaMask,
    # ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    Occlusion,
    FeatureAblation,
    TimeForwardTunnel,
)
from tint.attr.models import (
    # ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
# from tint.datasets import Mimic3
from datasets.mimic3 import Mimic3
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)

from mortality.cumulative_difference import cumulative_difference
from tint.models import MLP, RNN

from mortality.classifier import MimicClassifierNet


def main(
    explainers: List[str],
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    is_train: bool = True,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    output_file: str = "results.csv",
    model_type: str = "state",
    testbs: int = 0,
    top: int = 50,
    skip_train_motif: bool = True,
    skip_train_timex: bool = True,
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    mimic3 = Mimic3(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = MimicClassifierNet(
        feature_size=31,
        n_state=2,
        n_timesteps=48,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
        model_type=model_type
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(
            save_dir=".",
            version=random.getrandbits(128),
        ),
    )
    if is_train:
        trainer.fit(classifier, datamodule=mimic3)
        if not os.path.exists("./model/"):
            os.makedirs("./model/")
        th.save(classifier.state_dict(), "./model/{}_classifier_{}_{}".format(model_type, fold, seed))
    else:
        classifier.load_state_dict(th.load("./model/{}_classifier_{}_{}".format(model_type, fold, seed)))
    # Get data for explainers
    with lock:
        x_train = mimic3.preprocess(split="train")["x"].to(device)
        x_test = mimic3.preprocess(split="test")["x"].to(device)
        y_train = mimic3.preprocess(split="train")["y"].to(device)
        y_test = mimic3.preprocess(split="test")["y"].to(device)
        mask_train = mimic3.preprocess(split="train")["mask"].to(device)
        mask_test = mimic3.preprocess(split="test")["mask"].to(device)

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()
    
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(x_test, mask_test)
    test_loader = DataLoader(test_dataset, batch_size=testbs, shuffle=False)
    
    if model_type == "state":
        temporal_additional_forward_args = (False, False, False)
    else:
        temporal_additional_forward_args = (True, True, False)
    
    data_mask=mask_test
    data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )
        

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            additional_forward_args=(data_mask, timesteps, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            task="binary",
            show_progress=True,
        ).abs()
        
    # if "deep_lift_shap" in explainers:
    #     explainer = DeepLiftShap(classifier)
    #     attr["deep_lift"] = explainer.attribute(
    #         x_test,
    #         baselines=x_test * 0,
    #         additional_forward_args=(data_mask, timesteps, False),
    #         # temporal_additional_forward_args=temporal_additional_forward_args,
    #         target=1,
    #         # task="binary",
    #     ).abs()
    
    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = MaskNet(
            forward_func=classifier,
            perturbation="fade_moving_average",
            keep_ratio=list(np.arange(0.1, 0.7, 0.1)),
            deletion_mode=True,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            time_reg_factor=0.0,
            loss="cross_entropy",
        )
        explainer = DynaMask(classifier)
        _attr = explainer.attribute(
            x_test,
            trainer=trainer,
            mask_net=mask,
            additional_forward_args=(data_mask, timesteps, False),
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

    if "gate_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = GateMaskNet(
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            loss="cross_entropy",
            optim="adam",
            lr = 0.01,
            # lr=0.01,
        )
        explainer = GateMask(classifier)
        _attr = explainer.attribute(
            x_test,
            # additional_forward_args=(True,) (return_all = True) is it really considered?
            additional_forward_args=(data_mask, timesteps, False),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.5,
        )
        attr["gate_mask"] = _attr.to(device)

    # if "extremal_mask" in explainers:
    #     trainer = Trainer(
    #         max_epochs=500,
    #         accelerator=accelerator,
    #         devices=device_id,
    #         log_every_n_steps=2,
    #         deterministic=deterministic,
    #         logger=TensorBoardLogger(
    #             save_dir=".",
    #             version=random.getrandbits(128),
    #         ),
    #     )
    #     mask = ExtremalMaskNet(
    #         forward_func=classifier.predict,
    #         model=nn.Sequential(
    #             RNN(
    #                 input_size=x_test.shape[-1],
    #                 rnn="gru",
    #                 hidden_size=x_test.shape[-1],
    #                 bidirectional=True,
    #             ),
    #             MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
    #         ),
    #         lambda_1=lambda_1,
    #         lambda_2=lambda_2,
    #         loss="cross_entropy",
    #         optim="adam",
    #         lr=0.001,
    #     )
    #     explainer = ExtremalMask(classifier)
    #     _attr = explainer.attribute(
    #         x_test,
    #         additional_forward_args=(data_mask, timesteps, False),
    #         trainer=trainer,
    #         mask_net=mask,
    #         batch_size=100,
    #     )
    #     attr["extremal_mask"] = _attr.to(device)
        
    if "extremal_mask_develop" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = ExtremalMaskNet(
            forward_func=classifier.predict,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            loss="cross_entropy",
            optim="adam",
            preservation_mode=True,
            lr=0.001,
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(data_mask, timesteps, False),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask_develop"] = _attr.to(device)

    if "fit" in explainers:
        generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
        trainer = Trainer(
            max_epochs=200,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=10,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        explainer = Fit(
            classifier,
            generator=generator,
            datamodule=mimic3,
            trainer=trainer,
            features=x_train,
        )
        attr["fit"] = explainer.attribute(x_test, additional_forward_args=(data_mask, timesteps, False), show_progress=True)

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier.cpu()))
        attr["gradient_shap"] = explainer.attribute(
            x_test.cpu(),
            baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
            n_samples=50,
            stdevs=0.0001,
            additional_forward_args=(data_mask, timesteps, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            task="binary",
            show_progress=True,
        ).abs()
        classifier.to(device)

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier.predict))

        integrated_gradients = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            # Calculate Integrated Gradients for the current batch
            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                additional_forward_args=(data_mask, timesteps, False),
                temporal_additional_forward_args=temporal_additional_forward_args,
                task="binary",
                show_progress=False  # Disable progress bar for individual batches
            ).abs()
            
            # Append the IG attributes of the current batch to the list
            integrated_gradients.append(attr_batch.cpu())  # Move to CPU if necessary
        
        # Concatenate all batch IG attributes into a single tensor
        attr["integrated_gradients"] = th.cat(integrated_gradients, dim=0)

        
    if "integrated_gradients_fixed" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(
                x_test,
                baselines=x_test * 0,
                additional_forward_args=(data_mask, timesteps, False),
                temporal_additional_forward_args=temporal_additional_forward_args,
                task="binary",
                show_progress=True,
            )
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_fixed"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_point" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                for f in range(x_batch.shape[2]):
                    baselines = x_batch.clone()
                    baselines[:, t, f] = 0
                    attr_batch[:, t, f] = explainer.attribute(
                        x_batch,
                        baselines=baselines,
                        target=partial_targets,
                        additional_forward_args=(data_mask, timesteps, False),
                    )[:, t, f]
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_point"] = th.cat(integrated_gradients, dim=0)
    
    if "integrated_gradients_point_abs" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                for f in range(x_batch.shape[2]):
                    baselines = x_batch.clone()
                    baselines[:, t, f] = 0
                    attr_batch[:, t, f] = explainer.attribute(
                        x_batch,
                        baselines=baselines,
                        target=partial_targets,
                        additional_forward_args=(data_mask, timesteps, False),
                    )[:, t, f].abs()
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_point_abs"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_online" in explainers:
        explainer = IntegratedGradients(classifier.predict)

        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                baselines = x_batch.clone()
                baselines[:, t, :] = 0
                attr_batch[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :].abs()
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_online"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_online_v2" in explainers:
        explainer = IntegratedGradients(classifier.predict)

        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                baselines = x_batch.clone()
                baselines[:, t:, :] = 0
                attr_batch[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :].abs()
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_online_v2"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_feature" in explainers:
        explainer = IntegratedGradients(classifier.predict)
        
        integrated_gradients = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                attr_batch[:, :, f] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f].abs()
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_feature"] = th.cat(integrated_gradients, dim=0)
        
    if "deeplift_abs" in explainers:
        explainer = DeepLift(classifier)

        deeplift = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)
            
            # Calculate Integrated Gradients for the current batch
            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                #temporal_additional_forward_args=temporal_additional_forward_args,
                #task="binary",
                #show_progress=False  # Disable progress bar for individual batches
            ).abs()
            
            # Append the IG attributes of the current batch to the list
            deeplift.append(attr_batch.cpu())  # Move to CPU if necessary
        
        # Concatenate all batch IG attributes into a single tensor
        attr["deeplift"] = th.cat(deeplift, dim=0)

        
    if "deeplift_fixed" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        
        deeplift = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(
                x_test,
                baselines=x_test * 0,
                additional_forward_args=(data_mask, timesteps, False),
                temporal_additional_forward_args=temporal_additional_forward_args,
                task="binary",
                show_progress=True,
            )
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_fixed"] = th.cat(deeplift, dim=0)
        
    if "deeplift_point" in explainers:
        explainer = DeepLift(classifier)
        
        deeplift = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                for f in range(x_batch.shape[2]):
                    baselines = x_batch.clone()
                    baselines[:, t, f] = 0
                    attr_batch[:, t, f] = explainer.attribute(
                        x_batch,
                        baselines=baselines,
                        target=partial_targets,
                        additional_forward_args=(data_mask, timesteps, False),
                    )[:, t, f]
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_point"] = th.cat(deeplift, dim=0)
    
    if "deeplift_point_abs" in explainers:
        explainer = DeepLift(classifier)
        
        deeplift = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                for f in range(x_batch.shape[2]):
                    baselines = x_batch.clone()
                    baselines[:, t, f] = 0
                    attr_batch[:, t, f] = explainer.attribute(
                        x_batch,
                        baselines=baselines,
                        target=partial_targets,
                        additional_forward_args=(data_mask, timesteps, False),
                    )[:, t, f].abs()
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_point_abs"] = th.cat(deeplift, dim=0)
        
    if "deeplift_online" in explainers:
        explainer = DeepLift(classifier)

        deeplift = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                baselines = x_batch.clone()
                baselines[:, t, :] = 0
                attr_batch[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :].abs()
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_online"] = th.cat(deeplift, dim=0)
        
    if "deeplift_feature" in explainers:
        explainer = DeepLift(classifier)
        
        deeplift = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                attr_batch[:, :, f] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f].abs()
            
            deeplift.append(attr_batch.cpu())
        
        attr["deeplift_feature"] = th.cat(deeplift, dim=0)
    
    if "gradientshap_abs" in explainers:
        explainer = GradientShap(classifier.predict)

        gradientshap = []

        # Iterate over the DataLoader to process data in batches
        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            
            # Calculate Integrated Gradients for the current batch
            attr_batch = explainer.attribute(
                    x_batch,
                    baselines=x_batch * 0,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                ).abs()
            
            
            # Append the IG attributes of the current batch to the list
            gradientshap.append(attr_batch.cpu())  # Move to CPU if necessary
        
        # Concatenate all batch IG attributes into a single tensor
        attr["gradientshap"] = th.cat(gradientshap, dim=0)
        
    if "gradientshap_online" in explainers:
        explainer = GradientShap(classifier.predict)

        gradientshap = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                baselines = x_batch.clone()
                baselines[:, t, :] = 0
                attr_batch[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :].abs()
            
            gradientshap.append(attr_batch.cpu())
        
        attr["gradientshap_online"] = th.cat(gradientshap, dim=0)
        
    if "gradientshap_feature" in explainers:
        explainer = GradientShap(classifier.predict)
        
        gradientshap = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                attr_batch[:, :, f] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f].abs()
            
            gradientshap.append(attr_batch.cpu())
        
        attr["gradientshap_feature"] = th.cat(gradientshap, dim=0)
    
    
    if "integrated_gradients_feature_modify" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                
                mean_baselines = x_batch.clone()
                mean_values = x_batch[:, :, f].mean(dim=-1)
                mean_values = mean_values.unsqueeze(-1).expand(-1, x_batch.shape[1])
                mean_baselines[:, :, f] = mean_values
                
                attr_batch[:, :, f] = explainer.attribute(
                    mean_baselines,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f]
                
                attr_batch[:, :, f] += explainer.attribute(
                    x_batch,
                    baselines=mean_baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f]
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_feature_modify"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_smooth" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            
            for _ in range(10):
                noise = th.FloatTensor(x_batch.shape).uniform_(-1, 1).to(x_batch.device)
                x_batch = x_batch + noise
                for f in range(x_batch.shape[2]):
                    baselines = x_batch.clone()
                    baselines[:, :, f] = 0
                    attr_batch[:, :, f] += explainer.attribute(
                        x_batch,
                        baselines=baselines,
                        target=partial_targets,
                        additional_forward_args=(data_mask, timesteps, False),
                    )[:, :, f]
                    
            attr_batch /= 10
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_smooth"] = th.cat(integrated_gradients, dim=0)
        
    if "diff" in explainers:
        from captum._utils.common import _run_forward
        with th.autograd.set_grad_enabled(False):
            partial_targets = _run_forward(
                classifier,
                x_test,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
        partial_targets = th.argmax(partial_targets, -1)
        
        with th.no_grad():
            _attr = th.zeros_like(x_test).cpu()
            predictions = classifier(x_test, data_mask, timesteps, False)
            predictions = predictions.gather(1, partial_targets.unsqueeze(1)).squeeze(1)
            num_steps=0
            for t in range(x_test.shape[1]):
                for f in range(x_test.shape[2]):
                    baselines = x_test.clone()
                    baselines[:, t, f] = 0
                    
                    zero_predictions = classifier(baselines, data_mask, timesteps, False)
                    zero_predictions = zero_predictions.gather(1, partial_targets.unsqueeze(1)).squeeze(1)
                    
                    _attr[:, t, f] = (predictions - zero_predictions).reshape(-1).cpu()
                    
                    num_steps += 1
                    if num_steps % 10 == 0:
                        print(f"{num_steps} done")
        
        attr["diff"] = _attr
        
    if "diff_abs" in explainers:
        from captum._utils.common import _run_forward
        with th.autograd.set_grad_enabled(False):
            partial_targets = _run_forward(
                classifier,
                x_test,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
        partial_targets = th.argmax(partial_targets, -1)
        
        with th.no_grad():
            _attr = th.zeros_like(x_test).cpu()
            predictions = classifier(x_test, data_mask, timesteps, False)
            predictions = predictions.gather(1, partial_targets.unsqueeze(1)).squeeze(1)
            num_steps=0
            for t in range(x_test.shape[1]):
                for f in range(x_test.shape[2]):
                    baselines = x_test.clone()
                    baselines[:, t, f] = 0
                    
                    zero_predictions = classifier(baselines, data_mask, timesteps, False)
                    zero_predictions = zero_predictions.gather(1, partial_targets.unsqueeze(1)).squeeze(1)
                    
                    _attr[:, t, f] = (predictions - zero_predictions).abs().reshape(-1).cpu()
                    
                    num_steps += 1
                    if num_steps % 10 == 0:
                        print(f"{num_steps} done")
        
        attr["diff_abs"] = _attr
        
        
    if "integrated_two_stage" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr_first = th.cat(integrated_gradients, dim=0)
        
        q50 = th.quantile(attr_first.reshape(-1), 0.5)
        
        baselines = x_test.clone()
        baselines[attr_first > q50] = 0.0
        
        integrated_gradients = []
        for batch_idx, batch in enumerate(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            batch_baselines = baselines[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=batch_baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr_second = th.cat(integrated_gradients, dim=0)
        
        attr_second[attr_first <= q50] = attr_first[attr_first <= q50]
        attr["two_stage"] = attr_second.abs()
        
    if "integrated_two_stage_both" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr_first = th.cat(integrated_gradients, dim=0)
        
        q50 = th.quantile(attr_first.reshape(-1), 0.5)
        
        # q50 = 0.0
        
        baselines = x_test.clone()
        baselines[attr_first > q50] = 0.0
        
        integrated_gradients = []
        for batch_idx, batch in enumerate(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            batch_baselines = baselines[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=batch_baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr_second = th.cat(integrated_gradients, dim=0)
        
        baselines = x_test.clone()
        baselines[attr_first < q50] = 0.0
        
        integrated_gradients = []
        for batch_idx, batch in enumerate(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            batch_baselines = baselines[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=batch_baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr_first_fix = th.cat(integrated_gradients, dim=0)
        print(f"""
              attr_first
              min: {th.quantile(attr_first.reshape(-1), 0.0)}
              q1: {th.quantile(attr_first.reshape(-1), 0.25)}
              q2: {th.quantile(attr_first.reshape(-1), 0.5)}
              q3: {th.quantile(attr_first.reshape(-1), 0.75)}
              max: {th.quantile(attr_first.reshape(-1), 1.0)}
              """)
        
        print(f"""
              attr_first_fix
              min: {th.quantile(attr_first_fix.reshape(-1), 0.0)}
              q1: {th.quantile(attr_first_fix.reshape(-1), 0.25)}
              q2: {th.quantile(attr_first_fix.reshape(-1), 0.5)}
              q3: {th.quantile(attr_first_fix.reshape(-1), 0.75)}
              max: {th.quantile(attr_first_fix.reshape(-1), 1.0)}
              """)
        
        print(f"""
              attr_second
              min: {th.quantile(attr_second[attr_first > q50].reshape(-1), 0.0)}
              q1: {th.quantile(attr_second[attr_first > q50].reshape(-1), 0.25)}
              q2: {th.quantile(attr_second[attr_first > q50].reshape(-1), 0.5)}
              q3: {th.quantile(attr_second[attr_first > q50].reshape(-1), 0.75)}
              max: {th.quantile(attr_second[attr_first > q50].reshape(-1), 1.0)}
              """)
        
        print(th.max(attr_first_fix))
        # raise RuntimeError
        
        attr_second[attr_first <= q50] = attr_first_fix[attr_first <= q50]
        
        attr_total = attr_second + attr_first
        attr["two_stage_both"] = attr_total.abs() 
        
    if "integrated_i_j" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []
        integrated_gradients_i = []
        integrated_gradients_j = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)
            
            baselines_i = x_batch.clone()
            baselines_j = x_batch.clone()
            
            baselines_i[:, ::2, :] = 0
            baselines_j[:, 1::2, :] = 0

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            attr_batch_i = explainer.attribute(
                x_batch,
                baselines=baselines_i,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            attr_batch_j = explainer.attribute(
                x_batch,
                baselines=baselines_j,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
        
            integrated_gradients.append(attr_batch.cpu())
            integrated_gradients_i.append(attr_batch_i.cpu())
            integrated_gradients_j.append(attr_batch_j.cpu())
        
        attr_orig = th.cat(integrated_gradients, dim=0)
        attr_i = th.cat(integrated_gradients_i, dim=0)
        attr_j = th.cat(integrated_gradients_j, dim=0)
        
        
        attr_total = attr_orig.abs() + attr_i.abs() + attr_j.abs()
        
        attr["integrated_i_j"] = attr_total 
        
    if "integrated_random_mask" in explainers:
        explainer = IntegratedGradients(classifier)
        
        importance_scores = []
        
        num_random = 20
        observe_ratio = 0.2
        
        for _ in range(num_random):
            integrated_gradients = []
            for batch in test_loader:
                x_batch = batch[0].to(device)
                data_mask = batch[1].to(device)
                batch_size = x_batch.shape[0]
                timesteps = timesteps[:batch_size, :]
                
                from captum._utils.common import _run_forward
                with th.autograd.set_grad_enabled(False):
                    partial_targets = _run_forward(
                        classifier,
                        x_batch,
                        additional_forward_args=(data_mask, timesteps, False),
                    )
                partial_targets = th.argmax(partial_targets, -1)
                
                baselines = x_batch.clone()
                
                mask = th.rand_like(baselines) > observe_ratio
                baselines = baselines * mask

                attr_batch = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                    # temporal_additional_forward_args=temporal_additional_forward_args,
                    # task="binary",
                    # show_progress=True,
                ).abs()
            
                integrated_gradients.append(attr_batch.cpu())
                
            attr_random = th.cat(integrated_gradients, dim=0)
            importance_scores.append(attr_random)
            
        attr[f"random_mask_{observe_ratio}-abs"] = th.stack(importance_scores, dim=0).mean(dim=0)
        
    if "integrated_three_stage" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr_first = th.cat(integrated_gradients, dim=0)
        
        q33 = th.quantile(attr_first.reshape(-1), 0.33)
        
        baselines = x_test.clone()
        baselines[attr_first > q33] = 0.0
        
        integrated_gradients = []
        for batch_idx, batch in enumerate(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            batch_baselines = baselines[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=batch_baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr_second = th.cat(integrated_gradients, dim=0)
        
        q66 = th.quantile(attr_second.reshape(-1), 0.66)
        
        baselines = x_test.clone()
        baselines[attr_second > q66] = 0.0
        
        integrated_gradients = []
        for batch_idx, batch in enumerate(test_loader):
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            batch_baselines = baselines[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=batch_baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr_third = th.cat(integrated_gradients, dim=0)
        attr_third[attr_second <= q66] = attr_second[attr_second <= q66]
        attr_third[attr_first <= q33] = attr_first[attr_first <= q33]
        
        attr["three_stage"] = attr_third
        
        
    if "integrated_gradients_online_feature" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                # Get model outputs
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                baselines = x_batch.clone()
                baselines[:, t, :] = 0
                attr_batch[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :].abs()
            
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                attr_batch[:, :, f] += explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f].abs()
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_online_feature"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_max" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                # Get model outputs
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_t = th.zeros_like(x_batch)
            for t in range(x_batch.shape[1]):
                baselines = x_batch.clone()
                baselines[:, t, :] = 0
                attr_t[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :]
            
            attr_f = th.zeros_like(x_batch)
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                attr_f[:, :, f] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f]
                
            attr_all = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
            
            attr_batch = th.max(th.stack([attr_all.abs(), attr_f.abs(), attr_t.abs()]), dim=0).values
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_max"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_base" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_base"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_base_abs" in explainers:
        explainer = IntegratedGradients(classifier.predict)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            attr_batch = explainer.attribute(
                x_batch,
                baselines=x_batch * 0,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            ).abs()
        
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_base_abs"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_base_cf" in explainers:
        explainer = IntegratedGradients(classifier)
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)

            baselines = th.zeros_like(x_batch).to(x_batch.device)
            baselines[:, 1:, :] = x_batch[:, :-1, :]
            attr_batch = explainer.attribute(
                x_batch,
                baselines=baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
                
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_base_cf"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_base_zero_cf" in explainers:
        explainer = IntegratedGradients(classifier)

        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)
                
            baselines = th.zeros_like(x_batch).to(x_batch.device)
            baselines[:, 1:, :] = x_batch[:, :-1, :]
            
            # print(baselines.shape)
            # print(partial_targets.shape)
            # print(data_mask.shape)
            # print(timesteps.shape)
            # raise RuntimeError
            attr_batch = explainer.attribute(
                baselines, 
                baselines=(baselines * 0),
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False)
            )
            
            attr_batch += explainer.attribute(
                x_batch,
                baselines=baselines,
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                # temporal_additional_forward_args=temporal_additional_forward_args,
                # task="binary",
                # show_progress=True,
            )
                
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_base_zero_cf"] = th.cat(integrated_gradients, dim=0)

    if "lime" in explainers:
        explainer = Lime(classifier)
        attr["lime"] = explainer.attribute(
            x_test,
            additional_forward_args=(data_mask, timesteps, False),
            # temporal_additional_forward_args=temporal_additional_forward_args,
            # task="binary",
            target=1,
            #show_progress=True,
        ).abs()
        
    if "kernelshap" in explainers:
        explainer = KernelShap(classifier)
        attr["kernelshap"] = explainer.attribute(
            x_test,
            baselines=0,
            target=1,
            additional_forward_args=(data_mask, timesteps, False),
            show_progress=True,
        ).abs()

    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            additional_forward_args=(data_mask, timesteps, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            task="binary",
            show_progress=True,
        ).abs()

    if "occlusion" in explainers:
        explainer = TemporalOcclusion(classifier.predict)
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            additional_forward_args=(data_mask, timesteps, False),
            # temporal_additional_forward_args=temporal_additional_forward_args,
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()
        
    if "fo_orig" in explainers:
        explainer = Occlusion(classifier.predict)
        attr["fo_orig"] = explainer.attribute(
            x_test,
            target=1,
            sliding_window_shapes=(1,1),
            baselines=(x_test*0),
            additional_forward_args=(data_mask, timesteps, False),
            #temporal_additional_forward_args=temporal_additional_forward_args,
            #attributions_fn=abs,
            #task="binary",
            show_progress=True,
        ).abs()
        
    if "fa" in explainers:
        explainer = FeatureAblation(classifier.predict)
        attr["fa_target_1"] = explainer.attribute(
            x_test,
            baselines=0,
            target=1,
            additional_forward_args=(data_mask, timesteps, False),
            #temporal_additional_forward_args=temporal_additional_forward_args,
            show_progress=True,
        ).abs()

    if "retain" in explainers:
        retain = RetainNet(
            dim_emb=128,
            dropout_emb=0.4,
            dim_alpha=8,
            dim_beta=8,
            dropout_context=0.4,
            dim_output=2,
            temporal_labels=False,
            loss="cross_entropy",
        )
        explainer = Retain(
            datamodule=mimic3,
            retain=retain,
            trainer=Trainer(
                max_epochs=50,
                accelerator=accelerator,
                devices=device_id,
                deterministic=deterministic,
                logger=TensorBoardLogger(
                    save_dir=".",
                    version=random.getrandbits(128),
                ),
            ),
        )
        attr["retain"] = (
            explainer.attribute(x_test, target=y_test, additional_forward_args=(data_mask, timesteps, False)).abs().to(device)
        )
        
    if "motif_ig" in explainers:
        explainer = ShapeletExplainer(
            model=classifier,
            base_explainer="ig",
            device=x_test.device,
            num_features=31,
            num_classes=2,
            data_name='mimic',
            path=Path('./motif/'),
        )

        train_dataset = TensorDataset(x_train, x_train, mask_train)
        train_loader = DataLoader(train_dataset, batch_size=testbs, shuffle=True)
        explainer._discover_shapelets(train_loader)

        integrated_gradients = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            
            attr_batch = explainer.attribute(
                x_test,
                data_mask
            )
            
            integrated_gradients.append(attr_batch)
        
        attr["motif_ig"] = th.cat(integrated_gradients, dim=0)
        
        # attr["motif_ig"] = attr_batch
    
    if "gradient" in explainers:
        
        integrated_gradients = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            x_batch.requires_grad = True
            from captum._utils.common import _run_forward
            predictions = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            
            partial_targets = th.argmax(predictions, -1)
            selected_logits = predictions.gather(1, partial_targets.unsqueeze(1)).squeeze(1)
            loss = selected_logits.mean()
            loss.backward()
            
            integrated_gradients.append(x_batch.grad.detach().cpu())
            
        attr["gradient"] = th.cat(integrated_gradients, dim=0)
        
    if "timex" in explainers:
        from attribution.timex import TimeXExplainer
        explainer = TimeXExplainer(
            model=classifier,
            device=x_test.device,
            num_features=31,
            num_classes=2,
            data_name='mimic',
            split=fold,
            is_timex=True,
        )
        
        explainer.train_timex(x_train, y_train, x_test, y_test, skip_train_timex)
            
        timex_results = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(
                x_batch,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
            timex_results.append(attr_batch.detach().cpu())
        
        
        attr["timex"] = th.cat(timex_results, dim=0)
        
    if "timex++" in explainers:
        from attribution.timex import TimeXExplainer
        explainer = TimeXExplainer(
            model=classifier,
            device=x_test.device,
            num_features=31,
            num_classes=2,
            data_name='mimic',
            split=fold,
            is_timex=False,
        )
        
        explainer.train_timex(x_train, y_train, x_test, y_test, skip_train_timex)
            
        timex_results = []

        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            attr_batch = explainer.attribute(
                x_batch,
                additional_forward_args=(data_mask, timesteps, False),
            )
            
            timex_results.append(attr_batch.detach().cpu())
        
        
        attr["timex++"] = th.cat(timex_results, dim=0)
    
    if "our" in explainers:
        from attribution.explainers import OUR
        
        explainer = OUR(classifier.predict)
        
        our_results = []
        
        for batch in test_loader:
            x_batch = batch[0].to(device)
            data_mask = batch[1].to(device)
            batch_size = x_batch.shape[0]
            timesteps = timesteps[:batch_size, :]
            
            from captum._utils.common import _run_forward
            with th.autograd.set_grad_enabled(False):
                partial_targets = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(data_mask, timesteps, False),
                )
            partial_targets = th.argmax(partial_targets, -1)
            
            # attr_batch = explainer.naive_attribute(
            attr_batch = explainer.attribute_random_time_segments_one_dim_same_for_batch(
                x_batch, 
                baselines=x_batch * 0,
                targets=partial_targets*0,
                additional_forward_args=(data_mask, timesteps, False),
                n_samples=1000,
                num_segments=50,
                min_seg_len=10,
                #max_seg_len=30,
            ).abs()
            
            our_results.append(attr_batch.detach().cpu())
            
        # attr["timeig_sample50_seg25_min7_max30"] = th.cat(our_results, dim=0)
        attr["timeig_sample1000_seg50_min10"] = th.cat(our_results, dim=0)
        # attr["naive_ig_beta"] = th.cat(our_results, dim=0)

    # # Classifier and x_test to cpu
    ## classifier.to("cpu")
    ## x_test = x_test.to("cpu")

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)

    # Dict for baselines
    baselines_dict = {0: "Average", 1: "Zeros"}
    
    # ## data_mask=mask_test.to("cpu")
    # data_mask = mask_test.to(x_test.device)
    # data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):
            for topk in areas:
                for k, v in attr.items():        
                    if topk == 0.1:
                        cum_diff, AUCC = cumulative_difference(
                            classifier,
                            x_test,
                            attributions=v.cpu(),
                            baselines=baselines,
                            topk=0.0,
                            top=args.top,
                            testbs=testbs,
                            additional_forward_args=(mask_test, timesteps, False),
                        )
                    else:
                        cum_diff = 0.0
                        AUCC = 0.0
                    
                    
                    
                    
                    total_acc = 0.0
                    total_comp = 0.0
                    total_ce = 0.0
                    total_lodds = 0.0
                    total_suff = 0.0
                    total_samples = 0

                    # 2. Loop over batches
                    for batch_idx, batch in enumerate(test_loader):
                        # batch = (input_tensor, data_mask, ...)
                        x_batch = batch[0].to(device)
                        data_mask_batch = batch[1].to(device)
                        batch_size = x_batch.shape[0]

                        # If timesteps is sized for the entire dataset, slice for this batch
                        # Example (adjust accordingly if needed):
                        timesteps_batch = timesteps[batch_idx * batch_size : batch_idx * batch_size + batch_size]

                        # Prepare baselines for the batch
                        # If baselines is a tensor like x_avg, slice it for the batch dimension
                        if isinstance(baselines, th.Tensor):
                            baselines_batch = baselines[batch_idx * batch_size : batch_idx * batch_size + batch_size]
                            baselines_batch = baselines_batch.to(device)
                        else:
                            # e.g., if baselines=0.0 or a scalar, you might just keep it as-is
                            # Or replicate it: baselines_batch = torch.zeros_like(x_batch)
                            baselines_batch = baselines

                        # Similarly slice the attribution tensor 'v'
                        v_batch = v[batch_idx * batch_size : batch_idx * batch_size + batch_size].to(device)

                        # 3. Compute metrics for this batch
                        acc = accuracy(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        comp = comprehensiveness(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        ce = cross_entropy(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        l_odds = log_odds(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )
                        suff = sufficiency(
                            classifier,
                            x_batch,
                            attributions=v_batch,
                            baselines=baselines_batch,
                            topk=topk,
                            additional_forward_args=(data_mask_batch, timesteps_batch, False)
                        )

                        # 4. Accumulate results (multiply by batch_size if metrics are averages)
                        #    If your metric function already returns a sum, you may not need to multiply.
                        total_acc += acc * batch_size
                        total_comp += comp * batch_size
                        total_ce += ce * batch_size
                        total_lodds += l_odds * batch_size
                        total_suff += suff * batch_size
                        total_samples += batch_size
                        
                    mean_acc = total_acc / total_samples
                    mean_comp = total_comp / total_samples
                    mean_ce = total_ce / total_samples
                    mean_lodds = total_lodds / total_samples
                    mean_suff = total_suff / total_samples

                    fp.write(str(seed) + ",")
                    fp.write(str(fold) + ",")
                    fp.write(baselines_dict[i] + ",")
                    fp.write(str(topk) + ",")
                    fp.write(k + ",")
                    fp.write(str(lambda_1) + ",")
                    fp.write(str(lambda_2) + ",")
                    fp.write(str(lambda_3) + ",")
                    fp.write(f"{cum_diff:.4},")
                    fp.write(f"{AUCC:.4},")
                    fp.write(f"{mean_acc:.4},")
                    fp.write(f"{mean_comp:.4},")
                    fp.write(f"{mean_ce:.4},")
                    fp.write(f"{mean_lodds:.4},")
                    fp.write(f"{mean_suff:.4}")
                    fp.write("\n")

    if not os.path.exists("./results_gate/"):
        os.makedirs("./results_gate/")
    for key in attr.keys():
        result = attr[key]
        if isinstance(result, tuple): result = result[0]
        np.save('./results_gate/{}_result_{}_{}.npy'.format(key, fold, seed), result.detach().cpu().numpy())
    
    print(f"{explainers} done")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            # "deep_lift",
            # "dyna_mask",
            # "extremal_mask",    #1018265, mean(0.2939）
            "gate_mask",    #577485
            # "fit",
            # "gradient_shap",
            # "integrated_gradients",
            # "lime",
            # "augmented_occlusion",
            # "occlusion",
            # "retain",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train thr rnn classifier.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=0.001,   # 0.01
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-3",
        type=float,
        default=0.01,    #0.01
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results_gate.csv",
        help="Where to save the results.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="state",
        choices=["state", "mtand", "seft", "transformer"],
    )
    parser.add_argument(
        "--testbs",
        type=int,
        default=200
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50
    )
    parser.add_argument(
        "--skip_train_motif",
        action='store_true'
    )
    parser.add_argument(
        "--skip_train_timex",
        action='store_true'
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    print(f"set seed as {seed}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(
        explainers=args.explainers,
        areas=args.areas,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        is_train=args.train,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        output_file=args.output_file,
        model_type=args.model_type,
        testbs=args.testbs,
        top=args.top,
        skip_train_motif=args.skip_train_motif,
        skip_train_timex=args.skip_train_timex
    )

