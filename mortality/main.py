import sys
from os import path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))


import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn
import os
from utils.tools import print_results

from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
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
    output_file: str = "results.csv",
    model_type: str = "state",
    testbs: int = 0,
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
        y_test = mimic3.preprocess(split="test")["y"].to(device)
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
            lr=0.01,
        )
        explainer = GateMask(classifier)
        _attr = explainer.attribute(
            x_test,
            # additional_forward_args=(True,) Can I modify it??
            additional_forward_args=(True, data_mask, timesteps, False),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.5,
        )
        attr["gate_mask"] = _attr.to(device)

    if "extremal_mask" in explainers:
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
            lr=0.01,
        )
        explainer = ExtremalMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(data_mask, timesteps, False),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask"] = _attr.to(device)

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
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))

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
        
    if "integrated_gradients_online" in explainers:
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
                baselines = x_batch.clone()
                baselines[:, t, :] = 0
                attr_batch[:, t, :] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, t, :]
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_online"] = th.cat(integrated_gradients, dim=0)
        
    if "integrated_gradients_feature" in explainers:
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
                attr_batch[:, :, f] = explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f]
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_feature"] = th.cat(integrated_gradients, dim=0)
        
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
                )[:, t, :]
            
            for f in range(x_batch.shape[2]):
                baselines = x_batch.clone()
                baselines[:, :, f] = 0
                attr_batch[:, :, f] += explainer.attribute(
                    x_batch,
                    baselines=baselines,
                    target=partial_targets,
                    additional_forward_args=(data_mask, timesteps, False),
                )[:, :, f]
            
            integrated_gradients.append(attr_batch.cpu())
        
        attr["integrated_gradients_online_feature"] = th.cat(integrated_gradients, dim=0)
        
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
            baselines = th.zeros_like(x_batch).to(x_batch.device)
            baselines[:, 1:, :] = x_batch[:, :-1, :]
            
            attr_batch = explainer.attribute(
                baselines, 
                baselines=(baselines * 0),
                target=partial_targets,
                additional_forward_args=(data_mask, timesteps, False))
            
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
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            additional_forward_args=(data_mask, timesteps, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            task="binary",
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
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            additional_forward_args=(data_mask, timesteps, False),
            temporal_additional_forward_args=temporal_additional_forward_args,
            attributions_fn=abs,
            task="binary",
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

    # Classifier and x_test to cpu
    classifier.to("cpu")
    x_test = x_test.to("cpu")

    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)

    # Dict for baselines
    baselines_dict = {0: "Average", 1: "Zeros"}
    
    data_mask=mask_test
    data_len, t_len, _ = x_test.shape
        
    timesteps=(
        th.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):
            for topk in areas:
                for k, v in attr.items():
                    acc = accuracy(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        additional_forward_args=(data_mask, timesteps, False)
                    )
                    comp = comprehensiveness(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        additional_forward_args=(data_mask, timesteps, False)
                    )
                    ce = cross_entropy(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        additional_forward_args=(data_mask, timesteps, False)
                    )
                    l_odds = log_odds(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        additional_forward_args=(data_mask, timesteps, False)
                    )
                    suff = sufficiency(
                        classifier,
                        x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                        additional_forward_args=(data_mask, timesteps, False)
                    )

                    fp.write(str(seed) + ",")
                    fp.write(str(fold) + ",")
                    fp.write(baselines_dict[i] + ",")
                    fp.write(str(topk) + ",")
                    fp.write(k + ",")
                    fp.write(str(lambda_1) + ",")
                    fp.write(str(lambda_2) + ",")
                    fp.write(f"{acc:.4},")
                    fp.write(f"{comp:.4},")
                    fp.write(f"{ce:.4},")
                    fp.write(f"{l_odds:.4},")
                    fp.write(f"{suff:.4}")
                    fp.write("\n")

    if not os.path.exists("./results_gate/"):
        os.makedirs("./results_gate/")
    for key in attr.keys():
        result = attr[key]
        if isinstance(result, tuple): result = result[0]
        np.save('./results_gate/{}_result_{}_{}.npy'.format(key, fold, seed), result.detach().cpu().numpy())


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
        "--output-file",
        type=str,
        default="results_gate.csv",
        help="Where to save the results.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="state",
        choices=["state", "mtand", "seft"],
    )
    parser.add_argument(
        "--testbs",
        type=int,
        default=200
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
        output_file=args.output_file,
        model_type=args.model_type,
        testbs=args.testbs
    )

