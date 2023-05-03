"""Simplified example of training Model."""

import os
import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.constants import GRAPHNET_ROOT_DIR

from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from torch.profiler import tensorboard_trace_handler
import wandb
from typing import Any

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="profiling_test",
    entity="graphnet-team",
    save_dir=WANDB_DIR,
    log_model=True,
)


class TorchTensorboardProfilerCallback(pl.Callback):
    """Callback for invoking TensorboardProfiler during training.

    For greater robustness, extend the pl.profiler.profilers.BaseProfiler.
    See https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html
    """

    def __init__(self, profiler: torch.profiler.profiler.profile):
        """Initialise."""
        super().__init__()
        self.profiler = profiler

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Information at batch end."""
        self.profiler.step()
        pl_module.log_dict(outputs)  # also logging the loss, while we're here


def train(wandb: Any, profiler: torch.profiler.profiler.profile) -> None:
    """Run example."""
    try:
        print(wandb.run.config)
    except Exception as e:
        print(e)
    try:
        print(wandb.config)
    except Exception as e:
        print(e)
    # Configuration
    config = TrainingConfig(
        target="energy",
        early_stopping_patience=5,
        fit={"gpus": [1], "max_epochs": 5},
        dataloader={"batch_size": 128, "num_workers": 10},
    )

    archive = "/groups/icecube/qgf305/storage/profiling/"
    run_name = "dynedge_{}_example".format(config.target)

    # Construct dataloaders
    dataset_config = DatasetConfig.load(
        GRAPHNET_ROOT_DIR
        + "/configs/datasets/dev_lvl7_robustness_muon_neutrino_0000.yml"
    )
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **config.dataloader,
    )
    with profiler:
        profiler_callback = TorchTensorboardProfilerCallback(profiler)
        # Build model
        model_config = ModelConfig.load(
            GRAPHNET_ROOT_DIR + f"/configs/models/{run_name}.yml"
        )
        model = Model.from_config(model_config, trust=True)

        # Log configurations to W&B
        # NB: Only log to W&B on the rank-zero process in case of multi-GPU
        #     training.
        if rank_zero_only == 0:
            wandb_logger.experiment.config.update(config)
            wandb_logger.experiment.config.update(model_config.as_dict())
            wandb_logger.experiment.config.update(dataset_config.as_dict())

        # Train model
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
            ),
            ProgressBar(),
            profiler_callback,
            DeviceStatsMonitor(cpu_stats=True),
        ]

        model.fit(
            dataloaders["train"],
            dataloaders["validation"],
            callbacks=callbacks,
            logger=wandb_logger,
            **config.fit,
        )

    # Get predictions
    if isinstance(config.target, str):
        prediction_columns = [config.target + "_pred"]
        additional_attributes = [config.target]
    else:
        prediction_columns = [target + "_pred" for target in config.target]
        additional_attributes = config.target

    results = model.predict_as_dataframe(
        dataloaders["test"],
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    # Save predictions and model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    os.makedirs(path, exist_ok=True)
    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    # model.save(f"{path}/model.pth")


def main() -> None:
    """Profiling example."""
    config = {"batch_size": 128, "workers": 10}
    with wandb.init(project="trace", config=config, group="DDP") as run:
        # Set up profiler
        wait, warmup, active, repeat = 1, 1, 2, 1
        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=tensorboard_trace_handler(
                "wandb/latest-run/tbprofile"
            ),
            with_stack=False,
        )

        train(wandb, profiler)

        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(
            glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0],
            "trace.pt.trace.json",
        )
        run.log_artifact(profile_art)


if __name__ == "__main__":
    main()

# https://docs.wandb.ai/guides/track/advanced/distributed-training
# if __name__ == "__main__":
#     # Get args
#     args = parse_args()
#     # Initialize run
#     run = wandb.init(
#         entity=args.entity,
#         project=args.project,
#         group="DDP",  # all runs for the experiment in one group
#     )
#     # Train model with DDP
#     train(args, run)
