"""Simplified example of training Model."""

from typing import List, Optional, Dict
import os
import glob

from pytorch_lightning.callbacks import EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from graphnet.constants import EXAMPLE_OUTPUT_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger
from torch.profiler import tensorboard_trace_handler
import torch
from graphnet.utilities.wandb.base_profiler_callback import (
    TorchTensorboardProfilerCallback,
)


def main(
    dataset_config_path: str,
    model_config_path: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    prediction_names: Optional[List[str]],
    suffix: Optional[str] = None,
    wandb: bool = False,
    wandb_profiling: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )
        if wandb_profiling:
            # wandb trace config example
            config = {"batch_size": 128, "workers": 10}
            # Initialize profiler
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
            # Initialize wandb
            # wandb_logger.init(project="trace", config=config, group="DDP")
            # run = wandb_logger.run
            # Modify the original wandb_logger to include profiling information
            # wandb_logger.set_job_type("train")
            wandb_logger.log_hyperparams(config)
            wandb_logger.log_hyperparams({"group": "DDP"})
            wandb_logger.log_hyperparams({"job_type": "train"})

            # Create a profiler callback using the wandb logger
            profiler_callback = TorchTensorboardProfilerCallback(
                profiler, wandb_logger
            )

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model = Model.from_config(model_config, trust=True)

    # Configuration
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    if suffix is not None:
        archive = os.path.join(EXAMPLE_OUTPUT_DIR, f"train_model_{suffix}")
    else:
        archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_model")
    run_name = "dynedge_{}_example".format(
        "_".join(config.target)
    )  # ignore: type

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        **config.dataloader,
    )  # ignore: type

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    if wandb and rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(model_config.as_dict())
        wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Train model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,  # ignore: type
        ),
        ProgressBar(),
    ]

    # add callbacks specific to profiling
    if wandb_profiling:
        callbacks.append(DeviceStatsMonitor(cpu_stats=True))
        callbacks.append(profiler_callback)

    model.fit(
        dataloaders["train"],
        dataloaders["validation"],
        callbacks=callbacks,
        logger=wandb_logger if wandb else None,
        **config.fit,
    )

    # Get predictions
    if isinstance(config.target, str):
        prediction_columns = [config.target + "_pred"]
        additional_attributes = [config.target]
    else:
        prediction_columns = [target + "_pred" for target in config.target]
        additional_attributes = config.target

    if prediction_names:
        prediction_columns = prediction_names

    logger.info(f"prediction_columns: {prediction_columns}")

    results = model.predict_as_dataframe(
        dataloaders["test"],
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    if wandb_profiling:
        profile_art = wandb_logger.use_artifact(
            artifact=f"trace-{wandb_logger._id}", artifact_type="profile"
        )
        profile_art.add_file(
            glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0],
            "trace.pt.trace.json",
        )
        # run.log_artifact(profile_art)
        wandb_logger.log_artifact(profile_art)

    # Save predictions and model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model.
"""
    )

    parser.with_standard_arguments(
        "dataset-config",
        "model-config",
        "gpus",
        ("max-epochs", 1),
        "early-stopping-patience",
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--prediction-names",
        nargs="+",
        help="Names of each prediction output feature (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    parser.add_argument(
        "--wandb_profiling",
        action="store_true",
        help="If True, Weights & Biases are used to profile the code flow.",
    )

    args = parser.parse_args()

    if args.wandb_profiling == True and args.wandb == False:
        print("Error: wandb must be True when wandb_profiling is True")

    main(
        args.dataset_config,
        args.model_config,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.prediction_names,
        args.suffix,
        # args.wandb,
        wandb=True,
        # wandb_profiling=args.wandb_profiling,
        wandb_profiling=True,
    )
