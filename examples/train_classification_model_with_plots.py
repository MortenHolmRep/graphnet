"""Example of training Model."""

import os

import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset
import wandb

from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.models import Model
from graphnet.training.callbacks import ProgressBar
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)

import numpy as np
import matplotlib.pyplot as plt
import io
import pytorch_lightning as pl

from typing import List, Any, Dict
import sklearn.metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.nn.functional import one_hot

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="plotting_example",
    entity="graphnet-team",
    save_dir=WANDB_DIR,
    log_model=True,
)


class Wandb_plotter:
    """Class to determine output based on task type."""

    def __init__(self, wandb_logger: WandbLogger, results: Any):
        """Init."""
        super().__init__()
        self.wandb_logger = wandb_logger
        self.results = results


class Classification_plotter(Wandb_plotter):
    """Class to plot classification tasks."""

    def __init__(self, *args, **kwargs):  # type: ignore
        """Init."""
        super().__init__(*args, **kwargs)

    def plot_roc(self) -> io.BytesIO:  # type: ignore
        """Plot reciever operating curve."""
        class_options = {
            1: 0,
            -1: 0,
            13: 1,
            -13: 1,
            12: 2,
            -12: 2,
            14: 2,
            -14: 2,
            16: 2,
            -16: 2,
        }
        pid_transform = torch.tensor(
            [class_options[int(value)] for value in self.results["pid"]]
        )

        y_test = one_hot(pid_transform)
        y_prob = self.results[
            ["pid_noise_pred", "pid_muon_pred", "pid_neutrino_pred"]
        ]
        print("same shape: ", y_test.shape == y_prob.shape, y_test.shape)

        nb_classes = y_test.shape[1]
        fpr: Dict[str, Any] = dict()
        tpr: Dict[str, Any] = dict()
        roc_auc: Dict[str, Any] = dict()
        for i in range(nb_classes):
            fpr[i], tpr[i], _ = roc_curve(  # type: ignore
                y_test[:, i], y_prob.iloc[:, i], pos_label=1
            )  # type: ignore
            roc_auc[i] = auc(fpr[i], tpr[i])  # type: ignore

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_prob.values.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(
            np.concatenate([fpr[i] for i in range(nb_classes)])  # type: ignore
        )

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(nb_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # type: ignore

        # Finally average it and compute AUC
        mean_tpr /= nb_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        macro_roc_auc_ovo = roc_auc_score(
            y_test, y_prob, multi_class="ovo", average="macro"
        )
        weighted_roc_auc_ovo = roc_auc_score(
            y_test, y_prob, multi_class="ovo", average="weighted"
        )
        macro_roc_auc_ovr = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )
        weighted_roc_auc_ovr = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="weighted"
        )

        # Plot all ROC curves
        fig, ax = plt.subplots()
        # inset axes
        axins = ax.inset_axes([0.56, 0.03, 0.42, 0.42])

        ax.plot(
            fpr["micro"],
            tpr["micro"],
            color="tab:olive",
            linestyle="dashdot",
            label="micro-average ROC curve (area = {0:0.4f})"
            "".format(roc_auc["micro"]),
            linewidth=2,
        )

        axins.plot(
            fpr["micro"],
            tpr["micro"],
            color="tab:olive",
            linestyle="dashdot",
            label="micro-average ROC curve (area = {0:0.4f})"
            "".format(roc_auc["micro"]),
            linewidth=2,
        )

        ax.plot(
            fpr["macro"],
            tpr["macro"],
            color="tab:purple",
            linestyle="dashdot",
            label="macro-average ROC curve (area = {0:0.4f})"
            "".format(roc_auc["macro"]),
            linewidth=2,
        )

        axins.plot(
            fpr["macro"],
            tpr["macro"],
            color="tab:purple",
            linestyle="dashdot",
            label="macro-average ROC curve (area = {0:0.4f})"
            "".format(roc_auc["macro"]),
            linewidth=2,
        )

        colors = ["tab:blue", "tab:green", "tab:orange"]

        for i, item in enumerate(["noise", "muon", "neutrino"]):
            ax.plot(
                fpr[i],  # type: ignore
                tpr[i],  # type: ignore
                color=colors[i],
                label="ROC of {0} class (area = {1:0.4f})"
                "".format(item, roc_auc[i]),  # type: ignore
            )
            axins.plot(
                fpr[i],  # type: ignore
                tpr[i],  # type: ignore
                color=colors[i],
                label="ROC of {0} class (area = {1:0.4f})"
                "".format(item, roc_auc[i]),  # type: ignore
            )

        ax.plot([0, 1], [0, 1], "k--", label="random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver operating characteristic to multi-class")
        ax.legend(loc="upper right", framealpha=0.99)

        # sub region of the original image
        x1, x2, y1, y2 = 0, 0.02, 0.92, 1
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        ax.indicate_inset_zoom(axins, edgecolor="black")

        ax.text(
            0.775,
            0.6,
            f"""One-vs-One ROC AUC scores:
            \n{macro_roc_auc_ovo:.4f} (macro),
            \n{weighted_roc_auc_ovo:.4f} (weighted by prevalence)
            \nOne-vs-Rest ROC AUC scores:
            \n{macro_roc_auc_ovr:.4f} (macro),
            \n{weighted_roc_auc_ovr:.4f} (weighted by prevalence)""",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.99),
        )

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")

        # Save plot to wandb_logger
        self.wandb_logger.experiment.log({"ROC Curves": wandb.Image(buf)})


def train(general_config: Dict[str, Any]) -> None:
    """Train model with configuration given by `config`."""
    # Configuration
    config = TrainingConfig(
        target="pid",
        early_stopping_patience=0,
        fit={"gpus": None, "max_epochs": 1},
        dataloader={"batch_size": 128, "num_workers": 2},
    )

    run_name = "dynedge_{}_classification_example".format(config.target)

    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    #
    dataset_config = DatasetConfig.load(
        GRAPHNET_ROOT_DIR + "/configs/datasets/" + general_config["dataset"]
    )
    datasets = Dataset.from_config(dataset_config)

    # Construct datasets from multiple selections
    train_dataset = ConcatDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = ConcatDataset(
        [datasets[key] for key in datasets if key.startswith("valid")]
    )
    test_dataset = ConcatDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
    )

    # Construct dataloaders
    train_dataloaders = DataLoader(
        train_dataset, shuffle=True, **config.dataloader
    )
    valid_dataloaders = DataLoader(
        valid_dataset, shuffle=False, **config.dataloader
    )
    test_dataloaders = DataLoader(
        test_dataset, shuffle=False, **config.dataloader
    )

    wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Build model
    model_config = ModelConfig.load(
        GRAPHNET_ROOT_DIR + "/configs/models/" + general_config["model"]
    )
    model = Model.from_config(model_config, trust=True)

    wandb_logger.experiment.config.update(model_config.as_dict())

    # class_names = ["noise", "muon", "neutrino"]
    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
        ),
        ProgressBar(),
    ]

    model.fit(
        train_dataloaders,
        valid_dataloaders,
        callbacks=callbacks,
        logger=wandb_logger,
        **config.fit,
    )

    # Get predictions
    if isinstance(config.target, str):
        prediction_columns = [
            config.target + "_noise_pred",
            config.target + "_muon_pred",
            config.target + "_neutrino_pred",
        ]
        additional_attributes = [config.target]
    else:
        prediction_columns = [target + "_pred" for target in config.target]
        additional_attributes = config.target

    results = model.predict_as_dataframe(
        test_dataloaders,
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    Classification_plotter(wandb_logger, results)  # type: ignore

    # Save predictions and model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = os.path.join(general_config["archive"], db_name, run_name)
    os.makedirs(path, exist_ok=True)
    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


def main() -> None:
    """Run example."""
    # General configuration
    general_config = {
        "dataset": "PID_classification_last_one_lvl3MC.yml",
        "model": "dynedge_PID_classification_example.yml",
        "archive": "/groups/icecube/petersen/GraphNetDatabaseRepository/example_results/train_classification_model",
    }

    train(general_config)


if __name__ == "__main__":
    main()
