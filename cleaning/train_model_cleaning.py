import os.path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import  BinaryCrossEntropyLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices, get_desired_event_numbers
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeUpgrade, IceCubeUpgrade_V3
from graphnet.models.gnn import DynEdge_V3
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import BinaryClassificationTask
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import get_predictions, make_train_validation_dataloader, save_results

# Configurations
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="wand-db-test",
    entity="morten",
    save_dir='./results/wandb',
    log_model=False,
    offline=True,
)
# Main function definition
def main():

    print(f"features: {features}")
    print(f"truth: {truth}")

    # Configuration
    config = {
        "db": '/groups/icecube/asogaard/data/sqlite/dev_step4_numu_140021_second_run/data/dev_step4_numu_140021_second_run.db',
        "pulsemap": 'SplitInIcePulses',
        "batch_size": 256,
        "num_workers": 20,
        "gpus": [0],
        "target": 'truth_flag',
        "n_epochs": 50,
        "patience": 5,
    }
    archive = "/groups/icecube/qgf305/graphnet_user/results/"
    run_name = "dynedge_noiseClassification_DEEPCORE_{}".format(config["target"])

    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables
    train_selection = get_desired_event_numbers(config["db"], desired_size=1_000_000, fraction_nu_mu = 1)

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        config["db"],
        train_selection,
        config["pulsemap"],
        features,
        truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Building model
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge_V3(
        nb_inputs=detector.nb_outputs,
    )
    task = BinaryClassificationTask(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * config["n_epochs"]],
            'factors': [1e-2, 1, 1e-02],
        },
        scheduler_config={
            'interval': 'step',
        },
     )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        gpus=config["gpus"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [config["target"] + '_pred'],
        [config["target"], 'dom_type', 'event_no']
    )

    save_results(config["db"], run_name, results, archive, model)

# Main function call
if __name__ == "__main__":
    main()
