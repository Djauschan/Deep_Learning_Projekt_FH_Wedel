"""
Dieses Python Skript erstellt laedt eine uebergebene json Datei
Laedt alle benoetigten Elemente und startet das Training
"""

# Imports
import json
import sys

# Torch specific packages
import torch.nn as nn

# from torchvision import transforms
import torch.optim as optim

# My intern packages for dataloader, model etc
from dataloader.datasets import DatasetPreLoaded
from dataloader.transform import DatasetTransformer
from model.networkmaker import NetworkMaker
from nettrainer import NetTrainer


# Function to load json file with all the parameters
# - Model:
# - Dataloader:
# - Trainer:
# - Optimizer:
def load_json(param_file="test_params.json", params_dir="parameters"):
    """
    Laedt eine Json Datei die als erster Parameter uebergeben wurde.
    Gibt das geladene json zurueck.
    """
    param_file = sys.argv[1] if len(sys.argv) > 1 else param_file
    with open(params_dir + "/" + param_file, "r") as openfile:
        dataholder = json.load(openfile)
    # Return the parameter dictionary
    return dataholder


def main():
    # Laden der Json Parameter
    print("[MAIN]: Loading json file")
    dataholder = load_json()

    # Laden des Transformers
    print("[MAIN]: Loading transformer")
    transformer = DatasetTransformer(
        dataholder["seed"],
        dataholder["translateH_range"],
        dataholder["translateH_chance"],
        dataholder["translateV_range"],
        dataholder["translateV_chance"],
    )

    # Laden des Datasets
    print("[MAIN]: Loading dataset")
    dataset = DatasetPreLoaded(
        dataholder["linear"],
        dataholder["samples_per_class"],
        dataholder["seed"],
        transformer,
    )

    # Laden des Netzes
    print("[MAIN]: Loading net")
    net = NetworkMaker.fromDict(dataholder["net_struct"])

    # Laden der Loss Function
    print("[MAIN]: Loading criterion")
    criterion = None
    if dataholder["criterion"] == "mse":
        criterion = nn.MSELoss()
    elif dataholder["criterion"] == "crossentropy":
        criterion = nn.CrossEntropyLoss()

    # Error Check
    if net is None:
        raise Exception("[Dataholder]: No proper 'net' found!")
    if criterion is None:
        raise Exception("[Dataholder]: No proper 'criterion' found!")

    # Trainer erzeugen
    print("[MAIN]: Loading trainer")
    trainer = NetTrainer(
        net,
        dataset,
        criterion,
        dataholder.get("seed"),
        dataholder.get("gpu"),
        dataholder.get("name"),
        dataholder.get("validation_size"),
        dataholder.get("batchsize"),
        json.dumps(dataholder, indent=4),
    )

    # Laden Optimizer
    print("[MAIN]: Loading optimizer")
    optimizer = None
    if dataholder["optimizer"] == "sge":
        optimizer = optim.SGD(
            net.parameters(),
            lr=dataholder["learning_rate"],
            momentum=dataholder["momentum"],
        )

    elif dataholder["optimizer"] == "adam":
        optimizer = optim.Adam(net.parameters())

    if optimizer is None:
        raise Exception("[Dataholder]: No proper 'optimizer' found!")

    # Start Training
    print("[MAIN]: Start Training")
    trainer.train(
        dataholder.get("epochs"),
        optimizer,
        dataholder.get("patience"),
        dataholder.get("inflation"),
    )


if __name__ == "__main__":
    main()
