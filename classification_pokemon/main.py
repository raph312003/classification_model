from classification_pokemon.preprocessing import dataLoading,LoadingPreProcessing
from classification_pokemon.train import train
from classification_pokemon.test import test
from classification_pokemon.model.cnn_classifier import EncoderClassifier2D

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import torch

from argparse import ArgumentParser
from pathlib import Path
import yaml
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Train contrast enhance model.")

    # Get the config file by traversing parent directories
    PROJECT_ROOT = Path(__file__).resolve()

    while not (PROJECT_ROOT / "configs").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

    default_training_config_path = (
        PROJECT_ROOT / "configs" / "training" / "training.yaml"
    )

    parser.add_argument("--training_config", type=Path, default=None)
    parser.add_argument("--network_config", type=Path, default=None)
    # parser.add_argument("--data_augmentation", type=Path, default=None)
    args = parser.parse_args()  # Get args.training_config (Path/None) 

    # Load default training_config
    if not default_training_config_path.exists():
        LOGGER.error(
            "Default training config file not found at %s", default_training_config_path
        )
        sys.exit(1)
    with default_training_config_path.open("r") as file:
        config = yaml.safe_load(file)

    # If custom config is provided, update default config with custom values
    if args.training_config is not None:
        if not args.training_config.exists():
            LOGGER.error(
                "Custom training config file not found at %s", args.training_config
            )
            sys.exit(1)
        with args.training_config.open("r") as file:
            config = yaml.safe_load(file)

    # Load default net_config
    default_net_config_path = (
        PROJECT_ROOT / "configs" / "models" / (config["architecture"].lower() + ".yaml")
    )
    if not default_net_config_path.exists():
        LOGGER.error("Default net config file not found at %s", default_net_config_path)
        sys.exit(1)
    with default_net_config_path.open("r") as file:
        net_config = yaml.safe_load(file)

     # If custom net config is provided, update default net config with custom values
    if args.network_config is not None:
        if not args.network_config.exists():
            LOGGER.error("Custom net config file not found at %s", args.network_config)
            sys.exit(1)
        with args.network_config.open("r") as file:
            net_config = yaml.safe_load(file)

    loader = dataLoading(
        folder_path_0 = config["train_data"],
        folder_path_1 = config["val_data"], 
        target_size = config["target_size"]
    )

    X, y = loader._load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y      
    )

    train_dataset = LoadingPreProcessing(X_train, y_train, config["normalization"])
    val_dataset = LoadingPreProcessing(X_val, y_val, config["normalization"])

    train_dataloader = DataLoader(train_dataset, config["batchsize"], shuffle = True)
    val_dataloader = DataLoader(val_dataset, config["batchsize"], shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Running on device %s", device)

    model_name = config["architecture"]
    match model_name:
        case "cnn_classifier":
            from classification_pokemon.model.cnn_classifier import EncoderClassifier2D
            model = EncoderClassifier2D
        case _:
            LOGGER.error("Unexpected architecture = %s", config["architecture"])
            sys.exit(1)

    model = model(**net_config).to(device)

    # Choose the optimizer
    optimizer_name = config["optimizer"]
    match optimizer_name:
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        case _:
            LOGGER.error("Unexpected optimizer = %s", config["optimizer"])
            sys.exit(1)

    criterion_name = config["criterion"]
    match criterion_name:
        case "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        case _:
            LOGGER.error("Unexpected criterion = %s", config["criterion"])
            sys.exit(1)

    model_weights = train(
        model=model,
        device = device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        patience= config["early_stopping"],
        epoch=config["epoch"],
        criterion= criterion, 
        optimizer= optimizer
    )

    test(
        device = device,
        model = model,
        val_dataloader = val_dataloader,
        best_weight = model_weights,
    )

if __name__ == "__main__":
    main()