from classification_pokemon.preprocessing import (
    dataLoading, 
    LoadingPreProcessing, 
    DataAugmentor2D,
    AugmentationConfig
)
from classification_pokemon.train import train
from classification_pokemon.test import test, graph_loss_epoch, save_model_weight
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
    parser.add_argument("--data_augmentation", type=Path, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    args = parser.parse_args()  # Get args.training_config (Path/None) 
    track_architecture = args.architecture

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

    # Load default data_augmentation
    default_data_augmentation_config_path = (
        PROJECT_ROOT / "configs" / "augmentation" / "augmentation.yaml"
    )
    if not default_data_augmentation_config_path.exists():
        LOGGER.error("Default data augmentation file not found at %s", default_data_augmentation_config_path)
        sys.exit(1)
    with default_data_augmentation_config_path.open("r") as file:
        data_augmentation_config = yaml.safe_load(file)
    
    # If custom data augmentation config is provided, update default config with custom values
    if args.data_augmentation is not None:
        if not args.data_augmentation.exists():
            LOGGER.error("Custom data_augmentation file not found at %s", args.data_augmentation)
            sys.exit(1)
        with args.data_augmentation.open("r") as file:
            data_augmentation_config = yaml.safe_load(file)

    # Data augmentation
    if config["data_augmentation"]:
        LOGGER.info("Using data augmentation")
        augmentation_cfg = AugmentationConfig(**data_augmentation_config)
        augmentor = DataAugmentor2D(config = augmentation_cfg)
    else :
        augmentor = None

    if track_architecture:
        architecture = track_architecture
    else:
        architecture = config["architecture"]

    loader = dataLoading(
        folder_path_0 = config["train_data"],
        folder_path_1 = config["val_data"], 
        target_size = config["target_size"],
        conversion = config["conversion"]
    )

    X, y = loader._load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y      
    )

    train_dataset = LoadingPreProcessing(X_train, y_train, config["normalization"], config["conversion"])
    val_dataset = LoadingPreProcessing(X_val, y_val, config["normalization"], config["conversion"])

    train_dataloader = DataLoader(train_dataset, config["batchsize"], shuffle = True)
    val_dataloader = DataLoader(val_dataset, config["batchsize"], shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Running on device %s", device)

    model_name = architecture
    match model_name:
        case "cnn_classifier":
            from classification_pokemon.model.cnn_classifier import EncoderClassifier2D
            model = EncoderClassifier2D
        case "viT_classifier":
            from classification_pokemon.model.viT_classifier import ViT
            model = ViT
        case "cnn_FT_efficient_net":
            from classification_pokemon.model.cnn_FT_efficient_net import EfficientNetDevelopper
            model = EfficientNetDevelopper
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

    model_weights, list_avg_train_loss, list_avg_val_loss = train(
        model = model,
        device = device,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        patience = config["early_stopping"],
        epoch = config["epoch"],
        criterion = criterion, 
        optimizer = optimizer,
        augmentor = augmentor
    )

    save_model_weight(
        model_weights, 
        architecture
    )

    graph_loss_epoch(
        list_avg_train_loss, 
        list_avg_val_loss
    )

    test(
        device = device,
        model = model,
        val_dataloader = val_dataloader,
        best_weight = model_weights,
    )

if __name__ == "__main__":
    main()