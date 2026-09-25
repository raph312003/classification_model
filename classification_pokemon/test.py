import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os 

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

def test(
    device: torch.device,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    best_weight: dict,       
):

    list_output = []
    list_target = []
    model.load_state_dict(best_weight)
    model.eval()

    with torch.no_grad():
        for batch_image, batch_target in val_dataloader:
            batch_image = batch_image.to(device).float()
            batch_target = batch_target.to(device).float()

            output = model(batch_image)

            if hasattr(output, "logits"):
                output = output.logits

            predictions = torch.argmax(output, dim=1)

            output = predictions.cpu().numpy()
            batch_target = batch_target.cpu().numpy()

            list_output.append(output)
            list_target.append(batch_target)

        
        y_true = np.concatenate(list_target)
        y_pred = np.concatenate(list_output)

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        print(cm)
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Pikachu", "Rondoudou"]
        ).plot()

        plt.show()

def graph_loss_epoch(list_avg_train_loss, list_avg_val_loss):
    
    plt.plot(
        range(1, len(list_avg_train_loss)+1),
        list_avg_train_loss,
        color="blue",
        label="Train loss"
    )

    plt.plot(
        range(1, len(list_avg_val_loss)+1),
        list_avg_val_loss,
        color="red",
        label="Validation loss"

    )
    plt.ylabel("Avg loss")
    plt.xlabel("Epochs")
    plt.title("Avg loss/epoch")
    plt.legend()
    plt.show()


def save_model_weight(best_weights, architecture):
    LOGGER.info("Saving %s weights", architecture)
    os.makedirs(f"model_weights/{architecture}", exist_ok=True)
    torch.save(best_weights, f"model_weights/{architecture}/best_model.pth")

#def get_features_maps():

#def display_feature_maps():

                