import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

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


            