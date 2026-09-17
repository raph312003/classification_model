import torch
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from classification_pokemon.preprocessing import DataAugmentor2D


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    patience: int,
    epoch: int,
    criterion:torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    augmentor: DataAugmentor2D | None = None
):
    best_val_loss = float("inf")
    patience_counter = 0
    list_avg_train_loss = []
    list_avg_val_loss = []
    for i in range(epoch):
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {i+1}/{epoch}"
        )
        for batch_image, batch_target in train_bar:
            batch_image = batch_image.to(device).float()
            batch_target = batch_target.to(device).long()

            if augmentor is not None:
                batch_image = augmentor(batch_image)

            output = model(batch_image)

            loss = criterion(output,batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())
            train_loss += loss.item()

        list_avg_train_loss.append(train_loss/len(train_dataloader))

        model.eval()
        val_loss = 0.0

        val_bar = tqdm(
            val_dataloader,
            desc=f"Epoch {i+1}/{epoch}"
        )

        for batch_image, batch_target in val_bar:
            batch_image = batch_image.to(device).float()
            batch_target = batch_target.to(device).long()

            output = model(batch_image)

            loss = criterion(output, batch_target)

            val_bar.set_postfix(loss=loss.item())
            val_loss += loss.item()

        avg_val_loss = train_loss/len(val_dataloader)
        list_avg_val_loss.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            best_weights = copy.deepcopy(model.state_dict())

        else:
            patience_counter+=1
            if patience_counter == patience:
                break

    return (best_weights, list_avg_train_loss, list_avg_val_loss)
        