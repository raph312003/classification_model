import torch
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    patience: int,
    epoch: int,
    criterion:torch.nn.Module, 
    optimizer: torch.optim.Optimizer
):
    best_val_loss = float("inf")
    patience_counter = 0
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

            output = model(batch_image)

            loss = criterion(output,batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())
            train_loss += loss.item()

        avg_train_loss = train_loss/len(train_dataloader)

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            best_weights = copy.deepcopy(model.state_dict())

        else:
            patience_counter+=1
            if patience_counter == patience:
                break
        
        # print("nb epoch", i)
        # print("Avg val loss",avg_val_loss)
        # print("Avg train loss",avg_train_loss)

    return (best_weights)
        