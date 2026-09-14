from classification_pokemon.preprocessing import dataLoading,LoadingPreProcessing
from classification_pokemon.train import train
from classification_pokemon.test import test
from classification_pokemon.model.ccn_classifier import EncoderClassifier2D

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import torch

# from torch.nn import nn

def main():
    loader = dataLoading(
        folder_path_0 = "pikatchu/",
        folder_path_1 = "rondoudou/", 
        target_size = (64,64)
    )

    X, y = loader._load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y      
    )
    print(len(X_train))
    print(len(y_train))
    print(len(X_val))
    print(len(y_val))

    train_dataset = LoadingPreProcessing(X_train, y_train, normalization="percentile")
    val_dataset = LoadingPreProcessing(X_val, y_val, normalization="percentile")

    train_dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 4, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier2D
    model = model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    

    model_weights = train(
        model=model,
        device = device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        patience= 10,
        epoch=1000,
        criterion= torch.nn.CrossEntropyLoss(), 
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