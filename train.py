import torch
from tqdm import tqdm
import warnings
from utils import *
import numpy as np
import torch.nn as nn


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"


def fit(
    model,
    trainset,
    valset,
    epochs,
    mode,
    loss_fn,
    learning_rate,
    weight_decay,
    model_save_path,
):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()
    losses = {"train": [], "val": []}
    acc = {"train": [], "val": []}

    for epoch in range(epochs):

        epoch_loss = {"train": [], "val": []}
        epoch_acc = {"train": [], "val": []}

        model.train()
        for img, label in tqdm(trainset):

            img, label = img.unsqueeze(0), label.unsqueeze(0)

            with torch.cuda.amp.autocast():

                pred = model(img)
                loss = loss_fn(pred, label)

                epoch_loss["train"].append(loss)

                if mode == "classification":
                    a = accuracy(pred, label)
                    epoch_acc["train"].append(a)

                elif mode == "detection":
                    a = accuracy(pred[1:5], label[1:5])
                    epoch_acc["train"].append(a)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        model.eval()
        for img, label in tqdm(valset):

            img, label = img.unsqueeze(0), label.unsqueeze(0)

            with torch.cuda.amp.autocast():

                pred = model(img)
                loss = loss_fn(pred, label)

                epoch_loss["val"].append(loss)

                if mode == "classification":
                    a = accuracy(pred, label)
                    epoch_acc["val"].append(a)

                elif mode == "detection":
                    a = accuracy(pred[1:5], label[1:5])
                    epoch_acc["val"].append(a)

        losses["val"].append(sum(epoch_loss["val"]) / len(epoch_loss["val"]))
        losses["train"].append(sum(epoch_loss["train"]) / len(epoch_loss["train"]))

        if mode == "classification" or mode == "detection":

            acc["val"].append(sum(epoch_acc["val"]) / len(epoch_acc["val"]))
            acc["train"].append(sum(epoch_acc["train"]) / len(epoch_acc["train"]))

            print(
                f"{epoch+1}/{epochs} -- Train Loss: {losses['train'][-1]} -- Train acc: {acc['train'][-1]} -- Val Loss: {losses['val'][-1]} -- Val acc: {acc['val'][-1]}"
            )

        else:
            print(
                f"{epoch+1}/{epochs} -- Train Loss: {losses['train'][-1]} -- Val Loss: {losses['val'][-1]}"
            )

        torch.save(model, f"{model_save_path}\\model")

    if mode == "classification" or mode == "detection":
        return losses, acc

    else:
        return losses


def test_sample(model, image, label=None, loss_fn=nn.MSELoss()):

    pred = model(image)

    if label != None:
        loss = loss_fn(label, pred).detach()
        return pred, loss

    return pred
