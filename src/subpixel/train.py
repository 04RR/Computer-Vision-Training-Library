import torch
from tqdm import tqdm
import warnings
from subpixel.data import ImageDataset
import numpy as np
import torch.nn as nn


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(out, labels):

    c = 0

    preds = torch.round(out)
    preds = preds.detach().cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()

    for label, pred in zip(labels, preds):
        if pred == label:
            c += 1

    return c / len(out)


class Trainer:
    def __init__(
        self,
        model,
        trainset,
        valset=None,
        epochs=10,
        mode="classification",
        loss_fn=nn.MSELoss(),
        learning_rate=None,
        weight_decay=1e-5,
        model_save_path="./",
    ):
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.epochs = epochs
        self.mode = mode
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.model_save_path = model_save_path

        if learning_rate != None:
            learning_rate = self.find_lr()  # add FindLR class from utils after done

    def fit(self):

        flag = self.mode == "classification" or self.mode == "detection"

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler()
        losses = {"train": [], "val": []}
        acc = {"train": [], "val": []}

        for epoch in range(self.epochs):

            epoch_loss = {"train": [], "val": []}
            epoch_acc = {"train": [], "val": []}

            self.model.train()
            for img, label in tqdm(self.trainset):

                img, label = img.unsqueeze(0), label.unsqueeze(0)

                with torch.cuda.amp.autocast():

                    pred = self.model(img)
                    loss = self.loss_fn(pred, label)

                    epoch_loss["train"].append(loss)

                    if self.mode == "classification":
                        a = accuracy(pred, label)
                        epoch_acc["train"].append(a)

                    elif self.mode == "detection":
                        a = accuracy(pred[1:5], label[1:5])
                        epoch_acc["train"].append(a)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses["train"].append(sum(epoch_loss["train"]) / len(epoch_loss["train"]))

            if self.valset != None:

                self.model.eval()
                for img, label in tqdm(self.valset):

                    img, label = img.unsqueeze(0), label.unsqueeze(0)

                    with torch.cuda.amp.autocast():

                        pred = self.model(img)
                        loss = self.loss_fn(pred, label)

                        epoch_loss["val"].append(loss)

                        if self.mode == "classification":
                            a = accuracy(pred, label)
                            epoch_acc["val"].append(a)

                        elif self.mode == "detection":
                            a = accuracy(pred[1:5], label[1:5])
                            epoch_acc["val"].append(a)

                losses["val"].append(sum(epoch_loss["val"]) / len(epoch_loss["val"]))

                if flag:

                    acc["val"].append(sum(epoch_acc["val"]) / len(epoch_acc["val"]))
                    acc["train"].append(
                        sum(epoch_acc["train"]) / len(epoch_acc["train"])
                    )

                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]} -- Train acc: {acc['train'][-1]} -- Val Loss: {losses['val'][-1]} -- Val acc: {acc['val'][-1]}"
                    )
                else:
                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]} -- Val Loss: {losses['val'][-1]}"
                    )

            else:

                if flag:
                    acc["train"].append(
                        sum(epoch_acc["train"]) / len(epoch_acc["train"])
                    )

                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]} -- Train acc: {acc['train'][-1]}"
                    )
                else:
                    print(
                        f"{epoch+1}/{self.epochs} -- Train Loss: {losses['train'][-1]}"
                    )

            torch.save(self.model, f"{self.model_save_path}\\model")

        if flag:
            return losses, acc

        else:
            return losses

    def test_sample(self, image, label=None):

        pred = self.model(image)

        if label != None:
            loss = self.loss_fn(label, pred).detach()
            return pred, loss

        return pred

    def evaluate(self, test_path):

        testset = ImageDataset(test_path, self.mode, device)
        losses = []

        for img, label in testset:
            pred = self.model(img.unsqueeze(0))
            loss = self.loss_fn(label, pred).detach()
            losses.append(loss)

        return sum(losses) / len(losses)

    def find_lr(self):
        return 0.0

