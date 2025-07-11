# SkinScan/classifier.py
import os
import torch
import pandas as pd

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


class LesionDataset(Dataset):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.metadata = pd.read_csv("metadata.csv")
        for index, row in self.metadata.iterrows():
            # code smell, might change
            self.metadata.loc[index, "image_path"] = os.path.join(
                "data/lesions", f"{row['image_id']}.jpg"
            )

        labels = sorted(self.metadata["dx"].unique())
        self.label_to_int = {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        label = self.label_to_int[row["dx"]]
        # I believe all the images are RGB but ill leave the convert jic
        image = Image.open(row["image_path"]).convert("RGB")

        return self.transform(image), torch.tensor(label)


def train_classifier(model, dataloader, optimizer, loss_fn, device):
    running_loss = 0.0

    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # zero out gradients
        optimizer.zero_grad()
        # predict
        outputs = model(images)
        # compute loss and propagate
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate_classifier(model, dataloader, loss_fn, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader.dataset), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LesionDataset()

    # split dataset into training and validation sets
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # load pre-trained ResNet model and replace fc layer
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # gets number of inputs to fc layer
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 7)

    model.to(device)

    # TODO: should probably change the learning rate per layer, bit weird though
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 5

    print("Starting...")
    # TODO: should probably add some scheduler to prevent any overfitting
    for epoch in range(epochs):
        train_loss = train_classifier(model, train_loader, optimizer, loss_fn, device)
        val_loss, accuracy = evaluate_classifier(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, accuracy: {accuracy * 100.00}%"
        )

    torch.save(model.state_dict(), "models/classifier.pth")


if __name__ == "__main__":
    main()
