import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.dataset import NYUDepthDataset
from src.models import get_depth_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

csv_path = Path("../data/nyu2_train_subset.csv")

train_geom_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
])

val_geom_transform = v2.Compose([
    v2.Resize((256, 256)),
])

train_img_transform = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

depth_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

dataset = NYUDepthDataset(csv_path, )

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.geom_transform = train_geom_transform
train_dataset.dataset.img_transform = train_img_transform
train_dataset.dataset.depth_transform = depth_transform

val_dataset.dataset.geom_transform = val_geom_transform
val_dataset.dataset.img_transform = val_img_transform
val_dataset.dataset.depth_transform = depth_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = get_depth_model(pretrained=True).to(device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

n_epochs = 15

for param in model.parameters():
    param.requires_grad = True

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for images, depths in train_loader:
        images, depths = images.to(device), depths.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, depths)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for images, depths in val_loader:
            images, depths = images.to(device), depths.to(device)
            preds = model(images)
            loss = loss_fn(preds, depths)

            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{n_epochs}] | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}", flush=True)