import os
import json
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "Train",
    "Train"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "data",
    "Validation",
    "Validation"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "plant_disease_model.pth"
)

CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "classes.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

SEED = 2021

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

print("\nLoading dataset...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    VALIDATION_DIR,
    transform=validation_transform
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)

print(
    "Classes:",
    train_dataset.classes
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

with open(
    CLASSES_PATH,
    "w"
) as file:

    json.dump(
        train_dataset.classes,
        file
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating model...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze backbone
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace classifier
model.fc = nn.Sequential(

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        model.fc.in_features,
        len(train_dataset.classes)
    )
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.1
)


# ============================================================
# TRAINING
# ============================================================

best_validation_loss = float("inf")


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= train_total

    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += labels.size(0)


    validation_loss /= validation_total

    validation_accuracy = (
        validation_correct
        / validation_total
    )


    scheduler.step(
        validation_loss
    )


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        "Train Loss:",
        round(train_loss, 4)
    )

    print(
        "Train Accuracy:",
        round(train_accuracy * 100, 2),
        "%"
    )

    print(
        "Validation Loss:",
        round(validation_loss, 4)
    )

    print(
        "Validation Accuracy:",
        round(validation_accuracy * 100, 2),
        "%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "image_size": IMAGE_SIZE
            },
            MODEL_PATH
        )

        print(
            "Best model saved!"
        )


print("\n================================")
print("Training completed!")
print("Model:", MODEL_PATH)
print("Classes:", CLASSES_PATH)
print("================================")