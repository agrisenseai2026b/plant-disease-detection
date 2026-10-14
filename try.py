# ============================================================
# PLANT DISEASE CLASSIFICATION USING RESNET18
# TRAINING + COMPLETE EVALUATION METRICS
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# KEEPING YOUR ORIGINAL PATHS EXACTLY AS THEY ARE
TRAIN_DIR = os.path.join(BASE_DIR, "data", "Train", "Train")
VAL_DIR = os.path.join(BASE_DIR, "data", "Validation", "Validation")

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "accuracy.pth"
)

CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "classes.json"
)

# Additional file for training history
HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
SEED = 2021


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 70)
print("          PLANT DISEASE CLASSIFICATION")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not Available - Using CPU")


# ============================================================
# 4. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 5. SET RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 6. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nTraining Dataset Path:")
print(TRAIN_DIR)

print("\nValidation Dataset Path:")
print(VAL_DIR)


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

classes = train_dataset.classes

num_classes = len(classes)

print("\nNumber of Training Images:", len(train_dataset))
print("Number of Validation Images:", len(val_dataset))
print("Number of Classes:", num_classes)

print("\nClasses:")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")


# ============================================================
# 9. CHECK CLASS CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("CLASS CONSISTENCY CHECK")
print("=" * 70)

if train_dataset.classes == val_dataset.classes:

    print("Training and Validation classes match: YES")

else:

    print("Training and Validation classes match: NO")

    print("\nTraining Classes:")
    print(train_dataset.classes)

    print("\nValidation Classes:")
    print(val_dataset.classes)


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

train_class_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes
)

for i, class_name in enumerate(classes):

    print(
        f"{class_name:<40} : "
        f"{train_class_counts[i]}"
    )


print("\n" + "=" * 70)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 70)

val_class_counts = np.bincount(
    val_dataset.targets,
    minlength=num_classes
)

for i, class_name in enumerate(classes):

    print(
        f"{class_name:<40} : "
        f"{val_class_counts[i]}"
    )


# ============================================================
# 11. SAVE CLASS NAMES
# ============================================================

with open(
    CLASSES_PATH,
    "w"
) as f:

    json.dump(
        classes,
        f,
        indent=4
    )

print("\nClasses saved to:")
print(CLASSES_PATH)


# ============================================================
# 12. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 13. DATASET / TRAINING SETTINGS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print("Image Size       :", IMAGE_SIZE)
print("Batch Size       :", BATCH_SIZE)
print("Epochs           :", EPOCHS)
print("Learning Rate    :", LEARNING_RATE)
print("Weight Decay     :", WEIGHT_DECAY)
print("Random Seed      :", SEED)


# ============================================================
# 14. LOAD PRETRAINED RESNET18
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 15. FREEZE RESNET BACKBONE
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# 16. REPLACE FINAL CLASSIFIER
# ============================================================

in_features = model.fc.in_features

model.fc = nn.Sequential(

    nn.Dropout(0.3),

    nn.Linear(
        in_features,
        num_classes
    )
)


# Move model to device
model = model.to(DEVICE)


print("\nModel: ResNet18")

print("Pretrained: YES")

print("Original Classes: 1000")

print("New Classes:", num_classes)

print("Backbone: FROZEN")

print("Final Classifier: TRAINABLE")


# ============================================================
# 17. PARAMETER INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

frozen_parameters = (
    total_parameters -
    trainable_parameters
)

print("\nTotal Parameters:",
      f"{total_parameters:,}")

print("Trainable Parameters:",
      f"{trainable_parameters:,}")

print("Frozen Parameters:",
      f"{frozen_parameters:,}")


# ============================================================
# 18. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 19. OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 20. LEARNING RATE SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.1
)


# ============================================================
# 21. TRAINING VARIABLES
# ============================================================

best_val_loss = float("inf")

best_val_accuracy = 0.0

best_epoch = 0

training_history = []


# ============================================================
# 22. TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Loss
        running_train_loss += (
            loss.item() * images.size(0)
        )


        # Predictions
        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_train_loss /
        len(train_dataset)
    )

    train_accuracy = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            # Forward pass
            outputs = model(images)


            # Validation loss
            loss = criterion(
                outputs,
                labels
            )


            running_val_loss += (
                loss.item() *
                images.size(0)
            )


            # Predictions
            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        running_val_loss /
        len(val_dataset)
    )

    val_accuracy = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    training_history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "validation_loss": val_loss,

        "validation_accuracy": val_accuracy,

        "learning_rate": current_lr

    })


    # --------------------------------------------------------
    # PRINT EPOCH RESULTS
    # --------------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss       : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss  : {val_loss:.4f}"
    )

    print(
        f"Validation Acc.  : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate    : "
        f"{current_lr:.8f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    classes,

                "image_size":
                    IMAGE_SIZE

            },

            MODEL_PATH

        )


        print(
            "✓ Best model saved!"
        )


# ============================================================
# 23. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    training_history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 24. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    "\nBest Epoch:",
    best_epoch
)

print(
    "Best Validation Loss:",
    f"{best_val_loss:.4f}"
)

print(
    "Best Validation Accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "\nModel saved at:"
)

print(MODEL_PATH)

print(
    "\nTraining history saved at:"
)

print(HISTORY_PATH)


# ============================================================
# 25. LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL FOR FINAL EVALUATION")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# 26. FINAL VALIDATION PREDICTIONS
# ============================================================

all_true_labels = []

all_pred_labels = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        all_true_labels.extend(
            labels.cpu().numpy()
        )

        all_pred_labels.extend(
            predicted.cpu().numpy()
        )


# Convert to NumPy
all_true_labels = np.array(
    all_true_labels
)

all_pred_labels = np.array(
    all_pred_labels
)


# ============================================================
# 27. ACCURACY
# ============================================================

final_accuracy = accuracy_score(
    all_true_labels,
    all_pred_labels
)


# ============================================================
# 28. PRECISION
# ============================================================

final_precision = precision_score(
    all_true_labels,
    all_pred_labels,
    average="weighted",
    zero_division=0
)


# ============================================================
# 29. RECALL
# ============================================================

final_recall = recall_score(
    all_true_labels,
    all_pred_labels,
    average="weighted",
    zero_division=0
)


# ============================================================
# 30. F1 SCORE
# ============================================================

final_f1 = f1_score(
    all_true_labels,
    all_pred_labels,
    average="weighted",
    zero_division=0
)


# ============================================================
# 31. FINAL OVERALL METRICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(
    f"\nAccuracy  : {final_accuracy:.4f}"
)

print(
    f"Accuracy  : {final_accuracy * 100:.2f}%"
)

print(
    f"\nPrecision : {final_precision:.4f}"
)

print(
    f"Precision : {final_precision * 100:.2f}%"
)

print(
    f"\nRecall    : {final_recall:.4f}"
)

print(
    f"Recall    : {final_recall * 100:.2f}%"
)

print(
    f"\nF1 Score  : {final_f1:.4f}"
)

print(
    f"F1 Score  : {final_f1 * 100:.2f}%"
)


# ============================================================
# 32. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    all_true_labels,

    all_pred_labels,

    target_names=classes,

    digits=4,

    zero_division=0
)


print("\n")

print(report)


# ============================================================
# 33. PER-CLASS METRICS
# ============================================================

report_dict = classification_report(

    all_true_labels,

    all_pred_labels,

    target_names=classes,

    output_dict=True,

    zero_division=0
)


per_class_data = []


for class_name in classes:

    per_class_data.append({

        "Class":
            class_name,

        "Precision":
            report_dict[class_name]["precision"],

        "Recall":
            report_dict[class_name]["recall"],

        "F1 Score":
            report_dict[class_name]["f1-score"],

        "Support":
            report_dict[class_name]["support"]

    })


per_class_df = pd.DataFrame(
    per_class_data
)


print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

print(
    per_class_df.to_string(
        index=False
    )
)


# ============================================================
# 34. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_true_labels,

    all_pred_labels
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n")

print(cm)


# ============================================================
# 35. LABELED CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=[
        f"Actual: {c}"
        for c in classes
    ],

    columns=[
        f"Predicted: {c}"
        for c in classes
    ]

)


print("\n" + "=" * 70)
print("LABELED CONFUSION MATRIX")
print("=" * 70)

print("\n")

print(cm_df.to_string())


# ============================================================
# 36. CORRECT / INCORRECT PREDICTIONS
# ============================================================

correct_predictions = np.sum(

    all_true_labels ==
    all_pred_labels

)

incorrect_predictions = np.sum(

    all_true_labels !=
    all_pred_labels

)

total_predictions = len(
    all_true_labels
)


correct_percentage = (

    correct_predictions /
    total_predictions
) * 100


incorrect_percentage = (

    incorrect_predictions /
    total_predictions
) * 100


print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    "\nTotal Predictions:",
    total_predictions
)

print(
    "Correct Predictions:",
    correct_predictions
)

print(
    "Incorrect Predictions:",
    incorrect_predictions
)

print(
    f"Correct Percentage: "
    f"{correct_percentage:.2f}%"
)

print(
    f"Incorrect Percentage: "
    f"{incorrect_percentage:.2f}%"
)


# ============================================================
# 37. MACRO METRICS
# ============================================================

macro_precision = precision_score(

    all_true_labels,

    all_pred_labels,

    average="macro",

    zero_division=0
)


macro_recall = recall_score(

    all_true_labels,

    all_pred_labels,

    average="macro",

    zero_division=0
)


macro_f1 = f1_score(

    all_true_labels,

    all_pred_labels,

    average="macro",

    zero_division=0
)


print("\n" + "=" * 70)
print("MACRO AVERAGE METRICS")
print("=" * 70)

print(
    f"\nMacro Precision : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall    : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1 Score  : "
    f"{macro_f1:.4f}"
)


# ============================================================
# 38. WEIGHTED METRICS
# ============================================================

print("\n" + "=" * 70)
print("WEIGHTED AVERAGE METRICS")
print("=" * 70)

print(
    f"\nWeighted Precision : "
    f"{final_precision:.4f}"
)

print(
    f"Weighted Recall    : "
    f"{final_recall:.4f}"
)

print(
    f"Weighted F1 Score  : "
    f"{final_f1:.4f}"
)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(
    f"""
Model                : ResNet18
Pretrained            : ImageNet
Number of Classes     : {num_classes}
Image Size            : {IMAGE_SIZE} x {IMAGE_SIZE}

Training Images       : {len(train_dataset)}
Validation Images     : {len(val_dataset)}

Best Epoch            : {best_epoch}
Best Validation Loss  : {best_val_loss:.4f}

Accuracy              : {final_accuracy * 100:.2f}%
Precision             : {final_precision * 100:.2f}%
Recall                : {final_recall * 100:.2f}%
F1 Score              : {final_f1 * 100:.2f}%

Macro Precision       : {macro_precision * 100:.2f}%
Macro Recall          : {macro_recall * 100:.2f}%
Macro F1 Score        : {macro_f1 * 100:.2f}%

Correct Predictions   : {correct_predictions}
Incorrect Predictions : {incorrect_predictions}

Model File            : {MODEL_PATH}
Classes File          : {CLASSES_PATH}
History File          : {HISTORY_PATH}
"""
)

print("=" * 70)
print("                 DONE")
print("=" * 70)