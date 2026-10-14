import os
import json

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

image_size = checkpoint.get(
    "image_size",
    128
)


# ============================================================
# CREATE MODEL
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        model.fc.in_features,
        len(classes)
    )
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (image_size, image_size)
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
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Healthy": {

        "title": "Healthy Plant",

        "description":
            "The uploaded leaf appears healthy with no strong visual signs of the diseases covered by this model.",

        "recommendation":
            "Continue regular monitoring, proper irrigation, balanced nutrition and preventive crop management."

    },

    "Powdery": {

        "title": "Powdery Mildew",

        "description":
            "The model detected visual characteristics associated with powdery mildew.",

        "recommendation":
            "Inspect affected leaves, improve airflow around plants and follow locally recommended disease-management practices."
    },

    "Rust": {

        "title": "Rust Disease",

        "description":
            "The model detected visual characteristics associated with rust disease.",

        "recommendation":
            "Inspect the crop for rust symptoms, remove severely affected material where appropriate and follow locally recommended disease-management practices."
    }
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        DEVICE
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, index = torch.max(
            probabilities,
            dim=1
        )


    predicted_class = classes[
        index.item()
    ]

    confidence_percentage = (
        confidence.item() * 100
    )


    info = DISEASE_INFO.get(
        predicted_class,
        {}
    )


    return {

        "success": True,

        "prediction":
            predicted_class,

        "confidence":
            round(
                confidence_percentage,
                2
            ),

        "title":
            info.get(
                "title",
                predicted_class
            ),

        "description":
            info.get(
                "description",
                "Disease classification completed."
            ),

        "recommendation":
            info.get(
                "recommendation",
                "Monitor the plant carefully."
            )
    }