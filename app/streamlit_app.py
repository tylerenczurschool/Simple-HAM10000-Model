import streamlit as st
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models

# load model
model_state_dict = torch.load("models/classifier.pth")

# create model and replace fully connected layer
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 7)

model.load_state_dict(model_state_dict)
model.eval()

# (for future reference) values from ImageNet dataset
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image):
    image_tensor = transform(image)
    # model only takes batches
    image_tensor.unsqueeze_(0)
    output = model(image_tensor)
    _, prediction = torch.max(output.data, 1)
    return prediction.item()


st.title("Skin Lesion Classifier")

uploaded_file = st.file_uploader(
    "Upload an image of a skin lesion", type=["jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    prediction = predict(image)

    # load labels mapping from metadata.csv
    df = pd.read_csv("metadata.csv")
    labels = sorted(df["dx"].unique())

    # map predicted index to actual label
    predicted_label = labels[prediction]

    st.write(f"Predicted Lesion Type: {predicted_label}")
