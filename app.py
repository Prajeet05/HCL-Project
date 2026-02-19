import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class TrafficCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


@st.cache_resource
def load_artifacts():
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_map = json.load(f)

    class_names = [class_map[str(i)] for i in range(len(class_map))]
    model = TrafficCNN(num_classes=len(class_names))
    state = torch.load("trafficcnn.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names


def predict_image(image: Image.Image, model: nn.Module, class_names: list[str]):
    x = get_transform()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    return class_names[idx.item()], float(conf.item()), probs


def main():
    st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
    st.title("Traffic Sign Classifier")
    st.write("Upload an image and the CNN model will predict its traffic sign class.")

    try:
        model, class_names = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Ensure trafficcnn.pth and class_names.json are in this folder.")
        return
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not file:
        return

    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    pred, conf, probs = predict_image(image, model, class_names)
    st.success(f"Prediction: **{pred}**")
    st.info(f"Confidence: **{conf * 100:.2f}%**")

    topk = min(5, len(class_names))
    vals, inds = torch.topk(probs, k=topk)
    st.subheader("Top predictions")
    for rank, (v, i) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
        st.write(f"{rank}. {class_names[i]} ({v * 100:.2f}%)")


if __name__ == "__main__":
    main()
