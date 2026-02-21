import json

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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg: #f3f8ff;
            --ink: #0b1728;
            --muted: #44607d;
            --panel: rgba(255, 255, 255, 0.75);
            --stroke: rgba(151, 179, 209, 0.45);
            --teal: #0fa3b1;
            --orange: #f4a261;
            --navy: #133c55;
            --shadow: 0 20px 48px rgba(18, 52, 88, 0.14);
        }

        .stApp {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 14% 12%, rgba(175, 232, 255, 0.6) 0%, transparent 30%),
                radial-gradient(circle at 90% 8%, rgba(255, 219, 170, 0.62) 0%, transparent 28%),
                radial-gradient(circle at 80% 94%, rgba(191, 229, 215, 0.55) 0%, transparent 26%),
                linear-gradient(135deg, #f5f9ff 0%, #eef5ff 52%, #f8fbff 100%);
        }

        .block-container {
            max-width: 1080px;
            padding-top: 1.35rem;
            padding-bottom: 2rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.35);
            background: linear-gradient(130deg, #0f2f4f 0%, #1f4f6f 52%, #0f8f96 100%);
            padding: 1.45rem 1.45rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 22px 46px rgba(11, 41, 74, 0.33);
            color: #f4fbff;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            top: -160px;
            right: -60px;
            background: rgba(255, 255, 255, 0.1);
        }

        .hero-kicker {
            display: inline-block;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.22);
            padding: 0.18rem 0.62rem;
            border-radius: 999px;
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .hero h1 {
            margin: 0.65rem 0 0;
            font-size: 2rem;
            line-height: 1.08;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: #ddf2ff;
            max-width: 750px;
        }

        .glass-card {
            background: var(--panel);
            backdrop-filter: blur(10px);
            border: 1px solid var(--stroke);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }

        .image-frame {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(129, 163, 197, 0.3);
        }

        .pred-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.2rem;
            font-weight: 700;
        }

        .pred-main {
            font-size: 1.55rem;
            line-height: 1.15;
            color: var(--navy);
            font-weight: 700;
            margin-bottom: 0.62rem;
        }

        .meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.7rem;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.32rem 0.78rem;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .pill-confidence {
            color: #f8feff;
            background: linear-gradient(90deg, var(--teal), #31b9cc);
        }

        .pill-count {
            color: #3c4f63;
            background: rgba(219, 233, 247, 0.95);
            border: 1px solid rgba(159, 182, 208, 0.6);
        }

        .rank-item {
            margin-bottom: 0.7rem;
        }

        .rank-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
            color: #20364d;
            gap: 0.6rem;
        }

        .bar-shell {
            height: 0.6rem;
            width: 100%;
            border-radius: 999px;
            background: rgba(171, 196, 223, 0.32);
            border: 1px solid rgba(167, 191, 218, 0.35);
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #1fb8c2 0%, #57c785 50%, #f4b860 100%);
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #87a8ca;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.7);
            transition: all 0.2s ease;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #1fa9b8;
            background: rgba(207, 243, 248, 0.72);
        }

        @media (max-width: 850px) {
            .hero h1 {
                font-size: 1.65rem;
            }

            .pred-main {
                font-size: 1.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_rank_row(rank: int, label: str, pct: float) -> None:
    st.markdown(
        f"""
        <div class="rank-item">
            <div class="rank-head">
                <span>{rank}. {label}</span>
                <strong>{pct:.2f}%</strong>
            </div>
            <div class="bar-shell">
                <div class="bar-fill" style="width: {pct:.2f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Traffic Sign Classifier", page_icon=":traffic_light:", layout="wide")
    inject_styles()

    st.markdown(
        """
        <section class="hero">
            <span class="hero-kicker">Computer Vision Demo</span>
            <h1>Traffic Sign Vision</h1>
            <p>Drop an image to get class prediction, confidence score, and ranked alternatives from your trained CNN.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, class_names = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Ensure trafficcnn.pth and class_names.json are in this folder.")
        return
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    with st.container(border=False):
        file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if not file:
        st.caption("Supported formats: JPG, JPEG, PNG, BMP, WEBP")
        return

    image = Image.open(file)
    pred, conf, probs = predict_image(image, model, class_names)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="pred-label">Predicted Class</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-main">{pred}</div>', unsafe_allow_html=True)
        st.markdown(
            (
                '<div class="meta-row">'
                f'<span class="pill pill-confidence">Confidence {conf * 100:.2f}%</span>'
                f'<span class="pill pill-count">{len(class_names)} Classes</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("**Top predictions**")
        topk = min(5, len(class_names))
        vals, inds = torch.topk(probs, k=topk)
        for rank, (v, i) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
            render_rank_row(rank, class_names[i], v * 100)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
