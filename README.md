# HCL Traffic Sign Classifier

Streamlit app for traffic sign classification using a CNN model trained with PyTorch.

## Live App

[Open Streamlit App](https://hcl-project-lapptf8uyuupqtsmkwxn72a.streamlit.app/)

## Files

- `app.py` - Streamlit web app
- `trafficcnn.pth` - trained CNN weights
- `class_names.json` - class index to label mapping
- `requirements.txt` - Python dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Upload an image (`jpg`, `jpeg`, `png`, `bmp`, `webp`) to get prediction and confidence.
- The app uses the same CNN architecture and preprocessing used in training.
