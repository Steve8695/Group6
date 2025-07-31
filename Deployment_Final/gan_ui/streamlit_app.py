import streamlit as st
import requests
from PIL import Image
import io
import os
import numpy as np

import torch
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import yaml

# Load backend config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
backend_conf = config["backend"]
BACKEND_URL = f"http://{backend_conf['host']}:{backend_conf['port']}"

st.title("GAN Image Generator")

# --- Seed control
st.subheader("Image Generation")
seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=np.random.randint(0, 999999))
if st.button("Generate Image"):
    gen_response = requests.post(f"{BACKEND_URL}/generate", json={"seed": int(seed)})
    if gen_response.status_code == 200:
        img_response = requests.get(f"{BACKEND_URL}/image")
        if img_response.status_code == 200:
            img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
            st.image(img, caption=f"Generated Image (Seed: {seed})", use_column_width=True)

            # ---- LIME Visualization ----
            st.subheader("LIME Explanation")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            def batch_transform(imgs):
                return torch.cat([transform(Image.fromarray(i)).unsqueeze(0) for i in imgs], 0)

            classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            classifier.eval()

            def predict_fn(images):
                with torch.no_grad():
                    batch = batch_transform(images)
                    logits = classifier(batch)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                return probs.cpu().numpy()

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                np.array(img.resize((224, 224))),
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000,
            )
            lime_img, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            st.image(mark_boundaries(lime_img, mask), caption="LIME Explanation", use_column_width=True)
        else:
            st.error("Image not found. Try again.")
    else:
        st.error("Image generation failed. Check backend.")

st.caption("UI and interpretability")
