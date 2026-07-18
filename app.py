import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -------------------- Page --------------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("😷 Face Mask Detection")

# -------------------- Session State --------------------
if "open_camera" not in st.session_state:
    st.session_state["open_camera"] = False

# -------------------- Check Model --------------------
MODEL_PATH = "mask_final.keras"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"Model Loading Error:\n{e}")
    st.stop()

# -------------------- Prediction Function --------------------
def predict_mask(img):

    img = img.convert("RGB")
    img = img.resize((128, 128))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    prob = prediction[0][0]

    if prob > 0.5:
        st.error(f"❌ WITHOUT MASK ({prob:.2%})")
    else:
        st.success(f"✅ WITH MASK ({(1-prob):.2%})")

# -------------------- Upload Image --------------------
uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    predict_mask(img)

st.divider()

# -------------------- Camera --------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state["open_camera"] = True

with col2:
    if st.button("❌ Close Camera"):
        st.session_state["open_camera"] = False

if st.session_state.get("open_camera", False):

    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:

        img = Image.open(camera_image)

        st.image(img, caption="Captured Image", use_container_width=True)

        predict_mask(img)

        st.session_state["open_camera"] = False