import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -------------------- Page --------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown(
    """
    <style>

    .main-title {
        text-align:center;
        font-size:45px;
        font-weight:bold;
        background: linear-gradient(90deg,#00c6ff,#0072ff);
        -webkit-background-clip:text;
        color:transparent;
    }

    .sub-title {
        text-align:center;
        font-size:18px;
        color:gray;
    }

    .card {
        padding:20px;
        border-radius:15px;
        background-color:#f8f9fa;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        margin:10px;
    }

    .footer {
        text-align:center;
        color:gray;
        font-size:14px;
        margin-top:30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- Header --------------------

st.markdown(
    "<div class='main-title'>😷 Face Mask Detection</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-title'>
    AI Based Face Mask Detection System using Deep Learning & CNN
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# -------------------- Sidebar --------------------

st.sidebar.title("⚙️ About Project")

st.sidebar.info(
    """
😷 Face Mask Detection

Technology Used:

✅ TensorFlow  
✅ CNN Deep Learning  
✅ Streamlit  
✅ Image Processing  

Features:

📂 Image Detection  
📸 Live Camera Detection
"""
)


# -------------------- Session State --------------------

if "open_camera" not in st.session_state:
    st.session_state["open_camera"] = False


# -------------------- Model Loading --------------------

MODEL_PATH = "mask_final.keras"


if not os.path.exists(MODEL_PATH):

    st.error("❌ Model file not found!")
    st.stop()


try:

    model = load_model(MODEL_PATH)

    st.success("✅ AI Model Loaded Successfully")

except Exception as e:

    st.error(f"Model Loading Error:\n{e}")
    st.stop()



# -------------------- Prediction Function --------------------

def predict_mask(img):

    img = img.convert("RGB")
    img = img.resize((128,128))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array,verbose=0)

    prob = prediction[0][0]


    st.write("")


    if prob > 0.5:

        st.markdown(
            """
            <div class='card'>
            <h2 style='color:red;text-align:center;'>
            ❌ WITHOUT MASK
            </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.error(f"Confidence: {prob:.2%}")


    else:

        st.markdown(
            """
            <div class='card'>
            <h2 style='color:green;text-align:center;'>
            ✅ WITH MASK
            </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"Confidence: {(1-prob):.2%}")



# -------------------- Upload Image --------------------

st.subheader("📂 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)


if uploaded_file:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    predict_mask(img)



st.divider()


# -------------------- Camera Section --------------------

st.subheader("📸 Live Camera Detection")


col1,col2 = st.columns(2)


with col1:

    if st.button("📸 Open Camera",
                 use_container_width=True):

        st.session_state["open_camera"]=True


with col2:

    if st.button("❌ Close Camera",
                 use_container_width=True):

        st.session_state["open_camera"]=False



if st.session_state["open_camera"]:


    camera_image = st.camera_input(
        "Capture Face Image"
    )


    if camera_image:

        img = Image.open(camera_image)

        st.image(
            img,
            caption="Captured Image",
            use_container_width=True
        )


        predict_mask(img)

        st.session_state["open_camera"]=False



# -------------------- Footer --------------------

st.markdown(
    """
    <div class='footer'>
    🚀 Developed using TensorFlow + Streamlit <br>
    😷 Face Mask Detection Project
    </div>
    """,
    unsafe_allow_html=True
)
