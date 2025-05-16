# Combined Streamlit App for Malaria Detection (CNN + YOLOv5)

import streamlit as st
import os
import sys
import cv2
import torch
import numpy as np
import pyrebase
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
import pathlib

st.set_page_config(page_title="Malaria Detection Combined", layout="wide")
# Firebase Config
firebase_config = {
    "apiKey": "AIzaSyAqOGt4r1PblULehQAAP2ZOZSCBKhm2fEI",
    "authDomain": "maleria-detection-app.firebaseapp.com",
    "databaseURL": "https://maleria-detection-app-default-rtdb.firebaseio.com",
    "projectId": "maleria-detection-app",
    "storageBucket": "maleria-detection-app.appspot.com",
    "messagingSenderId": "772397472827",
    "appId": "1:772397472827:web:b4611030c722b74a22d01e",
    "measurementId": "G-86B2V4ZZGR"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Load Models
cnn_model = load_model("better_cnn_model.keras")

# Load YOLOv5 Model
pathlib.PosixPath = pathlib.WindowsPath  # Windows patch
sys.path.insert(0, './yolov5')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

@st.cache_resource
def load_yolo():
    return DetectMultiBackend('malaria_best.pt', device='cpu')

yolo_model = load_yolo()

st.title("🦟 Malaria Detection System")

if "user" not in st.session_state:
    st.session_state.user = None

# Authentication
if st.session_state.user is None:
    choice = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.sidebar.button("Sign Up"):
            try:
                user = auth.create_user_with_email_and_password(email, password)
                st.success("Account created. Please log in.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    elif choice == "Login":
        if st.sidebar.button("Login"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.success("Logged in successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.success(f"Logged in as: {st.session_state.user['email']}")
    detection_type = st.radio("Select Detection Mode:", ["Single Cell (CNN)", "Multiple Cells (YOLOv5)"])
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if detection_type == "Single Cell (CNN)":
            if st.button("🔍 Detect with CNN"):
                img = image.resize((130, 130))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                pred = cnn_model.predict(img)[0][0]
                conf = round((pred if pred > 0.5 else 1 - pred) * 100, 2)
                result = "✅ Malaria Negative" if pred > 0.5 else "🧪 Malaria Positive"
                st.markdown(f"### Prediction: **{result}**")
                st.markdown(f"### Confidence: **{conf}%**")

        else:
            if st.button("🔍 Detect with YOLOv5"):
                img_path = Path("input.jpg")
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                im0 = cv2.imread(str(img_path))
                im = letterbox(im0, 640, stride=32, auto=True)[0]
                im = im.transpose((2, 0, 1))[::-1]
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).float() / 255.0
                im = im.unsqueeze(0) if im.ndimension() == 3 else im

                pred = yolo_model(im, augment=False, visualize=False)
                pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
                names = yolo_model.names
                for det in pred:
                    if len(det):
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in det:
                            label = f"{names[int(cls)]} {conf:.2f}"
                            xyxy = [int(x.item()) for x in xyxy]
                            color = (255, 0, 0) if int(cls) == 0 else (255, 255, 0)
                            cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                            cv2.putText(im0, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                output_path = "runs/detect/streamlit"
                os.makedirs(output_path, exist_ok=True)
                output_img_path = os.path.join(output_path, "output.jpg")
                cv2.imwrite(output_img_path, im0)
                st.image(output_img_path, caption="✅ Prediction Result", use_column_width=True)

    if st.button("🚪 Logout"):
        st.session_state.user = None
        st.success("Logged out successfully!")
        st.rerun()
