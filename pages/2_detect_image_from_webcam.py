import streamlit as st

from PIL import Image
from func_det import load_model, show_pred_image,  save_image, download_prediction

st.sidebar.header('Настройки')

confidence_slider = st.sidebar.slider(
    "Порог уверенности для детекции",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.1
)

st.sidebar.write('---')

st.title("Распознавание пистолета с камеры")
st.text("Сделайте скриншот с пистолетом в кадре и модель его распознает")

st_image = st.camera_input("Ваша камера")
result_pil_image = None

model = load_model("models/yolov8n_v3.pt")

if st_image:
    pil_image = Image.open(st_image)
    result_pil_image = model(pil_image, conf=confidence_slider)
    show_pred_image(result_pil_image)
    save_image(result_pil_image)
    download_prediction("temp/images/output.png", "your_detection.png")
