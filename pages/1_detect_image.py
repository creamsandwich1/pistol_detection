import streamlit as st

from PIL import UnidentifiedImageError
from func_det import load_model, load_image, make_prediction, show_pred_image


st.sidebar.header('Настройки')

confidence_slider = st.sidebar.slider(
    "Порог уверенности для детекции",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.1
)

st.sidebar.write('---')

st.title("Распознавание пистолета на фото")
st.text("Загрузите изображение с пистолетом и нажмите кнопку \"Распознать\"")

model = load_model("models/yolov8n_v3.pt")

image = None
try:
    image = load_image()
except UnidentifiedImageError:
    st.text("Неверный формат изображения")

detect_button = st.button("Распознать")
if detect_button and image is not None:
    with st.spinner("Подождите"):
        pred = make_prediction(model, image, conf=confidence_slider)
        show_pred_image(pred)
