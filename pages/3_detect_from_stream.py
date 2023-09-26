import streamlit as st

from func_det import load_model, webcam_detect

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
st.text("Возьмите пистолет в руки перед камерой и модель его распознает")

model = load_model("models/yolov8n_v3.pt")

start_btn = st.button("Начать")

if start_btn:
    webcam_detect(model, confidence_slider)
