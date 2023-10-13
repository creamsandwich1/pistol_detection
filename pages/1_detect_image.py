import streamlit as st

from func_det import load_model, load_image, show_pred_image, save_image, download_prediction

st.sidebar.header('Настройки:')

confidence_slider = st.sidebar.slider(
    "Порог уверенности для детекции",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.1
)

st.sidebar.write('---')

st.sidebar.header("Превью:")

st.title("Распознавание пистолета на фото")
st.text("Загрузите изображение с пистолетом и нажмите кнопку \"Распознать\"")

model = load_model("models/yolov8n_v3.pt")
image = load_image()

if image is not None:
    st.sidebar.image(image)
else:
    st.sidebar.text("Загрузите изображение\n"
                    "для предпросмотра")

detect_button = st.button("Распознать")
if detect_button and image is not None:
    with st.spinner("Подождите"):
        pred = model(image, conf=confidence_slider)
        show_pred_image(pred)
        save_image(pred)
        download_prediction("temp/images/output.png", "your_detection.png")
