import streamlit as st
import os

from func_det import load_model, load_video, save_video, frames2video, download_prediction

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

st.title("Распознавание пистолета на видео")
st.text("Загрузите видео с пистолетом и нажмите кнопку \"Распознать\"")

model = load_model("models/yolov8n_v3.pt")
video_file = load_video()

if video_file is not None:
    save_video(video_file)
    st.sidebar.video(os.path.join('temp', "videos", "input.mp4"))
else:
    st.sidebar.text("Загрузите видео\n"
                    "для предпросмотра")

detect_button = st.button("Распознать")
results = None
if detect_button and video_file is not None:
    with st.spinner("Подождите, время ожидания зависит от длины видео"):
        results = model(os.path.join('temp', "videos", "input.mp4"), conf=confidence_slider, stream=True)
        frames2video(results)

    st.video("temp/videos/output.mp4")
    download_prediction("temp/videos/output.mp4", "your_detection.mp4")
