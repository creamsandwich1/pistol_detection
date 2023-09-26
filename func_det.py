import streamlit as st
import cv2

from PIL import Image
from ultralytics import YOLO


def load_image() -> Image.Image:
    """
    Функция загрузки изображения

    :return: картинка PIL
    """
    uploaded_file = st.file_uploader(label="выберите изображение")
    if uploaded_file is not None:
        img_data = uploaded_file
        return Image.open(img_data)


def load_model(weights_path: str) -> YOLO:
    """
    Загружает модель по выбранному пути

    :param weights_path: Путь к обученной модели
    :return: модель YOLO
    """
    model = YOLO(weights_path)
    return model


def make_prediction(model: YOLO, img: Image.Image, conf=0.6) -> list:
    """
    Загружает изображение в модель и делает predict

    :param model: YOLO модель для предикта
    :param img: PIL картинка для предикта
    :param conf: уверенность модели
    :return: Список list с результатами распознавания
    """
    result = model(img, conf=conf)
    return result


def show_pred_image(prediction: list) -> None:
    """
    Выводит картинку с предиктом класса gun

    :param prediction: Список list с результатами распознавания
    :return: None
    """
    res_plotted = prediction[0].plot()[:, :, ::-1]
    st.image(res_plotted)


def webcam_detect(model: YOLO, conf: float) -> None:
    """
    Считывает кадры с вебкамеры, выводит картинку с предиктами

    :param model: YOLO модель для предикта
    :param conf: уверенность модели
    :return: None
    """
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    stop_button = st.button("Остановить запись")

    while cap.isOpened() and not stop_button:

        ret, frame = cap.read()

        if not ret:
            st.write("Запись окончена")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame, conf=conf)
        predict_frame = results[0].plot()

        frame_placeholder.image(predict_frame, channels="RGB")
