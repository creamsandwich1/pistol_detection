import streamlit as st
import imageio
import numpy as np
import os

from PIL import Image
from ultralytics import YOLO


def load_image() -> Image.Image:
    """
    Функция загрузки изображения

    :return: картинка PIL
    """
    uploaded_file = st.file_uploader(
        label="выберите изображение",
        type=["bmp", "jpg", "png", "jpeg", "dng", "mpo", "tif", "tiff", "webp", "pfm"]
        )

    if uploaded_file is not None:
        img_data = uploaded_file
        return Image.open(img_data)


def save_image(prediction: list):
    # Создает папку если ее нет
    if not os.path.exists('temp/images'):
        os.makedirs('temp/images')

    image = Image.fromarray(prediction[0].plot()[..., ::-1])
    image.save("temp/images/output.png")


def load_video():
    """
    Загружает входящее видео через st.file_uploader

    :return: st.file_uploader.UploadedFile
    """
    uploaded_file = st.file_uploader(label="выберите видео", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        video_data = uploaded_file
        return video_data


def save_video(uploaded_video) -> None:
    """
    Сохраняет загруженное видео как input.mp4

    :param uploaded_video: объект из st.file_uploader для записи
    :return: None
    """
    # Создает папку если ее нет
    if not os.path.exists('temp/videos'):
        os.makedirs('temp/videos')

    # Записывает файл на диск
    with open(os.path.join('temp', "videos", "input.mp4"), 'wb') as f:
        f.write(uploaded_video.getbuffer())


def frames2video(out) -> None:
    """
    Переводит предикт модели в видео mp4 и сохраняет его как output

    :param out: Результат распознования нейросетью
    :return: None
    """
    # запись серии картинок в видео
    images = np.stack([frame.plot()[..., ::-1] for frame in out], axis=0)

    # Create a writer object
    writer = imageio.get_writer('temp/videos/output.mp4', fps=30)

    # Loop over your images and add each one to the video
    for image in images:
        writer.append_data(image)

    # Close the writer
    writer.close()


def load_model(weights_path: str) -> YOLO:
    """
    Загружает модель по выбранному пути

    :param weights_path: Путь к обученной модели
    :return: модель YOLO
    """
    model = YOLO(weights_path)
    return model


def show_pred_image(prediction: list) -> None:
    """
    Выводит картинку с предиктом класса gun

    :param prediction: Список list с результатами распознавания
    :return: None
    """
    res_plotted = prediction[0].plot()[:, :, ::-1]
    st.image(res_plotted)


def download_prediction(file_path: str, output_name: str) -> None:
    """
    Позволяет скачать результат

    :param file_path: Путь до скаченного файла
    :param output_name: Имя скаченного файла с форматом
    :return:
    """
    with open(file_path, "rb") as file:
        st.download_button(
            label="Скачать",
            data=file,
            file_name=output_name,
        )
