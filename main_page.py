import streamlit as st

st.set_page_config(page_title="pistol detector")
st.sidebar.header('Выберете страницу')

st.title("Добро пожаловать!")

st.subheader("Это приложение содержит yolo модель, натренированную на поиск пистолетов")

st.markdown(
    ":gray[**Навигация**]:\n"
    "1. *detect image* - поможет найти пистолет на картинке\n"
    "2. *detect image from webcam* - поможет найти пистолет на скриншоте с вашей камеры\n"
    "3. *detect video* - поможет найти пистолет на видео\n"
)
