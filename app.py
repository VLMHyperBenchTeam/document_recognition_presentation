import streamlit as st
import os
import subprocess
from model_interface.model_factory import ModelFactory

# Функция для загрузки и инициализации модели
@st.cache_resource
def load_model():
    cache_directory = "model_cache"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)
    model_name_1 = "Qwen2.5-VL-7B-Instruct"
    model_family = "Qwen2.5-VL"
    package = "model_qwen2_5_vl"
    module = "models"
    model_class = "Qwen2_5_VLModel"
    model_class_path = f"{package}.{module}:{model_class}"
    ModelFactory.register_model(model_family, model_class_path)
    model_init_params = {
        "model_name": model_name_1,
        "system_prompt": "",
        "cache_dir": cache_directory,
    }
    model = ModelFactory.get_model(model_family, model_init_params)
    return model

# Загрузка модели
model = load_model()

# Создание вкладок
tabs = [
    "Документ классификация",
    "Сортировка страниц",
    "VQA многостраничный",
    "VQA",
    "Структурированный вывод INN",
    "Структурированный вывод Паспорт",
    "Структурированный вывод СНИЛС"
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tabs)

# Вкладка "Документ классификация"
with tab1:
    st.header("Классификация документов")
    uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, key="doc_classification_uploader")
    if st.button("Классифицировать", key="doc_classification_button"):
        if uploaded_files:
            image_paths = [os.path.join("temp", f.name) for f in uploaded_files]
            for file, path in zip(uploaded_files, image_paths):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
            
            question = (f"""Количество поданных страниц документов - {len(image_paths)}.
                Задача: Определите тип каждого документа на предоставленных изображениях и выведите их в виде последовательности цифр, где каждая цифра соответствует определенному типу документа. Ответ должен содержать только порядок цифр, без дополнительного текста.
                Типы документов:
                1 - old_tins: свидетельство о постановке на учет физического лица (документ желтого цвета).
                2 - new_tins: свидетельство о постановке на учет физического лица
                3 - interest_free_loan_agreement: договор беспроцентном займа
                4 - snils: страховое свидетельство обязательного пенсионного страхования (документ зеленого цвета).
                5 - invoice: счет фактура
                6 - passport: паспорт Российской федерации
                Пример ответа: 2,4,5,1,3
                Пожалуйста, предоставьте ответ в указанном формате.""")

            st.write(image_paths)
            model_answer = model.predict_on_images(images=image_paths, question=question)
            st.write(model_answer)
            
            # Словарь для текстового описания классов
            class_descriptions = {
                "1": "ИНН (старого образца)",
                "2": "ИНН (нового образца)",
                "3": "Договор беспроцентном займе",
                "4": "СНИЛС",
                "5": "Счет фактура",
                "6": "Паспорт РФ"
            }
            
            # Преобразование ответа модели в список классов
            class_list = model_answer.strip().split(',')
            
            # Проверка, что количество классов совпадает с количеством изображений
            if len(class_list) == len(image_paths):
                for idx, class_id in enumerate(class_list):
                    image_path = image_paths[idx]
                    class_description = class_descriptions.get(class_id, "Неизвестный класс")
                    st.image(image_path, caption=f"Класс документа: {class_id} ({class_description})", use_container_width=True)
            else:
                st.error("Количество классов в ответе модели не совпадает с количеством загруженных изображений.")
            
            subprocess.run(["nvidia-smi"])

# Вкладка "Сортировка страниц"
with tab2:
    st.header("Сортировка страниц")
    uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, key="pages_sorting_uploader")
    if uploaded_files:
        image_paths = [os.path.join("temp", f.name) for f in uploaded_files]
        for file, path in zip(uploaded_files, image_paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            st.image(path, caption=f"Загруженное изображение: {file.name}", use_container_width=True)
    if st.button("Отсортировать", key="pages_sorting_button"):
        if uploaded_files:
            question = (f"""Перед вами {len(image_paths)} изображений страниц одного типа документа, которые находятся в хаотичном порядке.
            Анализируя содержимое предоставленных страниц документа, определите логический порядок страниц и выведите их в виде цифр через запятую.
            Страницы содержат различные разделы договора о беспроцентном займе, включая условия займа, порядок передачи и возврата суммы займа, ответственность сторон, форс-мажорные обстоятельства, разрешение споров, изменения и досрочное расторжение договора, а также заключительные положения.
            Пожалуйста, проанализируйте текст на каждой странице и укажите правильный порядок только в виде порядка страниц через запятую.""")
            
            model_answer = model.predict_on_images(images=image_paths, question=question)
            st.write(model_answer)
            subprocess.run(["nvidia-smi"])

# Вкладка "VQA многостраничный"
with tab3:
    st.header("VQA многостраничный")
    uploaded_files = st.file_uploader("Загрузите изображения", accept_multiple_files=True, key="vqa_multi_pages_uploader")
    user_question = st.text_area("Введите ваш вопрос", value="Напиши, пожалуйста, кто и кому сколько денег занимает?", key="vqa_multi_pages_question")
    if uploaded_files:
        image_paths = [os.path.join("temp", f.name) for f in uploaded_files]
        for file, path in zip(uploaded_files, image_paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            st.image(path, caption=f"Загруженное изображение: {file.name}", use_container_width=True)
    if st.button("Получить информацию", key="vqa_multi_pages_button"):
        if uploaded_files:
            additional_prompt = f"Количество поданных страниц документов - {len(image_paths)}."
            question = additional_prompt + user_question
            
            model_answer = model.predict_on_images(images=image_paths, question=question)
            st.write(model_answer)
            subprocess.run(["nvidia-smi"])

# Вкладка "VQA"
with tab4:
    st.header("VQA")
    uploaded_file = st.file_uploader("Загрузите изображение", accept_multiple_files=False, key="vqa_uploader")
    user_question = st.text_area("Введите ваш вопрос", value="Опиши документ.", key="vqa_question")
    if st.button("Описать документ", key="vqa_button"):
        if uploaded_file:
            image_path = os.path.join("temp", uploaded_file.name)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(image_path, caption="Загруженное изображение", use_container_width=True)
            question = user_question
            st.write(question)
            
            model_answer = model.predict_on_image(image=image_path, question=question)
            st.write(model_answer)
            subprocess.run(["nvidia-smi"])

# Вкладка "Структурированный вывод INN"
with tab5:
    st.header("Структурированный вывод INN")
    uploaded_file = st.file_uploader("Загрузите изображение", accept_multiple_files=False, key="structure_out_inn_uploader")
    if uploaded_file:
        image_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Загруженное изображение", use_container_width=True)
    if st.button("Извлечь информацию", key="structure_out_inn_button"):
        if uploaded_file:
            question = """
            Подано изображение свидетельства о постановке на учет физического лица в налоговом органе.
            Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.
            Поля для извлечения:
            "type": "Свидетельство о постановке на учет физического лица в налоговом органе"
            "issued_by": "Федеральная налоговая служба"
            "date_of_issue": Дата выдачи (в формате DD.MM.YYYY)
            "fio": Полное имя владельца (Фамилия Имя Отчество)
            "gender": Пол ("Муж." или "Жен.")
            "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
            "registration_date": "дата регистрации" (в формате DD.MM.YYYY)
            "inn_number": "ИНН"
            "signature": "подпись",
            "office": "должность подписавшего лица",
            "form_number": "номер формы",
            "code": "код"
            JSON-структура:
            {
            "type": "",
            "issued_by": "",
            "date_of_issue": "",
            "fio": "",
            "gender": "",
            "date_of_birth": "",
            "registration_date": "",
            "inn_number": "",
            "signature": "",
            "office": "",
            "form_number": "",
            "code": ""
            }
            Используйте данные из изображения для заполнения полей JSON.
            """
            
            model_answer = model.predict_on_image(image=image_path, question=question)
            st.write(model_answer)
            subprocess.run(["nvidia-smi"])

# Вкладка "Структурированный вывод Паспорт"
with tab6:
    st.header("Структурированный вывод Паспорт")
    uploaded_file = st.file_uploader("Загрузите изображение", accept_multiple_files=False, key="structure_out_passport_uploader")
    if uploaded_file:
        image_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Загруженное изображение", use_container_width=True)
    if st.button("Извлечь информацию", key="structure_out_passport_button"):
        if uploaded_file:
            question = """
            Подано изображение паспорта Российской Федерации.
            Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.
            Поля для извлечения:
            "country": Страна (например, "Российская Федерация")
            "issuing_authority": Орган, выдавший паспорт (например, "Управление МВД России по г. Кашира Московская область")
            "date_of_issue": Дата выдачи паспорта (в формате DD.MM.YYYY)
            "document_number": Номер документа
            "fio": Полное имя владельца паспорта (Фамилия Имя Отчество)
            "gender": Пол ("Муж." или "Жен.")
            "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
            "place_of_birth": Место рождения
            "passport_image": Присутствует ли изображение паспорта ("1" или "0")
            JSON-структура:
            {
            "country": "",
            "issuing_authority": "",
            "date_of_issue": "",
            "document_number": "",
            "fio": "",
            "gender": "",
            "date_of_birth": "",
            "place_of_birth": "",
            "passport_image": ""
            }
            """
            
            model_answer = model.predict_on_image(image=image_path, question=question) 
            st.write(model_answer)
            subprocess.run(["nvidia-smi"])

# Вкладка "Структурированный вывод СНИЛС"
with tab7:
    st.header("Структурированный вывод СНИЛС")
    uploaded_file = st.file_uploader("Загрузите изображение", accept_multiple_files=False, key="structure_out_snils_uploader")
    if uploaded_file:
        image_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Загруженное изображение", use_container_width=True)
    if st.button("Извлечь информацию", key="structure_out_snils_button"):
        if uploaded_file:
            question = """
            Подано изображение паспорта Российской Федерации.
            Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.
            Поля для извлечения:
            "country": Страна (например, "Российская Федерация")
            "type": "СТРАХОВОЕ СВИДЕТЕЛЬСТВО ОБЯЗАТЕЛЬНОГО ПЕНСИОННОГО СТРАХОВАНИЯ",
            "snils_number": "номер документа (в формате XXX-XXX-XXX XX)"
            "fio": Полное имя владельца (Фамилия Имя Отчество)
            "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
            "place_of_birth": Место рождения
            "gender": Пол ("Муж." или "Жен.")
            "registration_date": Дата регистрации (в формате DD.MM.YYYY)
            JSON-структура:
            {
            "country": "",
            "type": "",
            "snils_number": "",
            "fio": "",
            "date_of_birth": "",
            "place_of_birth": "",
            "gender": "",
            "registration_date": ""
            }
            """
            
            model_answer = model.predict_on_image(image=image_path, question=question)
            st.write(model_answer)
            subprocess.run(["nvidia-smi"])