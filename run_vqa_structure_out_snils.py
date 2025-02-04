import os
import subprocess

from model_interface.model_factory import ModelFactory


if __name__ == "__main__":

    cache_directory = "model_cache"

    # Сохраняем модели Qwen2-VL в примонтированную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Имена моделей и семейство моделей
    model_name_1 = "Qwen2.5-VL-7B-Instruct"
    model_family = "Qwen2.5-VL"

    # Инфо о том где взять класс для семейства моделей
    package = "model_qwen2_5_vl"
    module = "models"
    model_class = "Qwen2_5_VLModel"
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(model_family, model_class_path)

    # создаем модель
    model_init_params = {
        "model_name": model_name_1,
        "system_prompt": "",
        "cache_dir": "model_cache",
    }

    model = ModelFactory.get_model(model_family, model_init_params)

    # отвечаем на вопрос о по одной картинке
    image_path = "example_docs/snils/8.jpg"
    question = """
Подано изображение паспорта Российской Федерации.
Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.

Поля для извлечения:
- "country": Страна (например, "Российская Федерация")
- "type": "СТРАХОВОЕ СВИДЕТЕЛЬСТВО ОБЯЗАТЕЛЬНОГО ПЕНСИОННОГО СТРАХОВАНИЯ",
- "snils_number": "номер документа (в формате XXX-XXX-XXX XX)"
- "fio": Полное имя владельца (Фамилия Имя Отчество)
- "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
- "place_of_birth": Место рождения
- "gender": Пол ("Муж." или "Жен.")
- "registration_date": Дата регистрации (в формате DD.MM.YYYY)

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
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
