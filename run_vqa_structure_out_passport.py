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
    image_path = "example_docs/passport/9.jpg"
    question = """
Подано изображение паспорта Российской Федерации.
Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.

Поля для извлечения:
- "country": Страна (например, "Российская Федерация")
- "issuing_authority": Орган, выдавший паспорт (например, "Управление МВД России по г. Кашира Московская область")
- "date_of_issue": Дата выдачи паспорта (в формате DD.MM.YYYY)
- "document_number": Номер документа
- "fio": Полное имя владельца паспорта (Фамилия Имя Отчество)
- "gender": Пол ("Муж." или "Жен.")
- "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
- "place_of_birth": Место рождения
- "passport_image": Присутствует ли изображение паспорта ("1" или "0")

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
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
