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

    # отвечаем на вопрос о по нескольким картинкам сразу
    image_path1 = "example_docs/multipage_interest_free_loan_agreement/0/0.jpg"
    image_path2 = "example_docs/multipage_interest_free_loan_agreement/0/1.jpg"
    image_path3 = "example_docs/multipage_interest_free_loan_agreement/0/2.jpg"
    image_path4 = "example_docs/multipage_interest_free_loan_agreement/0/3.jpg"
    
    images=[image_path1, image_path2, image_path3, image_path4]
    
    question = (f"""Количество поданных страниц документов - {len(images)}.
Задача: Напиши, пожалуйста, кто и кому сколько денег занимает?""")
    
    print(question)
    
    model_answer = model.predict_on_images(
        images=images, question=question
    )
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
