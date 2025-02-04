# Презентация возможностей по работе с документам с помощью VLM

Данный репозиторий содержит Jupyter Notebook c демонстрацией возможностей VLM.

# Docker контейнер модели

Поддерживаются модели семейства Qwen2.5-VL:

* Qwen2.5-VL-3B-Instruct
* Qwen2.5-VL-7B-Instruct

## Build Docker image

Для сборки `Docker image` выполним команду:
```
docker build -t qwen2.5-vl:ubuntu22.04-cu124-torch2.4.0_demo_v0.1.0 -f docker/Dockerfile-cu124 .
```

## Run Docker Container

Для запуска `Docker Container` выполним команду:
```
docker run \
    --gpus all \
    -it \
    -v .:/workspace \
    qwen2.5-vl:ubuntu22.04-cu124-torch2.4.0_demo_v0.1.0 sh
```

Нам откроется терминал внутри `Docker Container`.

Для запуска предсказаний выполним в нем команду:
```
cd cd workspace
python run_predict.py
```