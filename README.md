# Gesture Framework

Python-библиотека для распознавания жестов руки и (распознования голоса), управления курсором и обучения пользовательских жестов.

В репозитории есть два основных сценария:

- библиотека `gesture_framework` для запуска распознавания и привязки жестов к действиям;
- GUI-приложение `gesture_builder_gui.py` для записи и обучения собственных жестов.

## Возможности

- распознавание базовых жестов руки;
- привязка жестов к действиям мыши и клавиатуры;
- поддержка пользовательских жестов через SVM;
- GUI для записи датасета и обучения модели;
- конфигурация через `config.json`;
- В будущем голосовых команд и мультимодального режима.

## Структура проекта

```text
.
|-- config.json                    # Основная конфигурация
|-- example_custom_click.py        # Пример запуска пользовательского жеста
|-- gesture_builder_gui.py         # GUI для записи и обучения жестов
|-- hand_landmarker.task           # Модель MediaPipe для детекции руки
|-- requirements.txt               # Базовые зависимости
|-- custom_gesture/
|   `-- my_click.pkl               # Пример обученного пользовательского жеста
`-- gesture_framework/
    |-- app.py                     # Основной API приложения
    |-- core/                      # Платформенный слой и мультимодальная логика
    |-- gestures/                  # Базовые, предопределенные и custom-жесты
    `-- utils/                     # Конфиг, фильтр Калмана, профилировщик, голос
```

## Основные модули

- `gesture_framework.app.App` - главный класс для запуска распознавания.
- `gesture_framework.gestures.predefined` - встроенные жесты: `POINTING`, `PINCH`, `OPEN_PALM`, `CLOSED_FIST`, `PEACE`.
- `gesture_framework.gestures.custom.CustomGesture` - обучение и детекция пользовательских жестов.
- `gesture_builder_gui.py` - интерфейс для создания собственного датасета и `.pkl`-модели.

## Установка

Рекомендуется Python 3.10 или 3.11.

### 1. Создать виртуальное окружение

Для Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Установить зависимости

Сначала установите зависимости из `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Для текущей версии кода дополнительно нужны пакеты, которые используются напрямую в исходниках:

```powershell
pip install mediapipe opencv-python scikit-learn psutil SpeechRecognition pydub
```

Примечания:

- `tkinter` обычно входит в стандартную поставку Python.
- для работы управления мышью и клавиатурой используется `pynput`.
- для голосового модуля может потребоваться дополнительная системная настройка микрофона.

## Необходимые файлы

Перед запуском убедитесь, что в корне проекта есть:

- `hand_landmarker.task` - модель MediaPipe для детекции руки;
- `config.json` - файл конфигурации.

Если вы используете пользовательский жест, нужен файл модели `.pkl`, например:

- `custom_gesture/my_click.pkl`

## Быстрый старт

### Запуск примера пользовательского жеста

```powershell
python .\example_custom_click.py
```

Этот пример:

- создает `App(display_camera=True)`;
- загружает модель `custom_gesture/my_click.pkl`;
- привязывает жест `my_click` к действию `mouse_left`.

### Запуск GUI для создания жеста

```powershell
python .\gesture_builder_gui.py
```

В GUI можно:

- задать имя и описание жеста;
- записать положительные примеры;
- записать отрицательные примеры;
- обучить SVM-модель;
- протестировать распознавание;
- сохранить результат в `.pkl`.

## Пример использования API

```python
from pathlib import Path
from gesture_framework import App

model_path = Path("custom_gesture") / "my_click.pkl"

app = App(display_camera=True)
app.add_gesture(
    name="my_click",
    model=str(model_path),
    action="mouse_left",
    confidence_threshold=0.70,
)
app.run()
```

## Поддерживаемые действия

Через `App.add_gesture(...)` можно привязывать жест к действиям:

- `mouse_left`
- `mouse_right`
- `mouse_double`
- `keyboard`
- `custom`

Для `keyboard` можно передать параметры, например:

```python
app.add_gesture(
    name="next_slide",
    model="custom_gesture/next_slide.pkl",
    action="keyboard",
    params={"key": "space"},
)
```

## Конфигурация

Файл `config.json` управляет поведением проекта:

- `engine` - FPS, smoothing, thresholds;
- `gestures` - включение жестов и их действия;
- `voice` - параметры голосового режима;
- `performance` - профилирование и лимиты ресурсов;
- `ui` - параметры отображения;
- `multimodal` - веса жестов и голоса.

Примеры базовых жестов из конфигурации:

- `POINTING` -> `move_cursor`
- `PINCH` -> `click`
- `OPEN_PALM` -> `pause`
- `CLOSED_FIST` -> `hold`
- `PEACE` -> `right_click`

## Как создать свой жест

1. Запустите `gesture_builder_gui.py`.
2. Введите имя жеста.
3. Запишите положительные примеры.
4. Запишите отрицательные примеры.
5. Нажмите обучение.
6. Сохраните `.pkl` модель.
7. Подключите модель через `App.add_gesture(...)`.

## Текущее состояние проекта

Проект рабочий как прототип и SDK-основа, но в текущем виде есть несколько важных моментов:


## Полезные файлы

- [config.json](./config.json)
- [example_custom_click.py](./example_custom_click.py)
- [gesture_builder_gui.py](./gesture_builder_gui.py)
- [gesture_framework/app.py](./gesture_framework/app.py)
