"""
Gesture Builder GUI - интерактивное создание и обучение жестов

Tkinter-based interface для:
- Запись жестов в реальном времени
- Визуализация landmarks
- Обучение SVM классификатора
- Тестирование новых жестов

UPDATED FOR MEDIAPIPE 0.10.30+
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.scrolledtext as st
from typing import List, Optional, Dict
import numpy as np
import threading
import json
import time
import math
from pathlib import Path
from datetime import datetime
import cv2

# NEW MEDIAPIPE IMPORT
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

from gesture_framework.gestures.base_gesture import HandLandmark
from gesture_framework.gestures.custom import CustomGesture, GestureRecorder
from gesture_framework.utils.config import ConfigManager


class GestureBuilderGUI:
    """
    Графический интерфейс для создания и обучения жестов
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Builder - Создатель Жестов")
        self.root.geometry("1200x800")
        
        # Загрузить конфигурацию
        try:
            self.config = ConfigManager('config.json')
        except:
            self.config = ConfigManager()
        
        # Состояние приложения
        self.current_gesture = None
        self.recorder = None
        self.custom_gesture = None
        self.is_recording = False
        self.is_recording_negative = False
        self.sample_count = 0
        self.negative_sample_count = 0
        self.required_samples = 10
        self.required_negative_samples = 10
        self.test_landmarks = None
        self.gesture_history = []  # История жестов для анализа
        self.last_landmarks = None
        self.negative_recorder = None
        
        # Инициализация MediaPipe - NEW API
        self._initialize_mediapipe()
        
        # Инициализация веб-камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        
        # Создать UI
        self._create_ui()
        
        # Запустить поток захвата видео
        self.video_thread = threading.Thread(target=self._capture_video, daemon=True)
        self.video_thread.start()
        
        print("Gesture Builder GUI инициализирован с захватом видео")
    
    def _initialize_mediapipe(self):
        """Инициализация MediaPipe с новым API"""
        model_path = 'hand_landmarker.task'
        
        # Проверить наличие модели
        if not Path(model_path).exists():
            print(f"[WARN] Модель {model_path} не найдена!")
            print("Скачайте модель:")
            print("   wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            messagebox.showwarning(
                "Требуется модель",
                f"Файл модели '{model_path}' не найден!\n\n"
                "Скачайте его командой:\n"
                "wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n\n"
                "Или вручную по ссылке:\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Создать опции для HandLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # Эквивалент static_image_mode=False
            num_hands=1,  # Эквивалент max_num_hands=1
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Создать детектор
        self.hands = vision.HandLandmarker.create_from_options(options)
        print("[OK] MediaPipe HandLandmarker инициализирован (новый API)")
    
    def _create_ui(self):
        """Создать пользовательский интерфейс"""
        
        # === Главное меню ===
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить жест", command=self._load_gesture)
        file_menu.add_command(label="Сохранить жест", command=self._save_gesture)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self._show_about)
        
        # === Главный контейнер ===
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === Левая панель - Контроль ===
        left_frame = ttk.LabelFrame(main_frame, text="Управление жестом")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        # Имя жеста
        ttk.Label(left_frame, text="Имя жеста:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.gesture_name_var = tk.StringVar(value="NEW_GESTURE")
        ttk.Entry(left_frame, textvariable=self.gesture_name_var, width=20).grid(row=0, column=1, padx=5, pady=5)
        
        # Описание
        ttk.Label(left_frame, text="Описание:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.gesture_desc_var = tk.StringVar(value="Мой жест")
        ttk.Entry(left_frame, textvariable=self.gesture_desc_var, width=20).grid(row=1, column=1, padx=5, pady=5)
        
        # Требуемое количество примеров
        ttk.Label(left_frame, text="Примеров:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.required_samples_var = tk.IntVar(value=10)
        ttk.Spinbox(left_frame, from_=2, to=50, textvariable=self.required_samples_var, width=18).grid(row=2, column=1, padx=5, pady=5)
        
        # === Кнопки управления ===
        control_frame = ttk.LabelFrame(left_frame, text="Управление")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)
        
        self.create_btn = ttk.Button(control_frame, text="Создать", command=self._create_gesture)
        self.create_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.record_btn = ttk.Button(control_frame, text="Начать запись", command=self._toggle_recording)
        self.record_btn.pack(fill=tk.X, padx=5, pady=5)
        self.record_btn.config(state='disabled')

        self.negative_record_btn = ttk.Button(control_frame, text="Запись НЕ-жеста", command=self._toggle_negative_recording)
        self.negative_record_btn.pack(fill=tk.X, padx=5, pady=5)
        self.negative_record_btn.config(state='disabled')
        
        self.train_btn = ttk.Button(control_frame, text="Обучить", command=self._train_gesture)
        self.train_btn.pack(fill=tk.X, padx=5, pady=5)
        self.train_btn.config(state='disabled')
        
        self.test_btn = ttk.Button(control_frame, text="Тест", command=self._test_gesture)
        self.test_btn.pack(fill=tk.X, padx=5, pady=5)
        self.test_btn.config(state='disabled')
        
        # === Статус записи ===
        status_frame = ttk.LabelFrame(left_frame, text="Статус")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)
        
        ttk.Label(status_frame, text="Примеров:", font=("Arial", 9)).pack(anchor=tk.W, padx=5)
        self.sample_count_label = tk.Label(status_frame, text="0/10", font=("Arial", 12, "bold"), foreground="blue")
        self.sample_count_label.pack(anchor=tk.W, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=10, length=200)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(status_frame, text="Примеров (НЕ-жест):", font=("Arial", 9)).pack(anchor=tk.W, padx=5)
        self.negative_count_label = tk.Label(status_frame, text="0/10", font=("Arial", 12, "bold"), foreground="purple")
        self.negative_count_label.pack(anchor=tk.W, padx=5)

        self.negative_progress_var = tk.DoubleVar()
        self.negative_progress_bar = ttk.Progressbar(status_frame, variable=self.negative_progress_var, maximum=10, length=200)
        self.negative_progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Статус:", font=("Arial", 9)).pack(anchor=tk.W, padx=5)
        self.status_label = tk.Label(status_frame, text="Ожидание...", font=("Arial", 9), foreground="gray")
        self.status_label.pack(anchor=tk.W, padx=5)
        
        # === Средняя панель - Информация ===
        middle_frame = ttk.LabelFrame(main_frame, text="Информация о жесте")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Текстовое поле с информацией
        self.info_text = st.ScrolledText(middle_frame, height=30, width=40, font=("Courier", 9), wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === Правая панель - Визуализация ===
        right_frame = ttk.LabelFrame(main_frame, text="Визуализация")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas для графика
        self.canvas = tk.Canvas(right_frame, bg='white', height=400, width=400)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Кнопка для обновления графика
        ttk.Button(right_frame, text="Обновить график", command=self._update_visualization).pack(padx=5, pady=5)
        
        self._update_info("Gesture Builder готов к работе\n\n1. Введите имя жеста\n2. Нажмите 'Создать'\n3. Нажмите 'Начать запись' и выполните жест\n4. Нажмите 'Запись НЕ-жеста' и выполните другие движения\n5. Нажмите 'Обучить'\n6. Проверьте жест")
    
    def _capture_video(self):
        """Захват видео с камеры и экстракция landmarks - UPDATED FOR NEW API"""
        def _safe_float(value, default):
            """Convert value to float, returning default on invalid or NaN."""
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return default
            return default if math.isnan(value_f) else value_f

        frame_timestamp_ms = 0
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Зеркально отразить для удобства
                frame = cv2.flip(frame, 1)
                
                # Конвертировать в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Создать MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Обработать с новым API
                results = self.hands.detect_for_video(mp_image, frame_timestamp_ms)
                frame_timestamp_ms += 33  # ~30 FPS
                
                # Если обнаружена рука
                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand_landmarks = results.hand_landmarks[0]  # Первая рука
                    
                    # Преобразовать landmarks в формат для recorder
                    # В новом API landmarks - это список объектов NormalizedLandmark
                    # ВАЖНО: Используем [lm.x, lm.y, lm.z, lm.presence] (visibility) с защитой от NaN
                    landmarks = []
                    for lm in hand_landmarks:
                        x = _safe_float(lm.x, 0.5)
                        y = _safe_float(lm.y, 0.5)
                        z = _safe_float(lm.z, 0.5)
                        presence = _safe_float(getattr(lm, 'presence', 1.0), 0.5)
                        landmarks.append([x, y, z, presence])
                    
                    landmarks = np.array(landmarks, dtype=np.float32)

                    # Сохранить последние landmarks для тестирования
                    self.last_landmarks = self._to_hand_landmarks(landmarks)
                    
                    # Добавить в recorder если идёт запись
                    if self.is_recording and self.recorder:
                        self._add_sample(landmarks, is_negative=False)
                    if self.is_recording_negative and self.negative_recorder:
                        self._add_sample(landmarks, is_negative=True)
                
            except Exception as e:
                print(f"Ошибка захвата видео: {e}")
                import traceback
                traceback.print_exc()
                
            # Небольшая задержка
            time.sleep(0.033)  # ~30 FPS
    
    def _create_gesture(self):
        """Создать новый жест"""
        name = self.gesture_name_var.get().strip()
        desc = self.gesture_desc_var.get().strip()
        
        if not name:
            messagebox.showerror("Ошибка", "Введите имя жеста")
            return
        
        self.required_samples = self.required_samples_var.get()
        self.custom_gesture = CustomGesture(name, desc)
        self.recorder = GestureRecorder(name, required_samples=self.required_samples)
        self.negative_recorder = GestureRecorder(f"not_{name}", required_samples=self.required_negative_samples)
        self.sample_count = 0
        self.negative_sample_count = 0
        self.is_recording = False
        self.is_recording_negative = False
        
        # Обновить UI
        self.record_btn.config(state=tk.NORMAL)
        self.negative_record_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_bar.config(maximum=self.required_samples)
        self.negative_progress_var.set(0)
        self.negative_progress_bar.config(maximum=self.required_negative_samples)
        self.sample_count_label.config(text=f"0/{self.required_samples}")
        self.negative_count_label.config(text=f"0/{self.required_negative_samples}")
        
        self._update_info(f"[OK] Жест '{name}' создан\n\nОписание: {desc}\nТребуется примеров: {self.required_samples}")
    
    def _toggle_recording(self):
        """Включить/отключить запись"""
        if not self.custom_gesture:
            messagebox.showerror("Ошибка", "Сначала создайте жест")
            return

        if self.is_recording_negative:
            messagebox.showerror("Ошибка", "Сначала остановите запись НЕ-жеста")
            return
        
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.recorder.start_recording()
            self.record_btn.config(text="Остановить запись")
            self.status_label.config(text="Запись...", foreground="red")
            self._update_info("Запись начата\n\nВыполняйте жест перед камерой...")
        else:
            self.recorder.stop_recording()
            self.record_btn.config(text="Начать запись")
            self.status_label.config(text="Ожидание...", foreground="gray")
            
            # Проверить готовность
            if self.recorder.is_ready():
                self._update_info(f"[OK] Достаточно примеров собрано!\n\nПримеров: {len(self.recorder.get_samples())}\n\nТеперь нажмите 'Обучить'")
                self.train_btn.config(state=tk.NORMAL)
            else:
                remaining = self.required_samples - len(self.recorder.get_samples())
                self._update_info(f"Нужно еще {remaining} примеров\n\nСобрано: {len(self.recorder.get_samples())}/{self.required_samples}")

    def _toggle_negative_recording(self):
        """Включить/отключить запись НЕ-жеста"""
        if not self.custom_gesture:
            messagebox.showerror("Ошибка", "Сначала создайте жест")
            return

        if self.is_recording:
            messagebox.showerror("Ошибка", "Сначала остановите запись жеста")
            return

        self.is_recording_negative = not self.is_recording_negative

        if self.is_recording_negative:
            self.negative_recorder.start_recording()
            self.negative_record_btn.config(text="Остановить НЕ-жест")
            self.status_label.config(text="Запись НЕ-жеста...", foreground="blue")
            self._update_info("Запись НЕ-жеста начата\n\nПокажите любые другие движения руки...")
        else:
            self.negative_recorder.stop_recording()
            self.negative_record_btn.config(text="Запись НЕ-жеста")
            self.status_label.config(text="Ожидание...", foreground="gray")

            if self.negative_recorder.is_ready():
                self._update_info(f"[OK] Достаточно НЕ-жестов собрано!\n\nПримеров: {len(self.negative_recorder.get_samples())}\n\nТеперь нажмите 'Обучить'")
                if self.recorder and self.recorder.is_ready():
                    self.train_btn.config(state=tk.NORMAL)
            else:
                remaining = self.required_negative_samples - len(self.negative_recorder.get_samples())
                self._update_info(f"Нужно еще {remaining} НЕ-жестов\n\nСобрано: {len(self.negative_recorder.get_samples())}/{self.required_negative_samples}")
    
    def _add_sample(self, landmarks, is_negative: bool = False):
        """Добавить пример во время записи"""
        if not self.is_recording and not self.is_recording_negative:
            return
        
        # Привести numpy массив к списку HandLandmark
        if isinstance(landmarks, np.ndarray):
            landmarks = self._to_hand_landmarks(landmarks)

        if is_negative:
            self.negative_recorder.add_landmarks(landmarks)
            self.negative_sample_count = len(self.negative_recorder.get_samples())
            self.negative_progress_var.set(min(self.negative_sample_count, self.required_negative_samples))
            self.negative_count_label.config(text=f"{self.negative_sample_count}/{self.required_negative_samples}")

            if self.negative_recorder.is_ready():
                self.is_recording_negative = False
                self.negative_record_btn.config(text="Запись НЕ-жеста")
                self._update_info(f"[OK] Достаточно НЕ-жестов!\n\nСобрано: {self.negative_sample_count}")
                self.status_label.config(text="Готово", foreground="green")
                if self.recorder and self.recorder.is_ready():
                    self.train_btn.config(state=tk.NORMAL)
        else:
            self.recorder.add_landmarks(landmarks)
            self.sample_count = len(self.recorder.get_samples())
            self.progress_var.set(min(self.sample_count, self.required_samples))
            self.sample_count_label.config(text=f"{self.sample_count}/{self.required_samples}")
            
            if self.recorder.is_ready():
                self.is_recording = False
                self.record_btn.config(text="Начать запись")
                self._update_info(f"[OK] Достаточно примеров!\n\nСобрано: {self.sample_count}")
                self.status_label.config(text="Готово", foreground="green")
                if self.negative_recorder and self.negative_recorder.is_ready():
                    self.train_btn.config(state=tk.NORMAL)

    def _to_hand_landmarks(self, landmarks: np.ndarray) -> List[HandLandmark]:
        """Преобразовать numpy landmarks (21x4) в список HandLandmark.
        
        Ожидает формат: [[x, y, z, visibility], ...]
        """
        converted: List[HandLandmark] = []
        for lm in landmarks:
            # Если только 3 значения, добавить visibility=1.0
            if len(lm) == 3:
                visibility = 1.0
                x, y, z = lm[0], lm[1], lm[2]
            # Если 4 значения, использовать presence/visibility
            else:
                x, y, z = lm[0], lm[1], lm[2]
                visibility = float(lm[3]) if len(lm) > 3 else 1.0
            
            converted.append(HandLandmark(
                x=float(x),
                y=float(y),
                z=float(z),
                confidence=visibility  # Используем presence/visibility как confidence
            ))
        return converted
    
    def _train_gesture(self):
        """Обучить классификатор"""
        if not self.recorder or not self.recorder.is_ready():
            messagebox.showerror("Ошибка", "Нужно записать достаточно примеров жеста")
            return

        if not self.negative_recorder or not self.negative_recorder.is_ready():
            messagebox.showerror("Ошибка", "Нужно записать достаточно НЕ-жестов")
            return
        
        try:
            # Сбросить предыдущие примеры
            self.custom_gesture.samples = []

            # Добавить примеры в жест с очисткой NaN
            valid_samples = 0
            invalid_samples = 0
            valid_negative = 0
            invalid_negative = 0
            
            for sample in self.recorder.get_samples():
                # Очистить NaN в landmarks перед добавлением
                cleaned_landmarks = self._clean_landmarks(sample.landmarks)
                
                if cleaned_landmarks:
                    self.custom_gesture.add_sample(cleaned_landmarks, label=self.custom_gesture.name)
                    valid_samples += 1
                else:
                    invalid_samples += 1

            negative_label = f"not_{self.custom_gesture.name}"
            for sample in self.negative_recorder.get_samples():
                cleaned_landmarks = self._clean_landmarks(sample.landmarks)

                if cleaned_landmarks:
                    self.custom_gesture.add_sample(cleaned_landmarks, label=negative_label)
                    valid_negative += 1
                else:
                    invalid_negative += 1
            
            if valid_samples < 5 or valid_negative < 5:
                messagebox.showerror("Ошибка", f"Недостаточно валидных примеров (жест: {valid_samples}, не-жест: {valid_negative})")
                return
            
            if invalid_samples > 0:
                print(f"[WARN] Отфильтровано {invalid_samples} примеров с NaN")
            if invalid_negative > 0:
                print(f"[WARN] Отфильтровано {invalid_negative} НЕ-жестов с NaN")
            
            # Обучить
            if self.custom_gesture.train_classifier():
                self._update_info(
                    "[OK] Жест успешно обучен!\n\n"
                    f"Примеров жеста: {valid_samples}\n"
                    f"Примеров НЕ-жеста: {valid_negative}\n\n"
                    "Теперь можно протестировать жест"
                )
                self.test_btn.config(state=tk.NORMAL)
            else:
                self._update_info("[WARN] Не удалось обучить (scikit-learn недоступен)\n\nВы можете сохранить примеры и обучить позже")
        except Exception as e:
            messagebox.showerror("Ошибка обучения", str(e))
            self._update_info(f"[ERROR] Ошибка: {e}")
    
    def _clean_landmarks(self, landmarks):
        """Очистить landmarks от NaN значений.
        
        Args:
            landmarks: Список HandLandmark объектов
            
        Returns:
            Очищенный список или None если слишком много NaN
        """
        if not landmarks:
            return None
        
        cleaned = []
        nan_count = 0
        
        for lm in landmarks:
            # Проверить на NaN в координатах
            if (np.isnan(lm.x) or np.isnan(lm.y) or np.isnan(lm.z) or 
                np.isnan(lm.confidence)):
                nan_count += 1
                # Заменить NaN на default значения
                x = 0.5 if np.isnan(lm.x) else lm.x
                y = 0.5 if np.isnan(lm.y) else lm.y
                z = 0.5 if np.isnan(lm.z) else lm.z
                conf = 0.5 if np.isnan(lm.confidence) else lm.confidence
                
                cleaned.append(HandLandmark(
                    x=float(np.clip(x, 0.0, 1.0)),
                    y=float(np.clip(y, 0.0, 1.0)),
                    z=float(z),
                    confidence=float(np.clip(conf, 0.0, 1.0))
                ))
            else:
                cleaned.append(lm)
        
        # Если больше 50% NaN, отклонить пример
        if nan_count > len(landmarks) / 2:
            return None
        
        return cleaned
    
    def _test_gesture(self):
        """Протестировать жест на НОВОМ кадре (не на тренировочных данных!)"""
        if not self.custom_gesture or not self.custom_gesture.is_trained:
            messagebox.showerror("Ошибка", "Жест не обучен")
            return

        # ВАЖНО: Тестируем ТОЛЬКО на текущем кадре с камеры (не на обучающих примерах!)
        if not self.last_landmarks:
            messagebox.showerror("Ошибка", 
                               "Покажите руку перед камерой для теста.\n\n" +
                               "ВАЖНО: Тест должен быть на НОВЫХ данных,\n" +
                               "не на примерах из обучающей выборки!")
            return
        
        landmarks = self.last_landmarks
        
        is_detected, confidence = self.custom_gesture.is_detected(landmarks)
        
        result_text = f"Тест жеста '{self.custom_gesture.name}'\n\n"
        result_text += f"Обнаружен: {'Да' if is_detected else 'Нет'}\n"
        result_text += f"Уверенность (текущий кадр): {confidence:.1%}\n\n"
        
        # Показываем точность модели на валидации
        if hasattr(self.custom_gesture, 'test_accuracy') and self.custom_gesture.test_accuracy is not None:
            result_text += f"Точность модели (валидация): {self.custom_gesture.test_accuracy:.1%}\n\n"
        
        if is_detected and confidence > 0.7:
            result_text += "Жест хорошо распознан!"
        elif is_detected and confidence > 0.5:
            result_text += "Жест распознан, но уверенность низкая"
        else:
            result_text += "Жест не распознан. Может потребоваться больше примеров"
        
        self._update_info(result_text)
    
    def _save_gesture(self):
        """Сохранить жест в файл"""
        if not self.custom_gesture:
            messagebox.showerror("Ошибка", "Нет жеста для сохранения")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialfile=f"{self.custom_gesture.name}.pkl"
        )
        
        if filepath:
            if self.custom_gesture.save(filepath):
                messagebox.showinfo("Успех", f"Жест сохранен в {filepath}")
                self._update_info(f"[OK] Жест сохранен\n\nФайл: {filepath}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сохранить жест")
    
    def _load_gesture(self):
        """Загрузить жест из файла"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            gesture = CustomGesture.load(filepath)
            if gesture:
                self.custom_gesture = gesture
                self.gesture_name_var.set(gesture.name)
                self.gesture_desc_var.set(gesture.description)
                self.test_btn.config(state=tk.NORMAL)
                self._update_info(f"[OK] Жест загружен\n\nИмя: {gesture.name}\nОписание: {gesture.description}\n\nможно тестировать")
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить жест")
    
    def _update_visualization(self):
        """Обновить визуализацию"""
        self.canvas.delete("all")
        
        if not self.recorder or not self.recorder.samples:
            self.canvas.create_text(200, 200, text="Нет данных для визуализации", font=("Arial", 12))
            return
        
        # Визуализировать статистику
        samples_count = len(self.recorder.samples)
        
        # График количества примеров
        self.canvas.create_rectangle(50, 50, 350, 400, outline="black", width=2)
        self.canvas.create_text(200, 30, text=f"Примеры: {samples_count}/{self.required_samples}", font=("Arial", 12, "bold"))
        
        # Прогресс
        if self.required_samples > 0:
            progress = (samples_count / self.required_samples) * 300
            self.canvas.create_rectangle(50, 100, 50 + progress, 130, fill="green", outline="darkgreen", width=2)
            self.canvas.create_text(200, 115, text=f"{samples_count}/{self.required_samples}", font=("Arial", 10), fill="white")
        
        # Статистика по координатам
        if samples_count > 0:
            all_landmarks = []
            for sample in self.recorder.samples:
                all_landmarks.extend([lm for lm in sample.landmarks])
            
            if all_landmarks:
                x_coords = [lm.x for lm in all_landmarks]
                y_coords = [lm.y for lm in all_landmarks]
                
                avg_x = np.mean(x_coords)
                avg_y = np.mean(y_coords)
                std_x = np.std(x_coords)
                std_y = np.std(y_coords)
                
                # Рисовать точку центра
                px = 50 + avg_x * 300
                py = 150 + avg_y * 200
                self.canvas.create_oval(px-5, py-5, px+5, py+5, fill="red", outline="darkred")
                
                # Рисовать эллипс для стандартного отклонения
                rx = std_x * 300
                ry = std_y * 200
                self.canvas.create_oval(px-rx, py-ry, px+rx, py+ry, outline="orange", width=2)
                
                self.canvas.create_text(200, 360, text=f"Center: ({avg_x:.2f}, {avg_y:.2f})", font=("Arial", 9))
                self.canvas.create_text(200, 380, text=f"Std Dev: ({std_x:.2f}, {std_y:.2f})", font=("Arial", 9))
    
    def _update_info(self, text):
        """Обновить информационное поле"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)
    
    def cleanup(self):
        """Очистить ресурсы при закрытии"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        print("Ресурсы освобождены")
    
    def _show_about(self):
        """Показать информацию о программе"""
        messagebox.showinfo("О программе", 
            "Gesture Builder v1.0\n\n"
            "Инструмент для создания и обучения пользовательских жестов\n\n"
            "© 2024 Gesture Control Framework\n\n"
            "Updated for MediaPipe 0.10.30+")


def main():
    """Главная функция"""
    root = tk.Tk()
    
    # Установить иконку (если доступна)
    try:
        root.iconbitmap('gesture_icon.ico')
    except:
        pass
    
    app = GestureBuilderGUI(root)
    
    # Обработчик закрытия окна
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()