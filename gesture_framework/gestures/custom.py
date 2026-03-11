"""
Custom Gesture Support - поддержка пользовательских жестов

Позволяет записывать, обучать и распознавать пользовательские жесты
с использованием SVM классификатора
"""

import json
import math
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from gesture_framework.gestures.base_gesture import HandLandmark, BaseGesture


class GestureType(Enum):
    """Типы жестов"""
    PREDEFINED = "predefined"  # Встроенные
    CUSTOM = "custom"          # Пользовательские


@dataclass
class GestureSample:
    """Один пример жеста"""
    landmarks: List[HandLandmark]
    label: str
    timestamp: float
    
    def to_features(self) -> np.ndarray:
        """Конвертировать landmarks в вектор признаков"""
        features = []
        for lm in self.landmarks:
            features.extend([lm.x, lm.y, lm.z, lm.confidence])
        return np.array(features)


class CustomGesture(BaseGesture):
    """
    Пользовательский жест
    
    Обучается на примерах и использует SVM классификатор
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Args:
            name: Имя жеста
            description: Описание жеста
        """
        self.name = name
        self.description = description
        self.samples: List[GestureSample] = []
        self.classifier = None
        self.is_trained = False
        self.confidence_threshold = 0.7
        self.scaler = None
        self.SVC = None
        self.OneClassSVM = None
        self.StandardScaler = None
        self.is_one_class = False
        
        # Try to import sklearn
        try:
            from sklearn.svm import SVC, OneClassSVM
            from sklearn.preprocessing import StandardScaler
            self.SVC = SVC
            self.OneClassSVM = OneClassSVM
            self.StandardScaler = StandardScaler
            self.scaler = StandardScaler()
        except ImportError:
            print("[WARN] scikit-learn не установлен, Custom Gestures недоступны")
    
    @property
    def n_samples(self) -> int:
        """Количество примеров"""
        return len(self.samples)
    
    def add_sample(self, landmarks: List[HandLandmark], label: str = None) -> None:
        """
        Добавить пример жеста
        
        Args:
            landmarks: Список из 21 ключевую точку руки
            label: Метка (по умолчанию используется self.name)
        """
        if len(landmarks) != 21:
            raise ValueError("Нужно ровно 21 ключевую точку")
        
        label = label or self.name
        sample = GestureSample(landmarks=landmarks, label=label, timestamp=__import__('time').time())
        self.samples.append(sample)
        print(f"Добавлен пример для {label} (всего: {len(self.samples)})")
    
    def train_classifier(self) -> bool:
        """
        Обучить SVM классификатор на собранных примерах
        
        Нужно минимум 2 примера для обучения
        
        Returns:
            True если обучение успешно, False иначе
        """
        if not self.SVC:
            print("[ERROR] scikit-learn не доступен")
            return False
        
        if len(self.samples) < 2:
            print(f"[ERROR] Нужно минимум 2 примера для обучения, есть {len(self.samples)}")
            return False
        
        try:
            # Подготовить данные
            X = np.array([sample.to_features() for sample in self.samples])
            y = np.array([sample.label for sample in self.samples])
            
            # Очистить NaN значения ПЕРЕД нормализацией
            # Найти строки и столбцы с NaN
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                print("[WARN] Обнаружены NaN значения, очищаю данные...")
                
                # Заполнить NaN средним значением столбца
                for col in range(X.shape[1]):
                    col_data = X[~np.isnan(X[:, col]), col]
                    if len(col_data) > 0:
                        X[nan_mask[:, col], col] = np.median(col_data)
                    else:
                        X[nan_mask[:, col], col] = 0.5  # default значение
                
                print("[OK] NaN значения заменены")
            
            # Проверить что нет NaN после очистки
            if np.any(np.isnan(X)):
                print("[ERROR] Все еще есть NaN после очистки, aborting")
                return False
            
            unique_labels = np.unique(y)
            
            # Нормализировать признаки
            X_scaled = self.scaler.fit_transform(X)
            
            if len(unique_labels) == 1:
                # One-class mode (только положительные примеры)
                self.classifier = self.OneClassSVM(
                    kernel='rbf',
                    gamma='scale',
                    nu=0.1
                )
                self.classifier.fit(X_scaled)
                self.is_one_class = True
                self.test_accuracy = None  # OneClass не имеет accuracy
                print(f"[OK] OneClassSVM обучен на {len(self.samples)} примерах")
            else:
                # Обычный SVM для нескольких классов
                # Разделяем данные на train/test (80/20) для честной оценки
                from sklearn.model_selection import train_test_split
                
                if len(X_scaled) >= 10:  # Достаточно данных для split
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=42, stratify=y
                        )
                    except ValueError:
                        # Если stratify не работает (мало примеров одного класса), без stratify
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=42
                        )
                    
                    self.classifier = self.SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        probability=True
                    )
                    self.classifier.fit(X_train, y_train)
                    self.is_one_class = False
                    
                    # Оценка точности на тестовом наборе (честная оценка!)
                    test_score = self.classifier.score(X_test, y_test)
                    self.test_accuracy = test_score
                    print("[OK] Классификатор обучен")
                    print(f"   Примеров: {len(X_train)} train + {len(X_test)} test")
                    print(f"   Точность на валидации: {test_score:.1%}")
                else:
                    # Если мало данных, обучаем на всех (но предупреждаем)
                    print(f"[WARN] Мало данных ({len(X_scaled)}) для train/test split")
                    print(f"   Модель может быть переобучена! Рекомендуется минимум 10 примеров")
                    
                    self.classifier = self.SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        probability=True
                    )
                    self.classifier.fit(X_scaled, y)
                    self.is_one_class = False
                    self.test_accuracy = None
                    print(f"[OK] Классификатор обучен на {len(self.samples)} примерах (без валидации)")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            print(f"[ERROR] Ошибка обучения: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_detected(self, landmarks: List[HandLandmark]) -> Tuple[bool, float]:
        """
        Определить, является ли жест текущим
        
        Args:
            landmarks: Список из 21 ключевую точку
        
        Returns:
            (is_detected, confidence)
        """
        if not self.is_trained or not self.classifier:
            return False, 0.0
        
        try:
            def _safe_float(value: float, default: float) -> float:
                """Convert value to float, returning default on invalid or NaN."""
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    return default
                return default if math.isnan(value_f) else value_f

            # Извлечь признаки из landmarks
            features = []
            for landmark in landmarks:
                # Использовать x, y, z, и confidence
                x = _safe_float(landmark.x, 0.5)
                y = _safe_float(landmark.y, 0.5)
                z = _safe_float(landmark.z, 0.5)
                conf = _safe_float(landmark.confidence, 0.5)
                features.extend([x, y, z, conf])
            
            # Конвертировать в numpy array с явным типом float64
            features = np.array(features, dtype=np.float64).reshape(1, -1)
            
            # Проверить на NaN значения (не должны быть после очистки)
            if np.isnan(features).any():
                return False, 0.0
            
            # Нормализировать
            features_scaled = self.scaler.transform(features)
            
            # Проверить на NaN после scaling
            if np.isnan(features_scaled).any():
                return False, 0.0
            
            if self.is_one_class:
                # One-class режим
                score = float(self.classifier.decision_function(features_scaled)[0])
                
                # Для OneClassSVM порог обычно около 0.0
                # Положительные значения = внутри класса, отрицательные = аномалии
                # Используем переданный threshold или 0.0 по умолчанию
                threshold = self.confidence_threshold if self.confidence_threshold is not None else 0.0
                
                # Нормализуем score для отображения через сигмоиду
                # Это преобразует score в диапазон 0-1 для визуализации
                normalized_confidence = 1.0 / (1.0 + math.exp(-score))
                
                # Для OneClassSVM проверяем что score выше порога (обычно 0)
                is_detected = score > threshold
                
                return is_detected, normalized_confidence
            
            # Предсказать (multi-class)
            prediction = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            
            # Максимальная вероятность
            max_prob = np.max(probabilities)
            
            # Проверить, что это наш жест
            is_our_gesture = prediction == self.name
            
            # Для multi-class порог должен быть в диапазоне 0-1 (это вероятность!)
            # Если передан отрицательный порог (ошибка), используем разумное значение
            effective_threshold = self.confidence_threshold
            if effective_threshold < 0 or effective_threshold > 1:
                effective_threshold = 0.7  # Разумный порог для multi-class
            
            # Дополнительная проверка: если есть несколько классов,
            # проверяем что вероятность нашего класса значительно выше других
            if len(probabilities) > 1 and is_our_gesture:
                # Найти индекс нашего класса
                our_class_idx = list(self.classifier.classes_).index(self.name)
                our_prob = probabilities[our_class_idx]
                
                # КРИТИЧЕСКАЯ ПРОВЕРКА: margin (разница с другими классами)
                other_probs = [p for i, p in enumerate(probabilities) if i != our_class_idx]
                max_other_prob = max(other_probs) if other_probs else 0.0
                
                margin = our_prob - max_other_prob
                min_margin = 0.25  # 25% разница для уверенности
                
                # Диагностика
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Gesture check: {self.name}")
                logger.debug(f"  Probabilities: {dict(zip(self.classifier.classes_, probabilities))}")
                logger.debug(f"  Our prob: {our_prob:.2%}, Max other: {max_other_prob:.2%}")
                logger.debug(f"  Margin: {margin:.2%} (need {min_margin:.0%})")
                logger.debug(f"  Threshold: {effective_threshold:.2%}")
                
                # Проверка: 1. Вероятность >= threshold, 2. Margin >= min_margin
                passes_threshold = our_prob >= effective_threshold
                passes_margin = margin >= min_margin
                
                logger.debug(f"  Passes threshold: {passes_threshold}, Passes margin: {passes_margin}")
                
                if passes_threshold and passes_margin:
                    return True, float(our_prob)
                else:
                    return False, float(our_prob)
            elif is_our_gesture and max_prob >= effective_threshold:
                return True, float(max_prob)
            else:
                return False, float(max_prob) if is_our_gesture else 0.0
        
        except Exception as e:
            print(f"[WARN] Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def save(self, filepath: str) -> bool:
        """
        Сохранить обученный жест в файл
        
        Args:
            filepath: Путь для сохранения
        
        Returns:
            True если успешно
        """
        try:
            import pickle
            
            data = {
                'name': self.name,
                'description': self.description,
                'is_trained': self.is_trained,
                'confidence_threshold': self.confidence_threshold,
                'is_one_class': self.is_one_class,
                'classifier': self.classifier,
                'scaler': self.scaler,
                'samples_count': len(self.samples)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"[OK] Жест сохранен: {filepath}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Загрузить обученный жест из файла
        
        Args:
            filepath: Путь к файлу
        
        Returns:
            True если успешно
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.name = data['name']
            self.description = data['description']
            self.is_trained = data['is_trained']
            self.confidence_threshold = data['confidence_threshold']
            self.classifier = data['classifier']
            self.scaler = data['scaler']
            self.is_one_class = data.get('is_one_class', False)
            if not self.is_one_class and self.classifier is not None:
                self.is_one_class = self.classifier.__class__.__name__ == 'OneClassSVM'
            
            print(f"[OK] Жест загружен: {filepath}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str, confidence_threshold: float = None) -> 'CustomGesture':
        """
        Загрузить обученный жест из файла (classmethod)
        
        Args:
            filepath: Путь к файлу
            confidence_threshold: Переопределить порог уверенности (опционально)
        
        Returns:
            Объект CustomGesture
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Создать новый объект
            gesture = cls(data['name'], data['description'])
            gesture.is_trained = data['is_trained']
            
            # Использовать переданный threshold или из модели
            if confidence_threshold is not None:
                gesture.confidence_threshold = confidence_threshold
            else:
                gesture.confidence_threshold = data['confidence_threshold']
            
            gesture.classifier = data['classifier']
            gesture.scaler = data['scaler']
            gesture.is_one_class = data.get('is_one_class', False)
            if not gesture.is_one_class and gesture.classifier is not None:
                gesture.is_one_class = gesture.classifier.__class__.__name__ == 'OneClassSVM'
            
            print(f"[OK] Жест загружен: {filepath}")
            print(f"   - Название: {gesture.name}")
            print(f"   - Тип: OneClassSVM" if gesture.is_one_class else f"   - Тип: Standard")
            print(f"   - Порог: {gesture.confidence_threshold:.2f}")
            return gesture
        
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки: {e}")
            return None


class GestureRecorder:
    """
    Инструмент для записи примеров жестов
    
    Позволяет записать несколько примеров жеста и подготовить их к обучению
    """
    
    def __init__(self, gesture_name: str, required_samples: int = 10):
        """
        Args:
            gesture_name: Имя жеста для записи
            required_samples: Количество примеров для записи
        """
        self.gesture_name = gesture_name
        self.name = gesture_name  # Alias for compatibility
        self.required_samples = required_samples
        self.samples: List[GestureSample] = []
        self.is_recording = False
        
        print(f"GestureRecorder для {gesture_name} готов (нужно {required_samples} примеров)")
    
    def start_recording(self) -> None:
        """Начать запись жеста"""
        self.is_recording = True
        print(f"Запись {self.gesture_name} началась...")
    
    def stop_recording(self) -> None:
        """Остановить запись"""
        self.is_recording = False
        print(f"⏹️  Запись остановлена (собрано {len(self.samples)} примеров)")
    
    def add_landmarks(self, landmarks: List[HandLandmark]) -> None:
        """
        Добавить пример во время записи
        
        Args:
            landmarks: Список из 21 ключевую точку
        """
        if not self.is_recording:
            return
        
        if len(landmarks) != 21:
            return
        
        sample = GestureSample(
            landmarks=landmarks,
            label=self.gesture_name,
            timestamp=__import__('time').time()
        )
        self.samples.append(sample)
        
        remaining = self.required_samples - len(self.samples)
        if remaining > 0:
            print(f"  Пример {len(self.samples)}/{self.required_samples} (осталось {remaining})")
        else:
            print("  [OK] Записано достаточно примеров!")
            self.stop_recording()
    
    def get_samples(self) -> List[GestureSample]:
        """Получить все записанные примеры"""
        return self.samples.copy()
    
    def clear(self) -> None:
        """Очистить записанные примеры"""
        self.samples = []
        self.is_recording = False
        print("Примеры очищены")
    
    def save_dataset(self, filepath: str) -> bool:
        """
        Сохранить датасет примеров в JSON файл
        
        Returns:
            True если успешно
        """
        try:
            data = {
                'gesture_name': self.gesture_name,
                'sample_count': len(self.samples),
                'samples': [
                    {
                        'landmarks': [
                            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'confidence': lm.confidence}
                            for lm in sample.landmarks
                        ],
                        'label': sample.label,
                        'timestamp': sample.timestamp
                    }
                    for sample in self.samples
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[OK] Датасет сохранен: {filepath}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения датасета: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Достаточно ли примеров для обучения"""
        return len(self.samples) >= self.required_samples


if __name__ == "__main__":
    print("Testing CustomGesture...")
    
    # Создать пользовательский жест
    custom = CustomGesture("CUSTOM_SWIPE", "Свайп вверх")
    
    # Имитировать записанные примеры
    from gesture_framework.gestures.base_gesture import HandLandmark
    
    for i in range(5):
        landmarks = [
            HandLandmark(x=0.5 + np.random.normal(0, 0.1),
                        y=0.5 + np.random.normal(0, 0.1),
                        z=0.0,
                        confidence=0.9)
            for _ in range(21)
        ]
        custom.add_sample(landmarks)
    
    # Обучить
    custom.train_classifier()
    
    print("\n[OK] CustomGesture test completed")
