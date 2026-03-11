"""
Kalman Filter - улучшенное сглаживание координат

Использует фильтр Калмана для более точного отслеживания позиции
курсора, уменьшая шум от MediaPipe landmarks
"""

import numpy as np
from typing import Tuple, Optional


class KalmanFilter1D:
    """
    1D Kalman Filter для сглаживания одной оси
    
    Параметры:
    - process_variance: насколько быстро ожидается изменение값ения (0.01)
    - measurement_variance: насколько шумное измерение (2.0)
    - initial_value: начальное значение (0.0)
    - initial_estimate_error: начальная ошибка оценки (1.0)
    """
    
    def __init__(self,
                 process_variance: float = 0.01,
                 measurement_variance: float = 2.0,
                 initial_value: float = 0.0,
                 initial_estimate_error: float = 1.0):
        """
        Args:
            process_variance: Q - дисперсия процесса (меньше = предполагаем гладкое движение)
            measurement_variance: R - дисперсия измерения (больше = не доверяем измерениям)
            initial_value: начальное значение позиции
            initial_estimate_error: начальная ошибка оценки
        """
        self.q = process_variance  # Process variance
        self.r = measurement_variance  # Measurement variance
        
        self.x = initial_value  # Estimated state
        self.p = initial_estimate_error  # Estimate error
        self.k = 0  # Kalman gain
    
    def update(self, measurement: float) -> float:
        """
        Обновить фильтр с новым измерением
        
        Args:
            measurement: новое измерение позиции
        
        Returns:
            сглаженное значение
        """
        # Предсказание
        self.p = self.p + self.q
        
        # Обновление
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        
        return self.x
    
    def reset(self, initial_value: float = 0.0) -> None:
        """Сбросить фильтр"""
        self.x = initial_value
        self.p = 1.0
        self.k = 0


class KalmanFilter2D:
    """
    2D Kalman Filter для сглаживания координат (x, y)
    
    Использует два независимых 1D фильтра для каждой оси
    """
    
    def __init__(self,
                 process_variance: float = 0.01,
                 measurement_variance: float = 2.0,
                 initial_x: float = 0.0,
                 initial_y: float = 0.0):
        """
        Args:
            process_variance: Q - дисперсия процесса
            measurement_variance: R - дисперсия измерения
            initial_x: начальная X позиция
            initial_y: начальная Y позиция
        """
        self.filter_x = KalmanFilter1D(
            process_variance=process_variance,
            measurement_variance=measurement_variance,
            initial_value=initial_x
        )
        
        self.filter_y = KalmanFilter1D(
            process_variance=process_variance,
            measurement_variance=measurement_variance,
            initial_value=initial_y
        )
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Обновить фильтр новыми координатами
        
        Args:
            x: новая X координата
            y: новая Y координата
        
        Returns:
            (smoothed_x, smoothed_y)
        """
        smoothed_x = self.filter_x.update(x)
        smoothed_y = self.filter_y.update(y)
        
        return (smoothed_x, smoothed_y)
    
    def reset(self, initial_x: float = 0.0, initial_y: float = 0.0) -> None:
        """Сбросить оба фильтра"""
        self.filter_x.reset(initial_x)
        self.filter_y.reset(initial_y)
    
    def get_state(self) -> Tuple[float, float]:
        """Получить текущее состояние"""
        return (self.filter_x.x, self.filter_y.x)


class KalmanFilterND:
    """
    N-мерный Kalman Filter - сглаживает массив значений
    
    Полезен для сглаживания всех 21 ключевых точек руки
    """
    
    def __init__(self,
                 dimensions: int,
                 process_variance: float = 0.01,
                 measurement_variance: float = 2.0):
        """
        Args:
            dimensions: количество измерений (для 21 точки руки: 21*2=42 для x,y или 21*3=63 для x,y,z)
            process_variance: Q
            measurement_variance: R
        """
        self.filters = [
            KalmanFilter1D(
                process_variance=process_variance,
                measurement_variance=measurement_variance
            )
            for _ in range(dimensions)
        ]
    
    def update(self, values: list) -> list:
        """
        Обновить фильтр с новыми значениями
        
        Args:
            values: список значений (должен быть той же длины что и dimensions)
        
        Returns:
            список сглаженных значений
        """
        if len(values) != len(self.filters):
            raise ValueError(f"Expected {len(self.filters)} values, got {len(values)}")
        
        return [f.update(v) for f, v in zip(self.filters, values)]
    
    def reset(self) -> None:
        """Сбросить все фильтры"""
        for f in self.filters:
            f.reset()


class AdaptiveKalmanFilter2D:
    """
    Адаптивный Kalman Filter - автоматически настраивает параметры
    
    Увеличивает process_variance когда обнаруживает быстрое движение
    """
    
    def __init__(self,
                 base_process_variance: float = 0.01,
                 base_measurement_variance: float = 2.0,
                 adaptation_threshold: float = 50.0):
        """
        Args:
            base_process_variance: базовая Q
            base_measurement_variance: базовая R
            adaptation_threshold: порог для обнаружения быстрого движения (пиксели)
        """
        self.base_q = base_process_variance
        self.base_r = base_measurement_variance
        self.adaptation_threshold = adaptation_threshold
        
        self.filter = KalmanFilter2D(
            process_variance=base_process_variance,
            measurement_variance=base_measurement_variance
        )
        
        self.last_x = 0.0
        self.last_y = 0.0
        self.velocity = 0.0
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Обновить фильтр с адаптацией к скорости движения
        
        Args:
            x: новая X
            y: новая Y
        
        Returns:
            (smoothed_x, smoothed_y)
        """
        # Рассчитать скорость
        dx = x - self.last_x
        dy = y - self.last_y
        self.velocity = (dx**2 + dy**2)**0.5  # Евклидово расстояние
        
        # Адаптивно изменить process_variance
        if self.velocity > self.adaptation_threshold:
            # Быстрое движение - увеличить Q (доверять временной модели меньше)
            adaptive_q = self.base_q * 5
        else:
            adaptive_q = self.base_q
        
        # Обновить process_variance в фильтре
        self.filter.filter_x.q = adaptive_q
        self.filter.filter_y.q = adaptive_q
        
        # Обновить фильтр
        smoothed_x, smoothed_y = self.filter.update(x, y)
        
        # Сохранить текущие значения
        self.last_x = smoothed_x
        self.last_y = smoothed_y
        
        return (smoothed_x, smoothed_y)
    
    def reset(self) -> None:
        """Сбросить"""
        self.filter.reset()
        self.last_x = 0.0
        self.last_y = 0.0
        self.velocity = 0.0


# Пример использования
if __name__ == "__main__":
    print("=" * 60)
    print("Kalman Filter Demo")
    print("=" * 60)
    
    # Тест 1D фильтров
    print("\n[1] Testing 1D Kalman Filter")
    print("-" * 40)
    
    kf1d = KalmanFilter1D()
    
    # Имитировать шумные измерения
    true_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    noisy_measurements = [v + np.random.normal(0, 0.5) for v in true_values]
    
    print("True -> Noisy -> Filtered")
    for true, noisy in zip(true_values, noisy_measurements):
        filtered = kf1d.update(noisy)
        print(f"{true:.2f} -> {noisy:.2f} -> {filtered:.2f}")
    
    # Тест 2D фильтра
    print("\n[2] Testing 2D Kalman Filter")
    print("-" * 40)
    
    kf2d = KalmanFilter2D()
    
    # Имитировать движение курсора
    positions = [(100, 100), (150, 120), (200, 140), (250, 160)]
    
    print("Original -> Filtered")
    for x, y in positions:
        fx, fy = kf2d.update(x, y)
        print(f"({x}, {y}) -> ({fx:.1f}, {fy:.1f})")
    
    print("\n[OK] Kalman Filter demo completed")
