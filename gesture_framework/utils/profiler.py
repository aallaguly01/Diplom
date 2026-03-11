"""
Performance Profiler - мониторинг производительности в реальном времени

Измеряет:
- FPS (frames per second)
- Задержка обработки (latency)
- Использование CPU
- Использование памяти
"""

import time
import psutil
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class MetricType(Enum):
    """Типы метрик"""
    FPS = "fps"
    GESTURE_LATENCY = "gesture_latency_ms"
    ACTION_LATENCY = "action_latency_ms"
    CPU_PERCENT = "cpu_percent"
    MEMORY_MB = "memory_mb"
    FRAME_TIME = "frame_time_ms"


@dataclass
class PerformanceMetric:
    """Одна метрика производительности"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp
        }


class PerformanceProfiler:
    """
    Профилировщик производительности
    
    Собирает метрики в реальном времени и предоставляет статистику
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Размер окна для усреднения метрик (количество кадров)
        """
        self.window_size = window_size
        
        # История метрик
        self.frame_times: deque = deque(maxlen=window_size)
        self.gesture_latencies: deque = deque(maxlen=window_size)
        self.action_latencies: deque = deque(maxlen=window_size)
        
        # Временные метки
        self.last_frame_time = time.time()
        self.process = psutil.Process(os.getpid())
        
        # Флаги
        self.is_enabled = True
        self.metrics_history: List[PerformanceMetric] = []
        
        print("Performance Profiler инициализирован")
    
    def mark_frame_start(self) -> float:
        """Отметить начало кадра"""
        return time.time()
    
    def mark_frame_end(self, start_time: float) -> None:
        """Отметить конец кадра и записать время"""
        if not self.is_enabled:
            return
        
        frame_time = (time.time() - start_time) * 1000  # в миллисекундах
        self.frame_times.append(frame_time)
    
    def mark_gesture_start(self) -> float:
        """Отметить начало обработки жеста"""
        return time.time()
    
    def mark_gesture_end(self, start_time: float) -> None:
        """Отметить конец обработки жеста"""
        if not self.is_enabled:
            return
        
        gesture_latency = (time.time() - start_time) * 1000
        self.gesture_latencies.append(gesture_latency)
    
    def mark_action_start(self) -> float:
        """Отметить начало выполнения действия"""
        return time.time()
    
    def mark_action_end(self, start_time: float) -> None:
        """Отметить конец выполнения действия"""
        if not self.is_enabled:
            return
        
        action_latency = (time.time() - start_time) * 1000
        self.action_latencies.append(action_latency)
    
    def get_fps(self) -> float:
        """
        Получить текущие FPS
        
        Returns:
            FPS (frames per second)
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        # Среднее время кадра
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_frame_time == 0:
            return 0.0
        
        fps = 1000.0 / avg_frame_time
        return fps
    
    def get_gesture_latency(self) -> float:
        """
        Получить среднюю задержку распознавания жеста (мс)
        """
        if not self.gesture_latencies:
            return 0.0
        
        return sum(self.gesture_latencies) / len(self.gesture_latencies)
    
    def get_action_latency(self) -> float:
        """
        Получить среднюю задержку выполнения действия (мс)
        """
        if not self.action_latencies:
            return 0.0
        
        return sum(self.action_latencies) / len(self.action_latencies)
    
    def get_cpu_percent(self) -> float:
        """
        Получить процент использования CPU
        """
        try:
            return self.process.cpu_percent(interval=0.01)
        except:
            return 0.0
    
    def get_memory_mb(self) -> float:
        """
        Получить использование памяти в МБ
        """
        try:
            return self.process.memory_info().rss / 1024 / 1024  # конвертировать в МБ
        except:
            return 0.0
    
    def get_all_metrics(self) -> Dict:
        """
        Получить все текущие метрики
        
        Returns:
            {
                'fps': 28.5,
                'gesture_latency_ms': 12.3,
                'action_latency_ms': 5.2,
                'cpu_percent': 15.2,
                'memory_mb': 125.4,
                'frame_count': 30,
                'is_healthy': True
            }
        """
        fps = self.get_fps()
        
        metrics = {
            'fps': round(fps, 2),
            'gesture_latency_ms': round(self.get_gesture_latency(), 2),
            'action_latency_ms': round(self.get_action_latency(), 2),
            'cpu_percent': round(self.get_cpu_percent(), 2),
            'memory_mb': round(self.get_memory_mb(), 2),
            'frame_count': len(self.frame_times),
            'gesture_count': len(self.gesture_latencies),
            'action_count': len(self.action_latencies),
            'timestamp': time.time(),
        }
        
        # Проверить здоровье системы
        metrics['is_healthy'] = self._check_health(metrics)
        
        return metrics
    
    def _check_health(self, metrics: Dict) -> bool:
        """
        Проверить здоровье системы на основе метрик
        
        Здоровье хорошее если:
        - FPS >= 25
        - CPU < 50%
        - Memory < 500MB
        """
        fps_ok = metrics['fps'] >= 25 or metrics['fps'] == 0  # 0 если данных нет
        cpu_ok = metrics['cpu_percent'] < 50
        memory_ok = metrics['memory_mb'] < 500
        
        return fps_ok and cpu_ok and memory_ok
    
    def get_latency_summary(self) -> Dict:
        """
        Получить сводку по задержкам
        """
        def get_stats(data: deque) -> Dict:
            if not data:
                return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
            
            sorted_data = sorted(data)
            return {
                'min': round(min(data), 2),
                'max': round(max(data), 2),
                'avg': round(sum(data) / len(data), 2),
                'median': round(sorted_data[len(sorted_data) // 2], 2),
                'count': len(data)
            }
        
        return {
            'gesture_latency_ms': get_stats(self.gesture_latencies),
            'action_latency_ms': get_stats(self.action_latencies),
            'frame_time_ms': get_stats(self.frame_times),
        }
    
    def print_stats(self) -> None:
        """Вывести красивую статистику"""
        metrics = self.get_all_metrics()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"  FPS:                  {metrics['fps']:.1f}")
        print(f"  Gesture Latency:      {metrics['gesture_latency_ms']:.2f} ms")
        print(f"  Action Latency:       {metrics['action_latency_ms']:.2f} ms")
        print(f"  CPU Usage:            {metrics['cpu_percent']:.1f}%")
        print(f"  Memory Usage:         {metrics['memory_mb']:.1f} MB")
        print(f"  Health Status:        {'HEALTHY' if metrics['is_healthy'] else 'SLOW'}")
        print("=" * 60 + "\n")
    
    def print_latency_summary(self) -> None:
        """Вывести сводку по задержкам"""
        summary = self.get_latency_summary()
        
        print("\n" + "=" * 60)
        print("LATENCY SUMMARY")
        print("=" * 60)
        
        for latency_type, stats in summary.items():
            if stats['count'] > 0:
                print(f"\n{latency_type}:")
                print(f"  Min:    {stats['min']:.2f} ms")
                print(f"  Max:    {stats['max']:.2f} ms")
                print(f"  Avg:    {stats['avg']:.2f} ms")
                print(f"  Median: {stats['median']:.2f} ms")
                print(f"  Count:  {stats['count']}")
    
    def check_health(self) -> Dict[str, any]:
        """
        Проверить здоровье системы
        
        Returns:
            {
                'status': 'healthy|warning|critical',
                'fps': fps_value,
                'issues': [list of issues]
            }
        """
        issues = []
        fps = self.get_fps()
        cpu = self.get_cpu_percent()
        memory = self.get_memory_mb()
        
        # Проверить критерии
        if fps < 20:
            issues.append("Low FPS")
        if cpu > 80:
            issues.append("High CPU")
        if memory > 500:
            issues.append("High Memory")
        
        # Определить статус
        if not issues:
            status = 'healthy'
        elif len(issues) == 1:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'fps': fps,
            'cpu': cpu,
            'memory': memory,
            'issues': issues
        }

        
        print("=" * 60 + "\n")
    
    def reset(self) -> None:
        """Сбросить все метрики"""
        self.frame_times.clear()
        self.gesture_latencies.clear()
        self.action_latencies.clear()
        print("Метрики сброшены")
    
    def enable(self) -> None:
        """Включить профилирование"""
        self.is_enabled = True
    
    def disable(self) -> None:
        """Отключить профилирование"""
        self.is_enabled = False


# Глобальный экземпляр для convenience
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Получить или создать глобальный профилировщик"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


if __name__ == "__main__":
    print("Testing Performance Profiler...")
    
    profiler = PerformanceProfiler(window_size=20)
    
    # Имитировать несколько кадров
    for i in range(20):
        frame_start = profiler.mark_frame_start()
        
        # Имитировать обработку жеста
        gesture_start = profiler.mark_gesture_start()
        time.sleep(0.01)  # 10ms
        profiler.mark_gesture_end(gesture_start)
        
        # Имитировать выполнение действия
        action_start = profiler.mark_action_start()
        time.sleep(0.005)  # 5ms
        profiler.mark_action_end(action_start)
        
        # Завершить кадр
        time.sleep(0.025)  # 25ms total
        profiler.mark_frame_end(frame_start)
    
    # Вывести результаты
    profiler.print_stats()
    profiler.print_latency_summary()
