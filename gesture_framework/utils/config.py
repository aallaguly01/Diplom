"""
Configuration Manager - управление параметрами приложения

Поддерживает JSON и YAML форматы для конфигурации
"""

import json
import os
from typing import Any, Dict, Optional, List
from pathlib import Path


class ConfigManager:
    """
    Менеджер конфигурации приложения
    
    Загружает и сохраняет параметры из JSON/YAML файлов
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Args:
            config_path: Путь к файлу конфигурации (JSON или YAML)
        """
        self.config_path = Path(config_path)
        self.config: Dict = {}
        self._defaults = self._get_defaults()
        
        # Загрузить конфигурацию если файл существует
        if self.config_path.exists():
            self.load()
        else:
            self.config = self._defaults.copy()
        
        print(f"ConfigManager инициализирован: {self.config_path}")
    
    @staticmethod
    def _get_defaults() -> Dict:
        """Получить конфигурацию по умолчанию"""
        return {
            "engine": {
                "fps_target": 30,
                "gesture_smoothing_window": 5,
                "confidence_threshold": 0.6,
                "min_gesture_interval": 0.1,
                "min_click_interval": 0.3,
                "use_kalman_filter": True,
            },
            "gestures": {
                "POINTING": {
                    "enabled": True,
                    "action": "move_cursor",
                    "confidence_threshold": 0.7
                },
                "PINCH": {
                    "enabled": True,
                    "action": "click",
                    "confidence_threshold": 0.7
                },
                "OPEN_PALM": {
                    "enabled": True,
                    "action": "pause",
                    "confidence_threshold": 0.8
                },
                "CLOSED_FIST": {
                    "enabled": True,
                    "action": "hold",
                    "confidence_threshold": 0.8
                },
                "PEACE": {
                    "enabled": True,
                    "action": "right_click",
                    "confidence_threshold": 0.75
                }
            },
            "voice": {
                "enabled": True,
                "language": "ru",
                "continuous_listening": False,
                "model_path": "model"
            },
            "performance": {
                "profiler_enabled": True,
                "profiler_window_size": 30,
                "cpu_limit_percent": 50,
                "memory_limit_mb": 300
            },
            "ui": {
                "show_landmarks": False,
                "show_metrics": True,
                "show_gesture_history": False,
                "theme": "dark"
            }
        }
    
    def load(self) -> bool:
        """
        Загрузить конфигурацию из файла
        
        Returns:
            True если успешно, False если ошибка
        """
        try:
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Try to import yaml
                try:
                    import yaml
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                except ImportError:
                    print("[WARN] YAML не установлен, используем JSON")
                    return False
            
            print(f"[OK] Конфигурация загружена из {self.config_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки конфигурации: {e}")
            self.config = self._defaults.copy()
            return False
    
    def save(self) -> bool:
        """
        Сохранить конфигурацию в файл
        
        Returns:
            True если успешно, False если ошибка
        """
        try:
            # Создать директорию если её нет
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                # Try YAML
                try:
                    import yaml
                    with open(self.config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    # Fall back to JSON
                    json_path = self.config_path.with_suffix('.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=2, ensure_ascii=False)
                    self.config_path = json_path
            
            print(f"[OK] Конфигурация сохранена в {self.config_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения конфигурации: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение из конфигурации
        
        Args:
            key: Ключ в формате "section.key" (e.g., "engine.fps_target")
            default: Значение по умолчанию если ключ не найден
        
        Returns:
            Значение из конфига или default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Установить значение в конфигурацию
        
        Args:
            key: Ключ в формате "section.key"
            value: Новое значение
        """
        keys = key.split('.')
        config = self.config
        
        # Создать структуру если её нет
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        print(f"Установлено {key} = {value}")
    
    def get_section(self, section: str) -> Dict:
        """
        Получить весь раздел конфигурации
        
        Args:
            section: Имя раздела (e.g., "engine", "gestures")
        
        Returns:
            Словарь с параметрами раздела
        """
        return self.config.get(section, {})
    
    def get_all(self) -> Dict:
        """Получить всю конфигурацию"""
        return self.config.copy()
    
    def reset_to_defaults(self) -> None:
        """Сбросить конфигурацию на значения по умолчанию"""
        self.config = self._defaults.copy()
        print("Конфигурация сброшена на значения по умолчанию")
    
    def validate(self) -> List[str]:
        """
        Валидировать конфигурацию
        
        Returns:
            Список найденных ошибок (пустой если всё OK)
        """
        errors = []
        
        # Проверить обязательные поля
        required_sections = ['engine', 'gestures', 'voice']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing section: {section}")
        
        # Проверить типы значений
        engine_config = self.get_section('engine')
        if 'fps_target' in engine_config and not isinstance(engine_config['fps_target'], (int, float)):
            errors.append("engine.fps_target должен быть числом")
        
        if 'confidence_threshold' in engine_config:
            threshold = engine_config['confidence_threshold']
            if not (0.0 <= threshold <= 1.0):
                errors.append("engine.confidence_threshold должен быть между 0.0 и 1.0")
        
        return errors
    
    def print_config(self) -> None:
        """Вывести конфигурацию в красивом виде"""
        print("\n" + "=" * 60)
        print("ТЕКУЩАЯ КОНФИГУРАЦИЯ")
        print("=" * 60)
        
        def print_dict(d: Dict, indent: int = 0) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                else:
                    print("  " * indent + f"  {key}: {value}")
        
        print_dict(self.config)
        print("=" * 60 + "\n")
    
    def merge(self, other_config: Dict) -> None:
        """
        Объединить с другой конфигурацией
        
        Args:
            other_config: Другой словарь конфигурации (перезапишет существующие значения)
        """
        self._merge_dict(self.config, other_config)
        print("Конфигурация объединена")
    
    @staticmethod
    def _merge_dict(base: Dict, overlay: Dict) -> None:
        """Рекурсивно объединить словари"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigManager._merge_dict(base[key], value)
            else:
                base[key] = value


if __name__ == "__main__":
    print("Testing ConfigManager...")
    
    # Создать конфиг
    config = ConfigManager('test_config.json')
    
    # Вывести конфиг
    config.print_config()
    
    # Изменить значение
    config.set('engine.fps_target', 60)
    config.set('gestures.POINTING.enabled', False)
    
    # Получить значение
    fps = config.get('engine.fps_target')
    print(f"FPS target: {fps}")
    
    # Валидировать
    errors = config.validate()
    if errors:
        print("[ERROR] Ошибки валидации:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("[OK] Конфигурация валидна")
    
    # Сохранить
    config.save()
    
    print("\n[OK] ConfigManager test completed")
