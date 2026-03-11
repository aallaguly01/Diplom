# gesture_framework/core/platform_abstraction.py
"""
Абстракция для управления системой - кроссплатформенный API
для Windows, Linux и MacOS
"""

import os
import platform
from typing import Tuple
import subprocess


class PlatformController:
    """
    Кроссплатформенный контроллер для управления системой
    
    Поддерживает:
    - Движение мыши
    - Клики мыши
    - Скроллинг
    - Нажатие клавиш
    - Открытие приложений
    """
    
    def __init__(self):
        """Инициализировать контроллер"""
        self.platform = self.get_platform()
        
        # Импортировать нужные библиотеки в зависимости от ОС
        try:
            from pynput.mouse import Controller as MouseController
            from pynput.keyboard import Controller as KeyboardController
            self.mouse = MouseController()
            self.keyboard = KeyboardController()
        except ImportError:
            raise ImportError(
                "Требуется установить pynput: pip install pynput"
            )
    
    def move_cursor(self, x: int, y: int) -> None:
        """
        Переместить курсор мыши
        
        Args:
            x: Координата X на экране
            y: Координата Y на экране
        """
        try:
            self.mouse.position = (x, y)
        except Exception as e:
            print(f"Ошибка при движении мыши: {e}")
    
    def click_mouse(self, button: str = 'left', count: int = 1) -> None:
        """
        Нажать кнопку мыши
        
        Args:
            button: 'left', 'right' или 'middle'
            count: Количество кликов
        """
        try:
            from pynput.mouse import Button
            
            buttons = {
                'left': Button.left,
                'right': Button.right,
                'middle': Button.middle,
            }
            
            if button not in buttons:
                print(f"Неизвестная кнопка мыши: {button}")
                return
            
            for _ in range(count):
                self.mouse.click(buttons[button])
        except Exception as e:
            print(f"Ошибка при клике мыши: {e}")
    
    def scroll(self, direction: str = 'up', amount: int = 1) -> None:
        """
        Скроллинг экрана
        
        Args:
            direction: 'up', 'down', 'left' или 'right'
            amount: Количество скроллов
        """
        try:
            if direction == 'up':
                self.mouse.scroll(0, amount)
            elif direction == 'down':
                self.mouse.scroll(0, -amount)
            elif direction == 'left':
                self.mouse.scroll(-amount, 0)
            elif direction == 'right':
                self.mouse.scroll(amount, 0)
            else:
                print(f"Неизвестное направление скроллинга: {direction}")
        except Exception as e:
            print(f"Ошибка при скроллинге: {e}")
    
    def press_key(self, key: str) -> None:
        """
        Нажать одну клавишу
        
        Args:
            key: Название клавиши (например 'a', 'return', 'space')
        """
        try:
            from pynput.keyboard import Key
            
            # Попытаться получить клавишу по названию
            if hasattr(Key, key):
                key_obj = getattr(Key, key)
            else:
                key_obj = key
            
            self.keyboard.press(key_obj)
            self.keyboard.release(key_obj)
        except Exception as e:
            print(f"Ошибка при нажатии клавиши: {e}")
    
    def press_hotkey(self, *keys: str) -> None:
        """
        Нажать комбинацию клавиш
        
        Args:
            *keys: Названия клавиш (например 'ctrl', 's')
        """
        try:
            from pynput.keyboard import Key
            
            key_objects = []
            for k in keys:
                if hasattr(Key, k):
                    key_objects.append(getattr(Key, k))
                else:
                    key_objects.append(k)
            
            # Нажать все
            for key_obj in key_objects:
                self.keyboard.press(key_obj)
            
            # Отпустить в обратном порядке
            for key_obj in reversed(key_objects):
                self.keyboard.release(key_obj)
        except Exception as e:
            print(f"Ошибка при нажатии комбинации клавиш: {e}")
    
    def type_text(self, text: str) -> None:
        """
        Напечатать текст
        
        Args:
            text: Текст для ввода
        """
        try:
            self.keyboard.type(text)
        except Exception as e:
            print(f"Ошибка при вводе текста: {e}")
    
    def launch_app(self, app_name: str) -> None:
        """
        Запустить приложение
        
        Args:
            app_name: Имя или путь до приложения
        """
        try:
            if self.platform == 'Windows':
                os.startfile(app_name)
            elif self.platform == 'Darwin':  # macOS
                subprocess.Popen(['open', app_name])
            elif self.platform == 'Linux':
                subprocess.Popen([app_name])
        except Exception as e:
            print(f"Ошибка при запуске приложения: {e}")
    
    @staticmethod
    def get_screen_size() -> Tuple[int, int]:
        """
        Получить размер экрана
        
        Returns:
            Кортеж (width, height) в пикселях
        """
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception as e:
            print(f"Ошибка при получении размера экрана: {e}")
            return 1920, 1080  # Значение по умолчанию
    
    @staticmethod
    def get_platform() -> str:
        """
        Получить текущую ОС
        
        Returns:
            'Windows', 'Linux' или 'Darwin' (macOS)
        """
        system = platform.system()
        if system == 'Windows':
            return 'Windows'
        elif system == 'Linux':
            return 'Linux'
        elif system == 'Darwin':
            return 'Darwin'
        else:
            return system
