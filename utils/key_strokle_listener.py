import keyboard 

class keyStrokeListener:
    def __init__(self):
        self.keys_pressed = set()
    
    def add_hotkey(self, hot_key:str):
        keyboard.add_hotkey(hot_key, self._key_pressed_callback, args=[hot_key])

    def _key_pressed_callback(self, key:str):
        self.keys_pressed.add(key)
        print(f"Key '{key}' was pressed.")

    def has_key_pressed(self, key:str)->bool:
        return key in self.keys_pressed
    
    def clear_keys(self):
        self.keys_pressed.clear()
    
    def clear_key(self, key:str):
        if key in self.keys_pressed:
            self.keys_pressed.remove(key)
