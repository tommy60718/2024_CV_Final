import time
import queue
import threading
from pynput.keyboard import Key, Controller
from .utils.timing_logger import timing_logger

class KeyboardController:
    def __init__(self):
        self.keyboard = Controller()
        self.action_queue = queue.Queue()
        self.running = True

        # Start keyboard control thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_control_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

    def _keyboard_control_loop(self):
        """Separate thread for keyboard control"""
        while self.running:
            try:
                action = self.action_queue.get(timeout=0.1)
                if action:
                    self._execute_action(action)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Keyboard control error: {e}")

    def _execute_action(self, action_tuple):
        """Execute a single action with duration"""
        start_time = time.time()
        action, duration_ms = action_tuple

        action_map = {
            'increase throttle': 'w',
            'decrease throttle': 's',
            'yaw left': 'a',
            'yaw right': 'd',
            'roll left': Key.left,
            'roll right': Key.right,
            'pitch forward': Key.up,
            'pitch back': Key.down
        }

        if action in action_map:
            key = action_map[action]
            try:
                print(f"Executing {action} for {duration_ms}ms")
                self.keyboard.press(key)
                time.sleep(duration_ms / 1000.0)
                self.keyboard.release(key)
                time.sleep(0.1)
                
                # Log action execution time
                timing_logger.log_time('action_execution', (time.time() - start_time) * 1000)
            except Exception as e:
                print(f"Keyboard action failed: {e}")
                self.keyboard.release(key)

    def execute_action(self, action_tuple):
        """Add action to queue"""
        self.action_queue.put(action_tuple)

    def stop(self):
        """Stop the keyboard control thread"""
        self.running = False
        self.keyboard_thread.join() 