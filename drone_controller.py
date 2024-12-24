import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller
import google.generativeai as genai
import base64
from io import BytesIO
from dotenv import load_dotenv
import os
import threading
import queue

class DroneController:
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

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def capture_screen():
    """Capture the simulator screen"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Main monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def encode_image(frame):
    """Convert frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def parse_command(command_text):
    """Parse command text into action and duration"""
    # Split the command into parts
    parts = command_text.lower().split()
    
    # Check for standard action patterns
    for i in range(len(parts)):
        if parts[i].endswith('ms'):
            try:
                action = ' '.join(parts[:i])
                duration = int(parts[i].replace('ms', ''))
                return (action, duration)
            except ValueError:
                continue
    return None

def process_drone_command(current_frame, original_command, last_execution=None):
    """Process drone command through Gemini"""
    
    # Prepare the prompt with more detailed instructions
    prompt = f"""
    Task: {original_command}
    Last Action: {last_execution if last_execution else 'None'}

    You are controlling a drone in a simulator. Analyze the current view and determine the next action.
    


    Available Actions:
    - increase throttle
    - decrease throttle
    - yaw left
    - yaw right
    - roll left
    - roll right
    - pitch forward
    - pitch back

    
    Response Format:
    1. If goal is not met, respond with exactly one action and duration like:
       "pitch forward 500ms" or "yaw left 1000ms"
    
    2. If goal is met (yellow house reached), respond with exactly:
       "command fulfilled standby"
    
    3. If you need to provide status, start with "STATUS:" followed by your observation
       Example: "STATUS: Approaching yellow house, adjusting altitude"

    Important:
    - Provide only ONE action at a time
    - Always include duration in milliseconds (ms)
    - Keep durations between 100ms and 2000ms for safety
    - Consider momentum and avoid overcompensation
    """
    
    # Convert frame to base64
    image_b64 = encode_image(current_frame)
    
    # Send to Gemini
    response = model.generate_content([
        prompt,
        {'mime_type': 'image/jpeg', 'data': image_b64}
    ])
    
    response_text = response.text.strip()
    
    # Handle status messages
    if response_text.startswith('STATUS:'):
        print(response_text)
        return last_execution or 'waiting'  # Maintain last action if status update
    
    return response_text

def main():
    """Main control loop"""
    drone_controller = DroneController()
    current_command = input("Enter high-level command (e.g., 'take off'): ")
    last_execution = None
    
    try:
        while True:
            # Capture current view
            frame = capture_screen()
            
            # Get next action from Gemini
            response = process_drone_command(frame, current_command, last_execution)
            
            if response == 'command fulfilled standby':
                print("Command completed. Waiting for next command...")
                current_command = input("Enter next command: ")
                last_execution = None
                continue
                
            # Parse and execute the command
            action_tuple = parse_command(response)
            if action_tuple:
                drone_controller.execute_action(action_tuple)
                last_execution = response
                time.sleep(0.1)  # Small delay between commands
    
    finally:
        drone_controller.stop()
        
if __name__ == "__main__":
    main()