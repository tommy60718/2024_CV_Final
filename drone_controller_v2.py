# This version use YOLO object detection as image input
# We need more granular coordinate & granular control
# We need depth to control the forward and backward.
import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller
import google.generativeai as genai
from ultralytics import YOLO
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

# class VisionProcessor:
#     def __init__(self):
#         self.model = YOLO('yolov8n.pt')  # Using the nano model for speed

#     def get_scene_description(self, frame):
#         """Process the frame and return a scene description."""
#         results = self.model(frame, verbose=False)
#         descriptions = []
#         for result in results:
#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 cls_name = self.model.names[cls_id]
#                 descriptions.append(cls_name)
#         unique_descriptions = list(set(descriptions))
#         return ', '.join(unique_descriptions) if unique_descriptions else 'no objects detected'

class VisionProcessor:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')  # Using the small model for better accuracy

    def get_position_in_grid(self, x_center, y_center, frame_width, frame_height):
        """
        Determine the 3×3 grid position of the object based on its center coordinates.
        """
        # Divide the frame into a 3×3 grid
        x_section = frame_width / 3
        y_section = frame_height / 3

        # Calculate the row and column of the grid (1-based indexing)
        col = int(x_center // x_section) + 1
        row = int(y_center // y_section) + 1

        return f"row{row}-col{col}"

    def get_scene_description(self, frame):
        """
        Process the frame and return a detailed scene description with grid positions.
        """
        results = self.model(frame, verbose=False)
        frame_height, frame_width, _ = frame.shape  # Get frame dimensions
        descriptions = []

        for result in results:
            for box in result.boxes:
                # Extract object details
                cls_id = int(box.cls[0])  # Class ID
                cls_name = self.model.names[cls_id]  # Class name
                confidence = float(box.conf[0]) * 100  # Confidence in percentage

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Convert to integers

                # Calculate the center of the bounding box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # Determine the grid position of the object
                position = self.get_position_in_grid(x_center, y_center, frame_width, frame_height)

                # Generate a detailed description for the object
                description = (
                    f"{cls_name}, {confidence:.1f}% certainty, position: {position}"
                )
                descriptions.append(description)

        return '\n'.join(descriptions) if descriptions else 'no objects detected'

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

def process_drone_command(vision_processor, current_frame, original_command, last_execution=None):
    """Process drone command using scene descriptions."""
    # Generate scene description
    scene_description = vision_processor.get_scene_description(current_frame)

    # Prepare the prompt with detailed instructions
    prompt = f"""
    Task: {original_command}
    Scene: {scene_description}
    Last Action: {last_execution if last_execution else 'None'}

    The simulator scene is divided into a 3×3 grid:
    - row1-col1: top-left
    - row1-col2: top-center
    - row1-col3: top-right
    - row2-col1: middle-left
    - row2-col2: middle-center
    - row2-col3: middle-right
    - row3-col1: bottom-left
    - row3-col2: bottom-center
    - row3-col3: bottom-right

    Your goal is to control the drone to complete the specified task by interpreting the scene description and object positions in the grid.

    Available Actions:
    - increase throttle
    - decrease throttle
    - yaw left
    - yaw right
    - roll left
    - roll right
    - pitch forward
    - pitch back

    tips:
    - If you yaw left, the scene of object will go right, but the absolute position of drone won't change
    - If you roll left, the object will go right, and the absolute position of drone will go left
    - If you increase trottle, the object will go lower, and the absolute position of drone will go up
    - If you pitch forward, the object get closer, and the absolute position of drone will go up

    Response Format:
    1. If the goal is not met, respond with exactly one action and duration like:
       "pitch forward 200ms" or "yaw left 500ms"

    2. If the goal is met (e.g., the specified destination is reached), respond with exactly:
       "command fulfilled standby"

    3. If you need to provide status, start with "STATUS:" followed by your observation
        Example: "OBJECTS: red clock in row1-col1, cloud in row3-col3"
        Example: "STATUS: Approaching the target in row2-col3, adjusting altitude"

    Important:
    - Provide only ONE action at a time
    - Always include duration in milliseconds (ms)
    - Keep durations between 100ms and 1000ms for safety
    - Consider object position and momentum to avoid overcompensation
    """

    try:
        # Use the correct method to generate text
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Handle status messages
        if response_text.startswith('STATUS:'):
            print(response_text)
            return last_execution or 'waiting'  # Maintain last action if it's a status update

        return response_text

    except Exception as e:
        print(f"Error processing command: {e}")
        return 'Error'


def main():
    """Main control loop"""
    drone_controller = DroneController()
    vision_processor = VisionProcessor()
    current_command = input("Enter high-level command (e.g., 'fly to the yellow house'): ")
    last_execution = None

    try:
        while True:
            # Capture current view
            frame = capture_screen()

            # Print object information
            print("\n--- Detected Objects ---")
            scene_description = vision_processor.get_scene_description(frame)
            print(f"Scene Description:\n{scene_description}")
            print("-------------------------\n")

            # Get next action using scene description
            response = process_drone_command(vision_processor, frame, current_command, last_execution)

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
            else:
                print(f"Invalid response from LLM: {response}")

    finally:
        drone_controller.stop()


if __name__ == "__main__":
    main()
