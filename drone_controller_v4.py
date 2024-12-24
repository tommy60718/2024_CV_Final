# This version integrate YOLO, depth anything

import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller
import google.generativeai as genai
from ultralytics import YOLO
from torchvision.transforms import Compose
from dotenv import load_dotenv
import os
import threading
import queue
import torch

import sys
# Adjust the path to where Depth-Anything is located
depth_anything_path = os.path.abspath('./Depth-Anything')
sys.path.append(depth_anything_path)
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Explicitly set the torchhub path to include the 'torchhub' directory
torch.hub.set_dir(os.path.abspath('./Depth-Anything/torchhub'))
print(os.path.abspath('./Depth-Anything/torchhub/facebookresearch_dinov2_main/hubconf.py'))

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

class VisionProcessor:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolov8s.pt')

        # Explicitly set the path to the Depth-Anything model
        depth_anything_path = os.path.abspath('./Depth-Anything/torchhub/facebookresearch_dinov2_main')

        # Load Depth-Anything model explicitly
        self.depth_model = torch.hub.load(
            depth_anything_path,
            'dinov2_vits14',
            source='local',  # Load from local path
            pretrained=False  # Avoid downloading pretrained weights (if not required)
        ).eval()

        print("Depth-Anything model loaded successfully!")

        # Depth processing transform
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def get_position_in_grid(self, x_center, y_center, frame_width, frame_height):
        """Determine the 3×3 grid position of the object based on its center coordinates."""
        x_section = frame_width / 3
        y_section = frame_height / 3

        col = int(x_center // x_section) + 1
        row = int(y_center // y_section) + 1

        return f"row{row}-col{col}"

    def get_depth_map(self, frame):
        """Estimate depth map from the frame."""
        image = frame / 255.0
        processed_image = self.transform({'image': image})['image']
        image_tensor = torch.from_numpy(processed_image).unsqueeze(0)
        depth = self.depth_model(image_tensor)
        return depth.squeeze().detach().numpy()

    def get_scene_description(self, frame):
        """Process the frame and return a detailed scene description with grid positions and depth information."""
        results = self.model(frame, verbose=False)
        frame_height, frame_width, _ = frame.shape
        descriptions = []

        # Get depth map
        depth_map = self.get_depth_map(frame)

        for result in results:
            for box in result.boxes:
                # Extract object details
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf[0]) * 100

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                position = self.get_position_in_grid(x_center, y_center, frame_width, frame_height)

                # Extract average depth for the bounding box
                box_depth = depth_map[y_min:y_max, x_min:x_max].mean() if depth_map.size > 0 else -1
                depth_info = f"{box_depth:.2f} (relative depth)" if box_depth > 0 else "N/A"

                # Generate a detailed description for the object
                description = f"{cls_name}, {confidence:.1f}% certainty, position: {position}, depth: {depth_info}"
                descriptions.append(description)

        return '\n'.join(descriptions) if descriptions else 'no objects detected'

def parse_command(command_text):
    """Parse command text into action and duration"""
    parts = command_text.lower().split()
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
    1. If the goal is not met, respond with exactly one action and duration like:
       "pitch forward 200ms" or "yaw left 500ms"

    2. If the goal is met (e.g., the specified destination is reached), respond with exactly:
       "command fulfilled standby"
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith('STATUS:'):
            print(response_text)
            return last_execution or 'waiting'

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

            action_tuple = parse_command(response)
            if action_tuple:
                drone_controller.execute_action(action_tuple)
                last_execution = response
                time.sleep(0.1)

    finally:
        drone_controller.stop()

if __name__ == "__main__":
    main()
