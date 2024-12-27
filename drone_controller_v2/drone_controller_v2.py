#This version use YOLO object detection as image input
# the image input is present using 3x3 grid
# We need more granular coordinate & granular control
# We need depth to control the forward and backward.    

import cv2
import numpy as np
import mss
import time
from drone_controller_v2.keyboard_controller import KeyboardController
from drone_controller_v2.vision_processor import VisionProcessor
from drone_controller_v2.command_processor import process_drone_command, parse_command
from drone_controller_v2.utils.timing_logger import timing_logger

def capture_screen():
    """Capture the simulator screen"""
    start_time = time.time()
    with mss.mss() as sct:
        monitor = sct.monitors[2]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        result = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    timing_logger.log_time('screen_capture', (time.time() - start_time) * 1000)
    return result

def print_timing_info():
    """Print formatted timing information"""
    print("\n=== Average Performance (ms) ===")
    components = ['yolo_inference', 'gemini_response', 'action_execution', 'total_iteration']
    
    for component in components:
        stats = timing_logger.get_stats(component)
        if stats:
            print(f"{component}: {stats['avg']:.2f}")
    print("==============================\n")

def main():
    """Main control loop"""
    drone_controller = KeyboardController()
    vision_processor = VisionProcessor()
    current_command = input("Enter high-level command (e.g., 'fly to the yellow house'): ")
    last_execution = None
    iteration_count = 0

    try:
        while True:
            iteration_start = time.time()
            
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

            # Print timing statistics every 10 iterations
            iteration_count += 1
            if iteration_count % 3 == 0:
                print_timing_info()
            
            # Log total iteration time
            timing_logger.log_time('total_iteration', (time.time() - iteration_start) * 1000)

    finally:
        drone_controller.stop()

if __name__ == "__main__":
    main()
