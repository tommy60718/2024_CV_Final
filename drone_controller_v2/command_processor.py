import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from .utils.timing_logger import timing_logger

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
#model = genai.GenerativeModel('gemini-2.0-flash-exp')

model = genai.GenerativeModel('gemini-1.5-flash-latest')
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
    start_time = time.time()
    
    # Time the scene description generation
    scene_desc_start = time.time()
    scene_description = vision_processor.get_scene_description(current_frame)
    timing_logger.log_time('scene_description', (time.time() - scene_desc_start) * 1000)

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
       "increase throttle 800ms" "pitch forward 200ms" or "yaw left 500ms"

    2. If the goal is met (e.g., the specified destination is reached), respond with exactly:
       "command fulfilled standby"

    3. If you need to provide status, start with "STATUS:" followed by your observation
        Example: "OBJECTS: red clock in row1-col1, cloud in row3-col3"
        Example: "STATUS: Approaching the target in row2-col3, adjusting altitude"
    """

    try:
        # Time the Gemini API response
        gemini_start = time.time()
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        timing_logger.log_time('gemini_response', (time.time() - gemini_start) * 1000)

        # Handle status messages
        if response_text.startswith('STATUS:'):
            print(response_text)
            return last_execution or 'waiting'  # Maintain last action if it's a status update

        # Log total command processing time
        timing_logger.log_time('total_command_processing', (time.time() - start_time) * 1000)
        return response_text

    except Exception as e:
        print(f"Error processing command: {e}")
        return 'Error' 