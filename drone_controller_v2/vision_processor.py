from ultralytics import YOLO
import time
from .utils.timing_logger import timing_logger

class VisionProcessor:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Using the small model for better accuracy

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
        # Time YOLO inference
        yolo_start = time.time()
        results = self.model(frame, verbose=False)
        timing_logger.log_time('yolo_inference', (time.time() - yolo_start) * 1000)

        # Time post-processing
        post_start = time.time()
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

        timing_logger.log_time('vision_post_processing', (time.time() - post_start) * 1000)
        return '\n'.join(descriptions) if descriptions else 'no objects detected' 