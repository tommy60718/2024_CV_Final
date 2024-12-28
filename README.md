---

# AI Drone Controller

An AI-powered drone controller that integrates real-time object detection (YOLOv8), depth estimation (Depth-Anything), and Google's Gemini AI to control a drone in a simulator.

## License Notice

⚠️ **PROPRIETARY SOFTWARE** ⚠️

This software is proprietary and closed-source. All rights reserved.  
No part of this software may be used, copied, modified, or distributed without express written permission.  
See LICENSE file for details.

## Features

- **Screen Capture**: Real-time simulator screen capture using `mss`.
- **Object Detection**: Detect objects and their positions with YOLOv8.
- **Depth Estimation**: Estimate object depth using Depth-Anything (DINOv2).
- **AI-Powered Decision Making**: Use Google Gemini AI to parse commands and generate control actions.
- **Keyboard Control**: Control the simulator through keyboard inputs.
- **Multi-threaded Execution**: Concurrent handling of action execution and keyboard input.
- **Command Queue System**: Efficient action queuing and processing.
- **Scene Description**: Generate detailed scene descriptions (object class, location, and depth).

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Drone Simulator
- Pre-trained YOLOv8 model weights
- Depth-Anything (DINOv2) setup

## Simulator Setup

1. Install The Drone Racing League Simulator from Steam.
2. Launch the simulator and go to Settings.
3. Set difficulty to Easy.
4. Enable "Height Automation" for better control.
5. Keep the simulator window visible on your main monitor.

## Installation

1. Clone the repository (requires authorization):
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and set up the Depth-Anything library:
   - Clone Depth-Anything into the project directory.
   - Set the `torchhub` path in the script to match your directory structure.
4. Ensure you have the YOLOv8 weights (e.g., `yolov8s.pt`).

## Configuration

1. Add your Google Gemini API key to an `.env` file:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
2. Verify all paths for Depth-Anything and YOLOv8 models are correct in the script.

## Running the Program

1. Start the simulator and position it on the primary monitor.
2. Run the script:
   ```bash
   python main_v2.py
   ```
3. Enter high-level commands when prompted (e.g., `"fly to the yellow house"`).
4. Monitor console logs for scene descriptions, detected objects, and actions.

## New Features in This Version

- **YOLOv8 Integration**: High-speed object detection for enhanced scene understanding.
- **Depth-Anything (DINOv2)**: Detailed depth maps for spatial awareness.
- **3×3 Grid Mapping**: Localize objects within a grid-based representation of the scene.
- **Enhanced Gemini Prompt Engineering**: Improved task-specific prompts for precise action generation.
- **Thread-Safe Action Queue**: Reliable handling of keyboard control actions.

## Notes

- Ensure your GPU drivers and PyTorch installation support YOLOv8 and Depth-Anything for optimal performance.
- For detailed depth estimation, maintain consistent lighting and visibility in the simulator environment.

---

This README reflects the updated features and ensures clarity for users setting up and running the project.
