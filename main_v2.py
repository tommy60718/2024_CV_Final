import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from drone_controller_v2.drone_controller_v2 import main

if __name__ == "__main__":
    main() 