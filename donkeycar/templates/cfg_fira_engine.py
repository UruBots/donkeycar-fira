# AprilTag detection frequency (Hz)
APRILTAG_HZ = 2

# Zebra crosswalk detection frequency (Hz)
ZEBRA_HZ = 1

# Stop duration (seconds)
STOP_DURATION = 5

# Turn duration (seconds)
TURN_DURATION = 2

# Proceed duration (seconds)
PROCEED_DURATION = 3

# Maximum turn throttle value
MAX_THROTTLE = 1.0

# Maximum turn angle (degrees)
MAX_ANGLE = 30.0

# Show bounding box around detected objects
DEBUG_VISUALS = True

# Debug mode (prints debug information)
DEBUG = True

# Proximity thresholds for each AprilTag ID
PROXIMITY_THRESHOLDS = {
    0: 0.1,  # STOP
    1: 0.1,  # DEAD_END
    2: 0.1,  # TURN_LEFT
    3: 0.1,  # TURN_RIGHT
    4: 0.1   # FORWARD
}

# Tag dictionary mapping AprilTag IDs to traffic sign names
TAG_DICT = {1: 'STOP', 2: 'DEAD_END', 3: 'TURN_RIGHT', 4: 'TURN_LEFT', 4: 'FORWARD', 5:'STOP'}

