import os

from donkeycar.templates.cfg_complete import *

# Competition profile selector: urban | race
FIRA_COMPETITION_MODE = os.getenv('FIRA_COMPETITION_MODE', 'urban').strip().lower()

# Competition operations: keep control fully onboard.
# Rule 2.8.1 prohibits wireless communication for controlling/communicating
# with the vehicle during competition.
WEB_CONTROL_ENABLED = False
WEBRTC_ENABLED = False
WEBRTC_ICE_SERVERS = []
USE_FPV = False
USE_NETWORKED_JS = False

# Start autonomous by default in competition profile.
WEB_INIT_MODE = 'local'
AI_LAUNCH_KEEP_ENABLED = False

# Safety defaults for both modes.
FIRA_REQUIRE_ZEBRA = True
FIRA_EXECUTE_ON_TAG_LOSS = False
FIRA_SAFETY_ARBITER = True
FIRA_HEALTH_TELEMETRY = True
FIRA_COMPETITION_METRICS = True
FIRA_COMPETITION_REPORT_ENABLED = True
FIRA_COMPETITION_REPORT_MIN_ACTIVE_FRAMES = 120

# Keep off by default unless mode enables them explicitly.
FIRA_MODULAR = False
FIRA_CHALLENGE_ENABLED = False
FIRA_ENGINE = False
FIRA_ENGINE_YOLO = False
FIRA_ENGINE_TF = False

if FIRA_COMPETITION_MODE == 'urban':
    # Urban: sign navigation + zebra/stop behavior.
    FIRA_MODULAR = True
    FIRA_CHALLENGE_ENABLED = False
    FIRA_SAFETY_SIGNAL_ESTIMATOR = True
    FIRA_PROFILE_SCHEDULER_ENABLED = False
    FIRA_CHECKPOINT_DETECTOR_ENABLED = False
    FIRA_SAFETY_URBAN_CONE_PROFILE = 'SAFE'
    AI_THROTTLE_MULT = 0.55

    # Urban compliance-oriented defaults.
    FIRA_REQUIRE_ZEBRA = True
    FIRA_REQUIRE_STOP_LINE = True
    FIRA_STOP_LINE_MIN_DIST_PX = 6
    FIRA_STOP_LINE_MAX_DIST_PX = 18
    FIRA_EXECUTE_ON_TAG_LOSS = False

    # Keep challenge anti-DNF guard active globally, even if challenge is off.
    FIRA_CHALLENGE_COLLISION_IMMINENT_ENABLED = True
    FIRA_CHALLENGE_COLLISION_IMMINENT_PROXIMITY = 0.88
    FIRA_CHALLENGE_COLLISION_IMMINENT_OCCUPANCY = 0.42
    FIRA_CHALLENGE_COLLISION_IMMINENT_THROTTLE_CAP = 0.02

elif FIRA_COMPETITION_MODE == 'race':
    # Race: lane + obstacle challenge behavior.
    FIRA_MODULAR = False
    FIRA_CHALLENGE_ENABLED = True
    FIRA_SAFETY_SIGNAL_ESTIMATOR = False
    FIRA_PROFILE_SCHEDULER_ENABLED = True
    FIRA_CHECKPOINT_DETECTOR_ENABLED = True
    AI_THROTTLE_MULT = 0.65

    # Race presets for safer high-speed operation and real checkpoint tracking.
    FIRA_CHALLENGE_COLLISION_IMMINENT_ENABLED = True
    FIRA_CHALLENGE_COLLISION_IMMINENT_PROXIMITY = 0.86
    FIRA_CHALLENGE_COLLISION_IMMINENT_OCCUPANCY = 0.40
    FIRA_CHALLENGE_COLLISION_IMMINENT_THROTTLE_CAP = 0.01

    FIRA_CHECKPOINT_COLOR_LOW = (0, 120, 120)
    FIRA_CHECKPOINT_COLOR_HIGH = (10, 255, 255)
    FIRA_CHECKPOINT_COLOR_LOW_2 = (170, 120, 120)
    FIRA_CHECKPOINT_COLOR_HIGH_2 = (179, 255, 255)
    FIRA_CHECKPOINT_ROI_TOP_RATIO = 0.72
    FIRA_CHECKPOINT_MIN_PIXEL_RATIO = 0.07
    FIRA_CHECKPOINT_COOLDOWN_FRAMES = 10

else:
    raise ValueError(
        f"Unsupported FIRA_COMPETITION_MODE='{FIRA_COMPETITION_MODE}'. Use 'urban' or 'race'."
    )
