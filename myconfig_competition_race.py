import os

# Freeze mode before loading template config.
os.environ['FIRA_COMPETITION_MODE'] = 'race'

from donkeycar.templates.cfg_competition import *

FIRA_COMPETITION_MODE = 'race'

# Competition hardening: keep onboard-only control channels.
WEB_CONTROL_ENABLED = False
WEBRTC_ENABLED = False
WEBRTC_ICE_SERVERS = []
USE_FPV = False
USE_NETWORKED_JS = False

# Race stage sanity constraints.
FIRA_MODULAR = False
FIRA_CHALLENGE_ENABLED = True
FIRA_SAFETY_SIGNAL_ESTIMATOR = False
FIRA_PROFILE_SCHEDULER_ENABLED = True
FIRA_CHECKPOINT_DETECTOR_ENABLED = True

# Conservative anti-DNF fallback.
FIRA_CHALLENGE_COLLISION_IMMINENT_ENABLED = True
FIRA_CHALLENGE_COLLISION_IMMINENT_PROXIMITY = 0.86
FIRA_CHALLENGE_COLLISION_IMMINENT_OCCUPANCY = 0.40
FIRA_CHALLENGE_COLLISION_IMMINENT_THROTTLE_CAP = 0.01
