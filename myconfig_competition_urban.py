import os

# Freeze mode before loading template config.
os.environ['FIRA_COMPETITION_MODE'] = 'urban'

from donkeycar.templates.cfg_competition import *

FIRA_COMPETITION_MODE = 'urban'

# Competition hardening: keep onboard-only control channels.
WEB_CONTROL_ENABLED = False
WEBRTC_ENABLED = False
WEBRTC_ICE_SERVERS = []
USE_FPV = False
USE_NETWORKED_JS = False

# Urban stage sanity constraints.
FIRA_MODULAR = True
FIRA_CHALLENGE_ENABLED = False
FIRA_SAFETY_SIGNAL_ESTIMATOR = True
FIRA_CHECKPOINT_DETECTOR_ENABLED = False
FIRA_REQUIRE_ZEBRA = True
FIRA_REQUIRE_STOP_LINE = True
