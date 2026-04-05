import importlib
import os
import sys


def _load_cfg(mode: str):
    old = os.environ.get('FIRA_COMPETITION_MODE')
    os.environ['FIRA_COMPETITION_MODE'] = mode
    try:
        sys.modules.pop('donkeycar.templates.cfg_competition', None)
        mod = importlib.import_module('donkeycar.templates.cfg_competition')
        return importlib.reload(mod)
    finally:
        if old is None:
            os.environ.pop('FIRA_COMPETITION_MODE', None)
        else:
            os.environ['FIRA_COMPETITION_MODE'] = old


def test_competition_urban_profile_enforces_compliance_defaults():
    cfg = _load_cfg('urban')

    assert cfg.WEB_CONTROL_ENABLED is False
    assert cfg.WEBRTC_ENABLED is False
    assert cfg.USE_FPV is False
    assert cfg.USE_NETWORKED_JS is False
    assert cfg.FIRA_MODULAR is True
    assert cfg.FIRA_CHALLENGE_ENABLED is False
    assert cfg.FIRA_CHECKPOINT_DETECTOR_ENABLED is False
    assert cfg.FIRA_REQUIRE_ZEBRA is True


def test_competition_race_profile_enables_checkpoint_and_anti_dnf():
    cfg = _load_cfg('race')

    assert cfg.WEB_CONTROL_ENABLED is False
    assert cfg.WEBRTC_ENABLED is False
    assert cfg.FIRA_MODULAR is False
    assert cfg.FIRA_CHALLENGE_ENABLED is True
    assert cfg.FIRA_PROFILE_SCHEDULER_ENABLED is True
    assert cfg.FIRA_CHECKPOINT_DETECTOR_ENABLED is True
    assert cfg.FIRA_CHALLENGE_COLLISION_IMMINENT_ENABLED is True
    assert cfg.FIRA_CHALLENGE_COLLISION_IMMINENT_THROTTLE_CAP <= 0.02
