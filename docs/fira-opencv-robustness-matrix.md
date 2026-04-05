# FIRA OpenCV Robustness Matrix

This document summarizes the robustness coverage added for the OpenCV fallback sign detector and the focused FIRA regression command.

## Scope

The coverage targets the blue-sign classes most sensitive to visual perturbations:

- TUNNEL
- DEAD_END

The tests validate detector behavior both in isolation and through the FiraModular fallback path.

## Robustness Matrix

The following conditions are covered in synthetic temporal tests:

- Baseline synthetic sign detection
- Temporal Gaussian noise
- Geometric jitter (small translation and scaling)
- Partial occlusion from image borders
- Inter-class alternation (TUNNEL <-> DEAD_END)
- Class-switch recovery in block sequences
- Extreme lighting via HSV value scaling (low light and overexposure)
- Motion blur sequences (horizontal and vertical)
- Resolution/scale variation (downscale/upscale and restore)
- Color cast variation (white-balance style channel shifts)
- JPEG compression artifacts

## Detector/Fallback Behavioral Checks

Also covered:

- Plain blue sign does not trigger false positives
- FiraModular fallback to OpenCV for STOP and TUNNEL when no AprilTag is detected
- OpenCV path is used when AprilTag backend is unavailable

## Focused Regression Command

Run this command from repository root:

```bash
/bin/python -m pytest \
  donkeycar/tests/test_opencv_sign_detector.py \
  donkeycar/tests/test_fira_health_metrics.py \
  donkeycar/tests/test_fira_safety_signals.py \
  donkeycar/tests/test_fira_template_inputs.py \
  donkeycar/tests/test_fira_safety_arbiter.py \
  donkeycar/tests/test_fira_engine_yolo_health.py \
  donkeycar/tests/test_fira_state_machine.py \
  donkeycar/tests/test_fira_engine_tf_health.py
```

Current expected result after the latest robustness additions:

- 57 passed
- 0 failed

## Notes

- These are deterministic synthetic tests with seeded random generators.
- The matrix is intended as a fast safety gate for fallback robustness before hardware runs.
- For on-track validation, combine this suite with real capture replay and device-specific camera tuning.
