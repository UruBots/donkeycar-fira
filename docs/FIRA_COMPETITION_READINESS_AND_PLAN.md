# FIRA 2025 Competition Readiness: Matrix, Gaps, and Execution Plan

## Scope
This document provides:
1. Rule-by-rule readiness matrix for Urban and Race.
2. Concrete changes prioritized by impact/risk.
3. A competition operating profile (Urban/Race presets).
4. A phased execution plan to reach "ready to win" status.

Current implementation status (April 2026):
- Real checkpoint detector integrated (`FiraCheckpointDetector`).
- Metrics consume real checkpoint events (with proxy fallback).
- Anti-DNF imminent-collision throttle guard in challenge mode.
- Official replay scoring utilities (SAR/SAUD/ST) and readiness battery script.
- Fixed competition launch presets: `myconfig_competition_urban.py` and `myconfig_competition_race.py`.

Primary rules reference:
- docs/FIRA Challenge - Autonomous Cars Rules 2025 (Pro) .md

---

## 1) Compliance Matrix (Urban + Race)

Legend:
- Status = OK | PARTIAL | MISSING | RISK
- Evidence references code/docs currently in repository.

| Requirement (Rules 2025) | Status | Current Evidence | Gap / Risk |
|---|---|---|---|
| No wireless communication for control/vehicle communication during competition (2.8.1) | OK | Competition presets force `WEB_CONTROL_ENABLED=False`, `WEBRTC_ENABLED=False`, `USE_FPV=False`, `USE_NETWORKED_JS=False`. Battery checks these assertions. | Operationally run only competition configs on event day. |
| Onboard processing for Ka=1.0 | PARTIAL | FIRA engines are local parts; no mandatory cloud dependency. | Operational discipline needed to avoid offboard usage during runs. |
| Urban signs include No Entry, Dead End, Right, Left, Forward, Stop, Tunnel, Bridge | PARTIAL | Stop/NoEntry/DeadEnd/Right/Left/Forward/Tunnel supported in OpenCV detector and tests. | Bridge sign path is not implemented/tested as explicit behavior. |
| AprilTag-based sign flow supported | OK | FIRA engine/apriltag integration in donkeycar/parts/fira_engine.py and FIRA modular flow. | Need calibration on event hardware. |
| Stop at designated stop signs | OK | Stop states/actions in fira_engine/fira_modular and state machine integration. | Validate on real track camera perspective. |
| Stop for >=3s before zebra crossings | OK | `wait_duration=3.0` and 3s waits in fira_engine/fira_modular. | Must verify timing under real loop jitter. |
| Stop before stop line (~1cm before zebra) | PARTIAL | Stop-line detection and min/max distance gate in modular flow. | Needs event-day camera geometry calibration for 1cm equivalence. |
| Follow correct lanes / right lane at checkpoints (Urban 6.2.7/6.2.8) | PARTIAL | Event-driven checkpoints now available and checkpoint crossings outside right-lane condition increment violations. | Still vision-dependent; validate with real checkpoint geometry. |
| Race checkpoints: cross all checkpoints, avoid skipping | PARTIAL | `FiraCheckpointDetector` + event-driven `FiraCompetitionMetrics`; official replay scoring script available. | Needs per-stage calibration of checkpoint color/ROI and final acceptance runs. |
| Hitting obstacle ends run (Urban/Race) | PARTIAL | Challenge includes imminent-collision throttle cap guard (anti-DNF). | Still requires physical validation and conservative throttle tuning per track. |
| 3 attempts, 5-minute attempt constraints | PARTIAL | Operationally external to code. | Need team runbook and timed rehearsal protocol. |
| Pre-deployment checklist completion | PARTIAL | Existing docs checklists are still manual and open-ended. | Convert checklist into evidence-driven go/no-go gate. |

---

## 2) Concrete Changes (Prioritized)

### P0 (Mandatory before event)
1. Enforce no-wireless competition mode by default for official runs.
  - Status: DONE in config + battery assertions.
2. Add Bridge sign support end-to-end.
   - Code: extend opencv_sign_detector + action mapping in modular/engine.
   - Tests: add synthetic Bridge detector tests and behavior transition tests.
3. Add stop-line distance behavior (not just zebra presence).
  - Status: PARTIAL (estimator + gating present; field calibration pending).
4. Add checkpoint/right-lane validation helper for race and urban scoring readiness.
  - Status: DONE baseline (event detector + metrics + tests + replay scorer).

### P1 (High value, strong score impact)
1. Track-side profile split with dedicated presets:
   - Urban: FIRA_MODULAR + SAFE + strict zebra/stop compliance.
   - Race: FIRA_CHALLENGE + lane retention + low-risk overtaking envelope.
2. Add competition preflight command that produces a pass/fail report.
   - Include tests, key config assertions, and runtime sanity checks.
3. Add run-time incident counters:
   - lane departures, obstacle near-misses, stop compliance counts.

### P2 (Performance optimization)
1. Latency profiling and deterministic loop tuning per mode.
2. Confidence-based dynamic speed policy per curvature/checkpoint segment.
3. Event-day calibration script (lighting and HSV auto-tune bounded by safe ranges).

---

## 3) Competition Mode Profile

New profile file:
- donkeycar/templates/cfg_competition.py

Behavior:
- `FIRA_COMPETITION_MODE=urban`
  - `FIRA_MODULAR=True`
  - `FIRA_CHALLENGE_ENABLED=False`
  - `FIRA_SAFETY_SIGNAL_ESTIMATOR=True`
  - conservative throttle defaults
- `FIRA_COMPETITION_MODE=race`
  - `FIRA_MODULAR=False`
  - `FIRA_CHALLENGE_ENABLED=True`
  - challenge policy active for dynamic obstacle handling
- Strict no-wireless defaults in both:
  - `WEB_CONTROL_ENABLED=False`
  - `WEBRTC_ENABLED=False`
  - `USE_FPV=False`
  - `USE_NETWORKED_JS=False`

How to run:
```bash
# Urban
export FIRA_COMPETITION_MODE=urban
python manage.py drive --myconfig=donkeycar/templates/cfg_competition.py

# Race
export FIRA_COMPETITION_MODE=race
python manage.py drive --myconfig=donkeycar/templates/cfg_competition.py
```

Fixed presets (recommended to avoid env mistakes on event day):
```bash
# Urban stage
python manage.py drive --myconfig=myconfig_competition_urban.py

# Race stage
python manage.py drive --myconfig=myconfig_competition_race.py
```

Readiness battery:
```bash
python scripts/fira_competition_battery.py
```

---

## 4) Execution Plan (Phased)

### Phase A (Today - 1 day): Compliance Lockdown
- Goal: eliminate rule-violation risk.
- Tasks:
  - Freeze competition config to no-wireless profile.
  - Add explicit event run command aliases for urban/race.
  - Produce one-page referee-facing compliance checklist.
- Exit criteria:
  - No web endpoints used in competition profile.
  - Team rehearses startup->run->stop with config only.

### Phase B (2-4 days): Rule Coverage Gaps
- Goal: close missing technical requirements.
- Tasks:
  - Implement Bridge sign detection + routing action.
  - Implement stop-line distance control before zebra.
  - Add checkpoint/right-lane monitoring logic.
- Exit criteria:
  - New tests for Bridge, stop-line, checkpoint lane compliance all passing.

### Phase C (3-5 days): Track Validation
- Goal: convert features into reliable points.
- Tasks:
  - Timed Urban rehearsals (3 attempts x 5 min protocol).
  - Timed Race rehearsals with checkpoint audit.
  - Collect incident metrics and retune thresholds.
- Exit criteria:
  - Zero disqualifying behaviors across repeated runs.
  - Stable top-score candidate run in both tracks.

### Phase D (1-2 days): Competition Hardened Release
- Goal: freeze a deterministic, low-risk release.
- Tasks:
  - Lock versions/configs and save immutable run profile.
  - Run full FIRA tests + scenario smoke tests + preflight report.
  - Prepare operator playbook for attempt turnaround window.
- Exit criteria:
  - Green preflight + signed checklist + final candidate configs.

---

## Current Objective Truth
- FIRA regression tests currently pass in this environment (57/57).
- System is strong in safety/obstacle/sign baseline.
- Winning readiness depends on closing the explicit scoring/compliance gaps above, especially:
  - no-wireless enforcement in operations,
  - Bridge support,
  - stop-line precision,
  - checkpoint/right-lane scoring behaviors.
