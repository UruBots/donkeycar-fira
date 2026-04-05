# FIRA Competition Day Cheat Sheet

Fast one-page checklist for event day operation.

Printable compact version:
- [FIRA_COMPETITION_CHEATSHEET_PRINT.md](FIRA_COMPETITION_CHEATSHEET_PRINT.md)

## 1) Pre-Run (2 min)

- Battery charged and secure.
- Camera clean, lens fixed, no motion blur.
- Correct profile file selected:
  - Urban: `myconfig_competition_urban.py`
  - Race: `myconfig_competition_race.py`
- No wireless control channels enabled (enforced by competition config).

## 2) Quick Technical Gate

Run this before attempts:

```bash
python scripts/fira_competition_battery.py
```

Expected:
- Config compliance checks pass for urban and race.
- Focused tests pass.
- Scoring replay smoke runs (can show no report files yet).

## 3) Start Commands

Urban stage:

```bash
python manage.py drive --myconfig=myconfig_competition_urban.py
```

Race stage:

```bash
python manage.py drive --myconfig=myconfig_competition_race.py
```

## 4) During Run

Watch these signals in telemetry/web overlay:
- `fira/competition/checkpoint_count`
- `fira/competition/checkpoint_progress`
- `fira/competition/lane_violations`
- `fira/competition/compliance_score`
- `fira/checkpoint_event` (if detector enabled)

If lane violations spike repeatedly:
- Reduce speed multiplier for that stage.
- Re-check camera angle and lane visibility.

If checkpoint_count does not increase:
- Verify marker color and detector ROI thresholds.
- Check `fira/checkpoint_confidence` trend.

## 5) Between Attempts (Fast Reset)

- Stop process cleanly.
- Save run notes: mode, track conditions, penalties seen.
- If reports exist, replay official scoring:

```bash
python scripts/fira_official_scoring_replay.py \
  --reports-glob "data/**/fira_competition_report_*.json" \
  --t-stage-race 120 --t-stage-urban 120 \
  --total-checkpoints-race 12 --total-checkpoints-urban 12
```

## 6) Go/No-Go Rules

Go:
- Clean startup.
- Stable lane behavior first 10-15 seconds.
- No immediate safety throttling oscillation.

No-Go:
- Repeated lane violations immediately after start.
- Checkpoint events never trigger in visible checkpoint sections.
- Large unstable steering swings or repeated near-collision guard activation.

## 7) Minimum Operator Notes

Log per attempt:
- Stage: urban or race
- Config file used
- Estimated penalties observed
- Checkpoint progress achieved
- Root cause if aborted

Keep this sheet open during competition.
