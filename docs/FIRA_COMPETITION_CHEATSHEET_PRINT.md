# FIRA Competition Cheat Sheet (Printable)

Use this one-page checklist at the operator station.

| Block | Urban | Race | Check |
|---|---|---|---|
| Config file | `myconfig_competition_urban.py` | `myconfig_competition_race.py` | [ ] |
| Launch command | `python manage.py drive --myconfig=myconfig_competition_urban.py` | `python manage.py drive --myconfig=myconfig_competition_race.py` | [ ] |
| Wireless off | `WEB_CONTROL=False`, `WEBRTC=False`, `FPV=False` | same | [ ] |
| Core stack | `FIRA_MODULAR=True`, `CHALLENGE=False` | `FIRA_MODULAR=False`, `CHALLENGE=True`, scheduler on | [ ] |
| Zebra/stop-line | Required | N/A for race behavior | [ ] |
| Checkpoint detector | Off by default | On by default | [ ] |
| Anti-DNF guard | On | On (tighter cap) | [ ] |

## Pre-Run (2 min)

| Item | Check |
|---|---|
| Battery secure and charged | [ ] |
| Camera clean and stable | [ ] |
| Correct stage config selected | [ ] |
| `python scripts/fira_competition_battery.py` passed | [ ] |

## In-Run Watchlist

| Signal | Expected | Check |
|---|---|---|
| `fira/competition/checkpoint_count` | Monotonic increase in checkpoint sections | [ ] |
| `fira/competition/lane_violations` | Low/controlled growth | [ ] |
| `fira/competition/compliance_score` | Stable upward trend | [ ] |
| `fira/checkpoint_event` | Pulses when crossing visible checkpoints | [ ] |

## Between Attempts (60-90s)

| Step | Check |
|---|---|
| Stop process cleanly | [ ] |
| Note penalties/incidents | [ ] |
| Confirm stage mode again | [ ] |
| Replay scoring (if reports available) | [ ] |

Replay command:

```bash
python scripts/fira_official_scoring_replay.py \
  --reports-glob "data/**/fira_competition_report_*.json" \
  --t-stage-race 120 --t-stage-urban 120 \
  --total-checkpoints-race 12 --total-checkpoints-urban 12
```

## Go / No-Go

| Condition | Decision |
|---|---|
| Stable first 10-15s, checkpoint events valid, no steering spikes | GO |
| Repeated lane violations, no checkpoint events, near-collision guard oscillation | NO-GO |
