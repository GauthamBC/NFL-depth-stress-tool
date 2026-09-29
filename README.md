# NFL Backup Depth Stress Test

A manual-triggered, data-driven NFL roster depth tool designed for an Action Network-style interactive.

## What the public tool answers

**Which NFL teams have the biggest current drop from a first-choice player to the next available option?**

The tool does **not** predict injuries, wins, losses or playoff qualification. It measures current backup depth.

For each team it evaluates 25 roles:

- 11 offense
- 11 defense
- kicker
- punter
- long snapper

The frontend lets a user open any team, inspect the current first choice and projected next-up player, and simulate up to three players becoming unavailable.

## Production architecture

```text
Manual update click
      ↓
Python updater
      ↓
Latest depth charts + rosters + snap counts + player stats
      ↓
Replacement mapping + player values + depth cliffs
      ↓
Validation CSVs
      ↓
Human review
      ↓
nfl_roster_stress_test.json
      ↓
HTML / JavaScript interactive on GitHub Pages
      ↓
Action Network iframe
```

Google Sheets is not required to power the tool. A sheet or XLSX can still be produced for editorial review, but the live webpage should read the generated JSON.

## Replacement rule

For every current first-choice role, the updater uses this hierarchy:

1. latest available depth-chart slot;
2. remove players currently marked unavailable;
3. select the next available player in that same slot;
4. use recent unit snap share as supporting evidence;
5. if the slot has no clear active successor, fall back to the same position family and prioritize recent usage;
6. flag low-confidence or proxy-based cases for human review.

The JSON stores the selection basis and evidence string so every projected replacement is auditable in the frontend.

## Player value and small samples

Starter and backup values are normalized within position families. Because backups can have very small NFL samples, the updater does not treat a handful of snaps as equally reliable to a full-season sample.

A sample-reliability factor shrinks thin performance samples toward the position-family median before the player is ranked. The frontend can display this reliability in the details panel.

Public player-level quality data are still weaker for offensive linemen and long snappers. Those roles are explicitly labelled as using a stability / experience proxy unless a validated override is supplied.

## Core scoring

Individual role impact:

```text
starter-to-next-up value gap
× recent role share
× position leverage
× current unit depletion
→ league-relative 0-100 impact index
```

The team **Depth Risk Index** is based on its five largest role impacts, weighted:

```text
45% + 25% + 15% + 10% + 5%
```

Higher means a thinner current backup-depth profile. It is not an injury probability.

### Cascade simulation

The updater also stores a second replacement where available.

When the user simulates the starter becoming unavailable:

```text
starter OUT
→ first replacement becomes the new first choice
→ second replacement becomes the new next-up player
→ remaining depth risk recalculates
```

This makes the interactive more meaningful than simply fading a player card.

## Run the updater

```bash
pip install -r requirements.txt
python update_stress_test.py --season 2026 --output-dir data --overrides-dir overrides
```

Outputs:

```text
data/nfl_roster_stress_test.json
data/validation_player_level.csv
data/validation_team_summary.csv
```

Review both validation CSVs before publishing a refreshed JSON.

## Manual update workflow

For the first production version, use a controlled update rather than an unattended schedule:

1. run the Colab notebook or Python script;
2. inspect all teams marked `REVIEW`;
3. check low-confidence replacements, offensive-line reshuffles and proxy-metric positions;
4. add any necessary override;
5. rerun;
6. publish the refreshed JSON after the checks pass.

Once replacement mapping has been proven reliable over several weeks, the same pipeline can be scheduled with a manual publication gate.

## Overrides

### Current availability

`overrides/injury_overrides.csv`

```csv
team,gsis_id,unavailable
BUF,00-0031234,true
```

### Validated player value

`overrides/player_value_overrides.csv`

```csv
gsis_id,player_value
00-0031234,81.5
```

## Frontend features

The current `index.html` includes:

- league-wide animated ranking;
- Overall / Offense / Defense / Special Teams (K/P/LS) filters;
- team search;
- clear explanation of what the score does and does not mean;
- 25-role team stress test;
- first-choice → projected next-up comparison;
- depth-cliff bars;
- replacement-confidence labels;
- snap-share display;
- expandable replacement evidence;
- sample-reliability detail;
- sorting by biggest cliff, position or replacement confidence;
- three-player scenario simulator;
- second-level replacement cascade where available;
- recalculated remaining-depth score and scenario rank;
- glossary for non-expert NFL users;
- methodology and data-source notes;
- responsive mobile layout.

If the live JSON is not present, `index.html` falls back to clearly labelled preview data so the design can still be reviewed safely.

## Hosting

Recommended GitHub Pages structure:

```text
index.html
update_stress_test.py
requirements.txt
overrides/
  injury_overrides.csv
  player_value_overrides.csv
data/
  nfl_roster_stress_test.json
```

The page fetches:

```text
./data/nfl_roster_stress_test.json
```

After review, upload the new JSON and the GitHub Pages version updates without changing the HTML.

## Current data design

The updater is designed around nflreadpy / nflverse data layers for:

- depth charts;
- PFR snap counts;
- weekly rosters;
- player statistics;
- player IDs;
- team metadata.

The current short-term injury feed needs a separate validated source or the included manual availability override. Before publication, confirm licensing and attribution requirements for every data source used in the final production version.
