# Simulation and temporal validation of ICU trajectories after stroke

Analysis code for:

> Lei P, Xu Y, Zhang Y, Ray N, Gao Y. Simulation and temporal validation of ICU trajectories
> after stroke: patient condition, transfer readiness and actual ICU exit. (Submitted.)

The study divides each hour of 6,366 ICU stays for stroke in MIMIC-IV v3.1 into time with
organ-support needs, time with stroke-specific needs and time not explained by these
pre-specified needs, using an hourly operational definition of transfer readiness. It then
builds an hourly simulator with separate modules for condition state, ICU death, readiness
gained or lost, destination (stay, move to the other unit type, leave the ICU) and post-exit
outcomes, freezes it on stays up to 2016, and validates it on 2017–2019 and on 2020 onwards
against direct prediction models (logistic regression, gradient boosting, GRU).

---

## This repository contains no patient data

MIMIC-IV and MIMIC-IV-ED are distributed by PhysioNet under credentialed access and the
PhysioNet Credentialed Health Data Use Agreement, which forbids redistributing the data or
patient-level material derived from it. To run this code you need your own access:
https://physionet.org/content/mimiciv/3.1/

Only code and the locked statistical analysis plans are published. The repository was built
from an explicit allow-list, and every file was audited automatically (file type, local paths,
anything resembling a patient identifier, e-mail addresses, credentials) before release.

**Fitted model files are not posted.** The frozen simulator (coefficients, LightGBM model
files, calibrators) contains no patient records but is derived from credentialed data. It
will be shared through PhysioNet under credentialed access; until then the corresponding
author (Ping.Pinglei@unige.ch) provides it to investigators with MIMIC-IV access.

---

## Contents

| Folder | What it does |
|---|---|
| `docs/analysis_plans.md` | English translation of the two locked statistical analysis plans and their amendments (Supplementary S3 of the paper) |
| `09_icu_simulator/00`–`06` | Data checks, condition states over the whole stay (frozen model of the earlier study), hourly readiness (definition versions 0.2 and 0.3) |
| `09_icu_simulator/paper4_formal/` | Hourly decomposition of ICU time (T1/T2/T3), reproducibility across eras, readiness timing and loss (plan A) |
| `09_icu_simulator/simulator/` | The simulator (plan B): training tables and audit (20–22), engine tests (30–31), module selection and system checks (40–41), direct models (42), final fit, freezing and temporal validation (51), parameter uncertainty (52), 2020 onwards (53), ablations (54), sensitivity analyses (55), tables, figures and interpretability (60–63), items run after co-author review (70) |
| `03_code/` | Shared modules from the earlier study (variable and item mappings; pomegranate patch) |

Shared libraries: `simlib.py` (features, L1 modules, simulation engine), `selection_lib.py`
(candidate algorithms L1/L2/L3 and the selection rule), `final_lib.py` (final fit),
`baseline_lib.py` (direct models), `interp_lib.py` and `prespec_lib.py` (parallel workers).

## Setup

1. Python 3.9 and `pip install -r requirements.txt`.
2. Edit `data_paths.cfg` to point to your MIMIC-IV 3.1 and MIMIC-IV-ED 2.2 folders.
3. The condition states use the frozen four-state hidden Markov model and frozen
   preprocessing of the earlier studies (cohort and model code:
   https://github.com/ping-ai685/stroke-states-mimic). Refit them with that code, or obtain the
   frozen files from the corresponding author.
4. Run the scripts in numerical order within each folder. Scripts write patient-level
   intermediates to `work/` folders and aggregate results to `outputs/` folders; neither may
   be shared.

## Licence

MIT (see `LICENSE`). Please cite the paper when using the code.
