"""
Variable -> MIMIC-IV v3.1 itemid mapping, confirmed with PI 2026-07-28.

Decisions locked in:
  - GCS verbal: not-assessable (intubated) values treated as missing, NOT floor-coded to 1.
      Total GCS computed from eye+verbal+motor only when verbal is assessable;
      otherwise flagged incomplete (gcs_total_incomplete=1) and eye+motor reported alongside.
  - Blood pressure (SBP/MAP): arterial (invasive) preferred, fall back to non-invasive cuff.
  - Sedation: binary flag = continuous infusion of propofol/midazolam/dexmedetomidine/lorazepam
      active at any point in the 6h window. (RASS not used for state definition.)
  - CRRT: continuous modalities only (CVVH/CVVHD/CVVHDF/SCUF). Intermittent HD excluded.
"""

# ---------------- chartevents (vitals + neuro) ----------------
GCS_EYE = 220739
GCS_VERBAL = 223900
GCS_MOTOR = 223901
# chartevents.value strings indicating verbal not assessable (intubated).
# Confirmed against actual value distribution for itemid 223900 (2026-07-28):
# "No Response-ETT" is the intubation-artifact marker (784,911 rows); plain
# "No Response" is a genuine clinical finding and is NOT treated as missing.
# "No Response-Trach" does not occur in this database (tracheostomy patients
# below the vocal cords can still phonate, so MIMIC-IV scores them normally).
GCS_VERBAL_NOT_ASSESSABLE_VALUES = ["No Response-ETT"]

HEART_RATE = 220045

SBP_ARTERIAL = 220050
SBP_NONINVASIVE = 220179
MAP_ARTERIAL = 220052
MAP_NONINVASIVE = 220181

RESP_RATE = 220210
SPO2 = 220277

TEMP_F = 223761
TEMP_C = 223762

# ---------------- labevents (blood gas + chemistry + hematology) ----------------
# each entry: list of itemids to UNION (old + new BIDMC lab-system generations)
PH = [50820]
PAO2 = [50821]
PACO2 = [50818]

WBC = [51301, 51755, 51756]
HEMOGLOBIN = [51222, 51640]
PLATELET = [51265, 53189]
CREATININE = [50912, 52546]
BUN = [51006, 52647]
SODIUM = [50983, 52623]
POTASSIUM = [50971, 52610]
GLUCOSE = [50931, 52569]
LACTATE = [50813, 52442, 53154]

# ---------------- procedureevents (interval-based: has starttime/endtime) ----------------
MECH_VENT = [225792]              # Invasive Ventilation
CRRT_MODES = [225802, 225803, 225809, 225955]  # CVVH, CVVHD, CVVHDF, SCUF

# ---------------- inputevents (continuous infusions, has starttime/endtime) ----------------
VASOPRESSORS = [221289, 221653, 221662, 221749, 221906, 221986, 222315,
                229617, 229630, 229631, 229632]  # incl. newer-system duplicate formulations
SEDATIVES = [221385, 221668, 222168, 225150, 229420]

# ---------------- outputevents (urine only, excludes GI outputs) ----------------
URINE_OUTPUT = [226559, 226560, 226561, 226563, 226564, 226565, 226567]
# Foley, Void, Condom Cath, Suprapubic, R Nephrostomy, L Nephrostomy, Straight Cath

# Decision 2026-07-28: lactate/pH/PaO2/PaCO2 exceed the 70%-window-missingness
# exclusion rule (section 8.1 of the design doc) -- excluded from the PRIMARY
# HMM state model, retained only for a sensitivity-analysis refit.
PRIMARY_MODEL_VARS = [
    "heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c",
    "gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total", "gcs_total_incomplete",
    "wbc", "hemoglobin", "platelet", "creatinine", "bun", "sodium", "potassium", "glucose",
    "mech_vent", "crrt", "vasopressor", "sedative", "urine_output_ml",
]
SENSITIVITY_ONLY_VARS = ["ph", "pao2", "paco2", "lactate"]

ALL_CHARTEVENTS_ITEMIDS = (
    [GCS_EYE, GCS_VERBAL, GCS_MOTOR, HEART_RATE, SBP_ARTERIAL, SBP_NONINVASIVE,
     MAP_ARTERIAL, MAP_NONINVASIVE, RESP_RATE, SPO2, TEMP_F, TEMP_C]
)
ALL_LABEVENTS_ITEMIDS = PH + PAO2 + PACO2 + WBC + HEMOGLOBIN + PLATELET + \
    CREATININE + BUN + SODIUM + POTASSIUM + GLUCOSE + LACTATE
ALL_PROCEDUREEVENTS_ITEMIDS = MECH_VENT + CRRT_MODES
ALL_INPUTEVENTS_ITEMIDS = VASOPRESSORS + SEDATIVES
ALL_OUTPUTEVENTS_ITEMIDS = URINE_OUTPUT
