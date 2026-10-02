# Codebook: `tmd_cervical_master.csv`

The file has 145 rows, one per patient, and 58 columns. It contains de-identified records of patients seen at the TMD & Orofacial Pain Clinic, Universidad de Valparaíso, 2020–2023 (ethics approval CECFAR-UV-10-2024). It was generated from the clinic's source spreadsheet, which is private and not distributed.

## De-identification
- **Removed:** names, national ID (RUT), clinical record numbers, dates of birth, dates of intake, and side/direction details of the examination findings.
- **Age:** released in whole years at the clinical examination.
- **Row order:** shuffled. `study_id` values are random and are not derived from any clinical identifier.

## Coding conventions
- `1` = present, `0` = absent.
- An empty cell means the item was **not recorded or not evaluated**. It does not mean absent.

## Variables

| Column(s) | Description | Values |
|---|---|---|
| `study_id` | Random study identifier | TMD001–TMD145 |
| `age` | Age at clinical examination | years (12–74) |
| `female` | Sex | 1 = female, 0 = male |
| `mov_pain_flexext`, `mov_pain_rot`, `mov_pain_lat` | Pain (VAS ≥ 1) during active cervical flexion/extension, rotation (70°), lateral inclination (60°) | 0/1 |
| `mov_restr_flexext`, `mov_restr_rot`, `mov_restr_lat` | Restricted range in the same movements | 0/1 |
| `palp_trap_r/l`, `palp_subocc_r/l`, `palp_scm_r/l` | Pain on palpation, rated 0–10, at the trapezius, suboccipital and sternocleidomastoid, right and left | 0–10 |
| `palp_flag` | Clinician's overall judgement of cervical palpation pain | 0/1 |
| `np_clinician_summary` | Clinicians' original summary column "neck pain". It differs from the item-level definition in 1 record (statistical analysis plan, data check DC1). | 0/1 |
| `beighton` | Beighton generalized hypermobility score | 0–9 |
| `hypermobility` | Generalized joint hypermobility (clinician) | 0/1 |
| `arthralgia` … `hyperplasia` | DC/TMD Axis I joint diagnoses: arthralgia, arthritis, disc displacements (`ddwr`, `ddwr_locking`, `ddwor_limited`, `ddwor_unlimited`), adhesions, ankylosis, subluxation, luxation, degenerative joint disease, systemic arthritis, other joint diseases, fracture, and developmental disorders | 0/1 |
| `myalgia_local`, `myofascial`, `myofascial_referral` | DC/TMD myalgia subtypes. They are **not mutually exclusive** in the source and should be analysed as "any myalgia" (statistical analysis plan, data check DC3). | 0/1 |
| `tendinitis` … `fibromyalgia` | Other DC/TMD masticatory muscle disorders | 0/1 |
| `headache_tmd` | Headache attributed to TMD | 0/1 |
| `clin_joint_pain`, `clin_joint_disorder`, `clin_muscle_disorder` | The clinicians' own aggregate codings, used as alternative groupings in the multiverse | 0/1 |

## Derived variables

Derived variables are **not stored** in the file. `analysis.py` builds them.

**Neck-pain outcomes**
- `NP_clin` (primary) = any of `mov_pain_*` OR `palp_flag`.
- `NP_move` = any of `mov_pain_*`.
- `NP_strict` = any of `mov_pain_*` AND `palp_flag`.
- `NP_palp` = `palp_flag`.
- `CPS` = sum of the 6 `palp_*` scores.
- `MPC` = number of `mov_pain_*` items equal to 1.

**Diagnosis groups**
- `MUSCLE` = any myalgia subtype.
- `JOINT` = `arthralgia` OR `arthritis`.
- `DISC` = any of the 4 disc-displacement items.

## Reference counts (n = 145)

| Measure | Count |
|---|---|
| NP_clin | 110 |
| NP_move | 60 |
| NP_strict | 58 |
| NP_palp | 108 |
| MUSCLE | 115 |
| JOINT | 61 |
| DISC | 64 |
| Female | 112 |
