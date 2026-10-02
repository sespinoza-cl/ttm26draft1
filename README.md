# Neck pain and painful temporomandibular disorders: data and analysis

De-identified dataset and analysis code for a retrospective cross-sectional study of 145 patients from the Temporomandibular Disorders and Orofacial Pain Clinic, Universidad de Valparaíso, Chile (2020–2023; ethics approval CECFAR-UV-10-2024).

## Contents

| File | Content |
|---|---|
| `data/tmd_cervical_master.csv` | De-identified dataset (145 patients × 58 variables): DC/TMD Axis I items, cervical examination, age, sex, joint hypermobility |
| `data/CODEBOOK.md` | Variable definitions, coding and reference counts |
| `analysis.py` | The complete analysis, in one script |
| `requirements.txt` | Pinned Python packages |

The dataset contains no names, national ID numbers, clinical record numbers or dates. Rows are shuffled, and `study_id` values are random. Blank cells mean the item was not recorded.

## Running the analysis

Requires Python 3.10.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python analysis.py                          # about 40 min; writes outputs/
QUICK=1 .venv/Scripts/python analysis.py                  # quick smoke test with few iterations
```

The script checks its Firth regression against R `logistf` reference values, then runs four sections in order:
1. Pre-specified models and the 64-specification multiverse.
2. Additional analyses requested in review.
3. Grouped and subgroup analyses (post hoc).
4. Figures.

Random seeds are fixed, so repeated runs give identical results. It writes tables (CSV), figures (PNG, 300 dpi) and JSON summaries to `outputs/`.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
