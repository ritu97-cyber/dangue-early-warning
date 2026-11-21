# dangue-early-warning
# Dengue Early-Warning System (Dhaka / Chattogram / Gazipur / Narayanganj)

Predicts dengue outbreak risk (Low / Medium / High / Critical) for each ward, 2 to 4 weeks ahead, from weather and ward conditions. It gives public health officers an early-warning window before outbreaks peak.

**Team:** Ritu Das (Member 1), jerin sadea eva (Member 2)

## Quick start (about 3 minutes)

Requires Python 3.12 and Git.

```bash
git clone https://github.com/ritu97-cyber/dangue-early-warning.git
cd dangue-early-warning
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# Mac/Linux:             source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The trained model (`models/model.joblib`) and the dataset (`data/`) are included, so the app starts without retraining. If `py` is your Python command on Windows, use `py -m venv .venv`.

Optional checks:
```bash
python src/features.py   # verifies lag features and the purged split
python src/predict.py    # quick prediction demo and timing
python src/train.py      # retrain (validation only; add --test for the one-time final test)
```

## How it works

1. **Weather lag pipeline** (`src/features.py`): for each ward, computes rainfall lags (1 to 6 weeks), rolling rainfall sums (2, 4, 8 weeks), and temperature and humidity lags. Every feature uses only data from the current week or earlier.
2. **Target:** total dengue cases in weeks t+2 to t+4 (the 2 to 4 week early-warning window).
3. **Model** (`src/train.py`): LightGBM with a Poisson objective for case counts.
4. **Chronological split with purging:** train 2016 to 2022, validation 2023, test 2024 to 2025. The last 4 weeks of each period are purged so no target window crosses a boundary.
5. **Risk tiers:** true tiers come from training-target percentiles (70, 90, 97). The predicted-case cut-offs are tuned on validation only. Test data is used once at the end.
6. **Drivers** (`src/predict.py`): LightGBM per-prediction contributions, grouped into Recent rainfall, Temperature, Humidity, Season, and Ward vulnerability. These show which factors the model weighed most. They are not proof of cause.

## Results (test set 2024 to 2025, used once)

| Metric | Model | Always-guess-the-average |
|---|---|---|
| MAE (cases per 3-week window) | 2.31 | 8.92 |
| RMSE | 4.40 | n/a |

- Exact tier correct: 83%. Within one tier: 98.9%.
- Limitation: Critical precision is only 13.7% (many false alarms). The cut-offs were tuned on 2023, a high-dengue year, and 2024 to 2025 had fewer cases, so the model is over-cautious. We kept the result as is and did not tune on test data.

## Cost and speed

No external API and no LLM at run time. The model is a serialized `joblib` file. One prediction takes about 30 ms. The dashboard loads the model and data once and caches them.

## Project structure

```
app.py               Streamlit dashboard
src/features.py      lag features, target, purged split
src/train.py         training, tier thresholds, evaluation
src/predict.py       prediction and driver breakdown
models/              model.joblib, thresholds.json, metrics files
data/                hackathon_preli_dataset.xlsx
ARCHITECTURE.md      data flow, split and model rationale
```

## AI disclosure

- **AI assistant:** Claude (Anthropic) via claude.ai chat. Used for project planning, task division, code generation and review of `src/features.py`, `src/train.py`, `src/predict.py`, the dashboard, and the documentation, and for debugging help. Both team members reviewed and ran the code and can explain it.
- **External APIs at run time:** none.
- **Libraries (open source):** pandas, numpy, openpyxl, scikit-learn, LightGBM, joblib, Streamlit, Plotly.
- **Data:** the official hackathon dataset provided with the challenge.
