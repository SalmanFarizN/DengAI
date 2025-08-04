# 🧿 DengAI: Predicting Disease Spread

This repository contains experiments and results from our participation in the [DengAI competition on DrivenData](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/).
The goal is to predict the number of dengue fever cases in San Juan and Iquitos using environmental and climate data.
Our repo is a 3-day exploration of this data, building a pipeline to process, analyze, and build a model for predictions that outputs CSV we could submit to the competition.

1227th of 16,349 participants (MAE of 25.0409), we still have a long way to go, but we learned a lot about feature engineering and model selections and their limitations and also how to organize a codebase for the given use case. We have plans to improve and refactor in the near future; you will find a TODO list at the bottom of this README with more info, always open to comments or suggestions too.

---

## 📁 Repository Structure

```
├── data/                     # Data files (cleaned, raw, predictions, etc.)
│   ├── archived_predictions/ # Previous predictions after adjustments to features and models
│   ├── cleaned/
│   ├── images/
│   ├── predictions/         # Main Prediction output directory
│   └── raw/
├── notebooks/               # Jupyter notebooks for data exploration and model prototyping
├── src/                     # Source code modules
│   ├── feature_augmentation.py
│   ├── feature_selector.py
│   ├── load_data.py
│   ├── main.py              # Main training and prediction pipeline
│   ├── output_processing.py
│   ├── preprocess.py
│   ├── stats_model_wrapper.py # Used for NegativeBinomial Model in our pipeline
│   └── test_main.py
├── requirements.txt         # Python dependencies
├── pyproject.toml
├── tasks.py                 # Task automation (e.g., using Invoke)
└── README.md
```

---

## Python

`Python 3.13`

## 🔀 Clone the repo

```bash
git clone https://github.com/SalmanFarizN/DengAI.git
cd dengai-submission
```

## 🥷️ Build virtual environment and install requirements

```bash
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

uv pip install -r requirements.txt
```

## 🏋️‍♂️ Train Model & Predict

```bash
python src/main.py  # Run the pipeline in its current state.
```

---

## 🧪 Models Explored

### 🔹 Baseline Models

- `RandomForestRegressor`
- `NegativeBinomial`

### 🔸 Advanced Models

| Model            | MAE (Private LB) | Notes                      |
| ---------------- | ---------------- | -------------------------- |
| **XGBRegressor** | **25.0409**      | Best performance           |
| SARIMAX          | \~27.71          | Close second               |
| Others           | >27              | Performed worse or overfit |

---

## ⚙️ Feature Engineering & Augmentation

Be aware: Feature augmentation and selection are set mainly in: `feature_augmentation.py` & `feature_selector.py`

### ✅ Feature Selection - Subset

Selected the most correlated features from our data exploration:

```python
[
  "reanalysis_specific_humidity_g_per_kg",
  "reanalysis_dew_point_temp_k",
  "reanalysis_min_air_temp_k",
  "station_min_temp_c",
  "reanalysis_relative_humidity_percent",
  "station_avg_temp_c",
  "reanalysis_precip_amt_kg_per_m2",
  "reanalysis_air_temp_k",
  "reanalysis_sat_precip_amt_mm",
  "reanalysis_avg_temp_k",
  "station_max_temp_c",
  "station_precip_mm",
  "ndvi_sw",
  "weekofyear_col"
]
```

**Additional features:**

We attempted some composite feature augmentation below, but found it had little impact on our MAE.

- **Saturation Deficit**:

```python
saturation_deficit = reanalysis_air_temp_k - reanalysis_dew_point_temp_k
```

- **Temperature Suitability Index**:

```python
temp_suitability = 1 - abs(station_avg_temp_c - 27.5) / 27.5
```

### 🔁 Lag Features

Lags had a strong effect. We tried different combinations to see how many were needed and looked at correlations in the notebook research we did at the beginning of our data exploration.

- Created lagged features from t-1 to t-5
- Best performance from LAGs 1, 2, 3, and 4

### 📅 Seasonality Handling

Added cyclical time components:

```python
week_sin = sin(2 * pi * weekofyear / 52)
week_cos = cos(2 * pi * weekofyear / 52)
```

---

## ✅ TODO

- 🧪 **Expand Pytest Coverage**

  - Develop a more extensive pytest test suite to validate all key components (feature engineering, model predictions, output formatting).

- 🧠 **Improve Pipeline Context Handling**

  - Add shared context throughout the pipeline to make it easier to experiment with and track feature selection/augmentation logic.

- ⚙️ **Introduce Config Files**

  - Implement YAML or JSON-based configuration files for:

    - Model parameters
    - Feature selection choices

  - This will allow us to log and reproduce model runs with clear historical context.

- 📝 **Add Pipeline Logging**

  - Set up run-level logging to capture:

    - Timestamp
    - Feature set used
    - Model used and hyperparameters
    - Evaluation metrics

  - Useful for auditing and comparing multiple experiments over time.

  - Train/Validation Step:
    - We were currently working from MAEs given on competion submission, this gave us little ability to finetune hyper parameters
      easily so I think to future improve this aspect and know better if we are over or underfitting this would also be useful to add.
