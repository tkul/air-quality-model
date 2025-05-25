# Air Quality Prediction

This project is a full-stack machine learning application that predicts **PM2.5 air pollution levels** using city-based environmental and geographic data. It includes data preprocessing, model training and optimization (with Optuna), and a deployed web application built with Streamlit.

🔗 Live Demo: [https://airquality.tugcekul.com](https://airquality.tugcekul.com)

---

## 📊 About the Project

- **Dataset:** WHO's 2022 global urban air quality dataset (`AAP_2022_city_v9`)
- **Target:** Predict `PM2.5 (μg/m3)` levels based on inputs like PM10, NO2, region, and country
- **Model:** XGBoost Regressor
- **Optimization:** Hyperparameter tuning using Optuna
- **Deployment:** Streamlit app served via NGINX on a custom domain

---


## 🛠️ Installation (Local Development)

### Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```


## 🧪 Example Inputs

* PM10: 50 μg/m³
* NO2: 30 μg/m³
* WHO Region: Western Pacific
* ISO3: TUR
* Country Name: Turkey

---

## 📈 Example Outputs

* PM2.5 prediction: \~33.27 μg/m³
* Feature importance and prediction scatterplots automatically generated

---
