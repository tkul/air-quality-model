````markdown
# Air Quality Prediction Project

This project analyzes air quality data and builds a predictive model using XGBoost. It includes data preprocessing, feature engineering, model optimization with Optuna, and result visualization.

## 📁 Project Structure

air_quality_with_tables/
├── main.py                         # Main script for training and evaluation  
├── requirements.txt               # Python dependencies  
├── data/  
│   └── data.xlsx                  # Raw air quality dataset  
└── outputs/  
    ├── correlation_matrix.png  
    ├── feature_importance.png  
    ├── optimized_feature_importance.png  
    ├── xgboost_actual_vs_predicted.png  
    └── optimized_xgboost_actual_vs_predicted.png  

## ⚙️ Installation

1. Extract the archive or clone the repository.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
````

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

To run the main analysis and modeling pipeline:

```bash
python main.py
```

The script will:

* Load and preprocess the dataset
* Train an XGBoost model
* Perform hyperparameter optimization using Optuna
* Generate result visualizations in the `outputs/` directory

## 📊 Output Files

* `correlation_matrix.png`: Feature correlation heatmap
* `feature_importance.png`: Importance of features (baseline model)
* `optimized_feature_importance.png`: Importance of features (optimized model)
* `xgboost_actual_vs_predicted.png`: Prediction results (baseline)
* `optimized_xgboost_actual_vs_predicted.png`: Prediction results (optimized)

```
