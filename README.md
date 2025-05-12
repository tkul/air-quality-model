İşte sadece **kod** kısımları kutu içinde olacak şekilde sadeleştirilmiş ve lisans + gereksinimler bölümü çıkarılmış İngilizce `README.md`:

---

# Air Quality Prediction Project

This project analyzes air quality data and builds a predictive model using XGBoost. It includes data preprocessing, feature engineering, model optimization with Optuna, and result visualization.

## 📁 Project Structure

air\_quality/
├── main.py — Main script for training and evaluation
├── requirements.txt — Python dependencies
├── data/
│   └── data.xlsx — Raw air quality dataset
└── outputs/
  ├── correlation\_matrix.png
  ├── feature\_importance.png
  ├── optimized\_feature\_importance.png
  ├── xgboost\_actual\_vs\_predicted.png
  └── optimized\_xgboost\_actual\_vs\_predicted.png

## ⚙️ Installation

1. Extract the archive or clone the repository.
2. (Optional) Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the main pipeline with:

```
python main.py
```

This will:

* Load and preprocess the dataset
* Train a base and optimized XGBoost model
* Save visual outputs to the `outputs/` directory

## 📊 Output Files

* `correlation_matrix.png`: Feature correlation heatmap
* `feature_importance.png`: Feature importance (baseline)
* `optimized_feature_importance.png`: Feature importance (optimized)
* `xgboost_actual_vs_predicted.png`: Actual vs Predicted (baseline)
* `optimized_xgboost_actual_vs_predicted.png`: Actual vs Predicted (optimized)

---

Hazırsa bu metni senin için `README.md` dosyası olarak klasöre kaydedeyim mi?
