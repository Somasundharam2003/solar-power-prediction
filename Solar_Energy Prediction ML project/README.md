# ☀️ Solar Energy Production Prediction using Machine Learning

This project leverages machine learning techniques to predict solar power generation in Berkeley, California. By analyzing how environmental variables such as temperature, humidity, sky cover, and solar noon proximity affect energy output, this notebook builds a robust predictive model using a Random Forest Regressor and hyperparameter optimization through GridSearchCV.

---

## 🧠 Project Motivation

Solar energy is one of the most promising sources of renewable energy. However, its efficiency and consistency are highly dependent on environmental conditions. Accurately forecasting energy production can:

* Optimize the operation of solar farms.
* Reduce reliance on non-renewable backup systems.
* Enhance decision-making in energy grid management.
* Contribute to the transition toward sustainable energy ecosystems.

---

## 📁 Repository Structure

```
.
├── Solar_Panel_Power_Generation.ipynb  # Full code: EDA, modeling, evaluation
├── Programming CA2 REPORT_20047046_Ertugrul_Asliyuce.pdf  # Detailed academic report
├── requirements.txt                    # Python dependencies
└── README.md                           # Project overview and instructions
```

---

## 📊 Dataset Overview

The dataset is based on a solar energy system located in Berkeley, California, with:

* **2929 rows** and **16 features** (both numerical and categorical)
* Columns include:

  * `Average Temperature (Day)`, `Relative Humidity`, `Visibility`, `Sky Cover`
  * `Distance to Solar Noon`, `Is Daylight`, `Wind Speed`, `Power Generated` (target)
  * Time-based features: `Day`, `Month`, `Year`, `First Hour of Period`

The data shows seasonal, temporal, and environmental variation in energy output.

---

## 🔧 Feature Engineering

The following preprocessing and engineering steps were applied:

* **Missing Value Handling**: Imputed using column mean to preserve structure.
* **Categorical Encoding**: One-hot encoding for `season`, `sky cover`, `time of day`, `solar proximity`.
* **New Features**:

  * `season`: Derived from month
  * `time of day`: Derived from hour
  * `solar proximity`: Based on distance to solar noon
* **Redundant Columns Removed**: Original temporal features dropped post-transformation.
* **Standardization**: Applied to all numerical features.

---

## 📈 Model Building

Model: `RandomForestRegressor` from `scikit-learn`
Why Random Forest?

* Captures non-linear relationships
* Robust against outliers
* Works with mixed data types
* Provides feature importance

### 🔍 Initial Model Results

* **Mean Squared Error (MSE):** `10,776,163.01`
* **R² Score:** `0.8976`

---

## ⚙️ Model Optimization

Hyperparameter tuning was done using `GridSearchCV` with 5-fold cross-validation.

**Best Parameters:**

```python
{
  'n_estimators': 200,
  'max_depth': 20,
  'min_samples_split': 5,
  'min_samples_leaf': 2,
  'max_features': 'sqrt'
}
```

**Improved Model Results:**

* **MSE:** `9,834,850.31`
* **R² Score:** `0.9061`

---

## 📉 Visualizations

The notebook includes:

* Histogram of power generation (shows skewness, likely due to night-time data)
* Correlation matrix (high negative correlation with distance to solar noon)
* Scatter plots between features and output (non-linear trends)
* Line plots to show seasonal production patterns
* Before-and-after model performance plots

---

## 💻 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/solar-energy-prediction.git
cd solar-energy-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook Solar_Panel_Power_Generation.ipynb
```

---

## 🚀 Future Improvements

* Integrate real-time weather data using public APIs
* Experiment with deep learning models (e.g., LSTM for time series)
* Deploy the model as a web service with Flask or FastAPI
* Use SHAP or LIME for model interpretability

---

## 📚 References

* Medium Article on [Random Forest Regression](https://medium.com/@byanalytixlabs/random-forest-regression-how-it-helps-in-predictive-analytics-01c31897c1d4)
* Kaggle Dataset: [Solar Power Generation](https://www.kaggle.com/datasets/vipulgote4/solar-power-generation/data)
* Fieguth, P. (2022). *An Introduction to Pattern Recognition and Machine Learning*.
* Camastra & Vinciarelli (2008). *Machine Learning for Audio, Image and Video Analysis*.

---

## 📜 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.  
See the [LICENSE](LICENSE) file for full legal text.

