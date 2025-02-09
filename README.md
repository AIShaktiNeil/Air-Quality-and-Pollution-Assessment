# Air Quality and Pollution Assessment

This repository contains a project report for assessing air quality and pollution using a multi-class classification approach. The project leverages data preprocessing, exploratory analysis, and machine learning techniques to build a robust predictive model. The dataset is sourced from [Kaggle: Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment).

---

## Project Overview

- **Objective:**  
  Predict air quality levels using various environmental and industrial features.  
- **Approach:**  
  - Perform exploratory data analysis (EDA) including histogram plots.  
  - Preprocess data by label encoding non-numeric (multiclass) features.  
  - Build and compare classification models using RandomForestClassifier and GradientBoostingClassifier.
  - Validate models with KFold cross-validation and hyperparameter tuning.

---

## Data Overview

- **Dataset:**  
  The dataset contains features related to air quality measurements and pollution indicators, such as CO levels, industrial proximity, and other environmental variables.
  
- **Key Features:**  
  - **CO:** Carbon Monoxide concentration.
  - **Industrial Proximity:** Distance to industrial areas.
  - Additional features capturing various pollutant concentrations and environmental factors.

---

## Exploratory Data Analysis (EDA)

- **Histograms:**  
  Plotted histograms for key numeric features to understand their distribution.
  
- **Label Encoding:**  
  Applied label encoding on non-numeric, multiclass classification columns to convert categorical data into numerical form for model training.

---

## Modeling & Evaluation

- **Models Used:**  
  - **RandomForestClassifier**
  - **GradientBoostingClassifier**
  
- **Validation Strategy:**  
  - Implemented **KFold cross-validation** to ensure robust model performance across different subsets of data.
  - Calculated cross-validation scores to compare model performance.
  
- **Hyperparameter Tuning:**  
  - Tuned hyperparameters for the GradientBoostingClassifier to optimize model accuracy.
  
- **Performance:**  
  - **Training Score:** 96%
  - **Testing Score:** 93%
  
- **Feature Importance:**  
  Analysis revealed that **CO** levels and **Industrial Proximity** are the most influential features for predicting air quality.

---

## How to Run the Project

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/AIShaktiNeil/Air-Quality-and-Pollution-Assessment/blob/main/Pollution_RF_Project.ipynb
   cd air-quality-pollution-assessment
   ```

2. **Install Dependencies:**

   Ensure you have Python installed, then install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook/Script:**

   Open the Jupyter Notebook (or run the Python script) to execute the analysis:

   ```bash
   http://localhost:8888/lab/tree/Downloads/Pollution_RF_Project.ipynb
   ```

---

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## Conclusion

This project demonstrates how to apply EDA, data preprocessing, and advanced machine learning techniques to assess air quality. With a well-tuned GradientBoostingClassifier achieving 96% training and 93% testing accuracy, and the identification of CO and Industrial Proximity as key predictors, the model provides actionable insights into pollution assessment. Future work may include further feature engineering and exploring additional ensemble methods.

---

## Acknowledgements

- Thanks to the Kaggle community for providing the dataset and inspiring this analysis.
- Special thanks to [Mujtaba Matin](https://www.kaggle.com/mujtabamatin) for the dataset.
