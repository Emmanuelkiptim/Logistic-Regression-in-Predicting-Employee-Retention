
# Employee Retention Prediction Model

##  Project Overview
This project uses **Machine Learning (Logistic Regression)** to predict whether an employee is likely to **leave the company** or **stay** based on HR analytics data.

The dataset contains employee-related information such as satisfaction level, monthly working hours, promotion history, salary level, and department.

The goal is to help HR teams identify employees at risk of leaving and take proactive retention measures.

---

## Dataset Source
Employee Retention Dataset from Kaggle / Codebasics:

- Kaggle: https://www.kaggle.com/giripujar/hr-analytics
- CSV Used:  
https://raw.githubusercontent.com/codebasics/py/refs/heads/master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Sklearn
- Joblib

---

## Project Steps

### 1️ Import Libraries
The required Python libraries are imported for:

- Data analysis
- Visualization
- Model building
- Model saving

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
import joblib
````

---

### 2️ Load Dataset

```python
df = pd.read_csv('HR_comma_sep.csv')
```

---

### 3️ Exploratory Data Analysis (EDA)

The dataset was analyzed to determine which factors strongly affect employee retention.

#### Key Findings:

Employees who leave tend to have:

* Lower satisfaction levels
* Higher monthly working hours
* Fewer promotions
* Lower salaries

---

### 4️ Data Visualization

#### Salary vs Retention

A bar chart was created to show the relationship between salary and employee retention.

#### Department vs Retention

A bar chart was created to compare employee turnover across departments.

---

### 5️ Feature Selection

The following features were selected for the model:

* satisfaction_level
* average_montly_hours
* promotion_last_5years
* salary

---

### 6️ Data Preprocessing

Since salary is a categorical feature, dummy variables were created using one-hot encoding.

---

### 7️ Train/Test Split

The data was split into:

* 90% Training Data
* 10% Testing Data

---

### 8  Build Logistic Regression Model

A Logistic Regression model was trained using the selected features.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

### 9️ Model Evaluation

The model accuracy was measured on test data.

#### Example Output:

```python
Model Accuracy: 78%
```

(Accuracy may vary depending on train/test split)

---

### 10 Save Model

The trained model was saved using Joblib.

```python
joblib.dump(model, 'Retention_Prediction_Model.pkl')
```

---

## Conclusion

The project successfully predicts employee retention using machine learning.

### Main factors influencing employee turnover:

* Low satisfaction
* Long working hours
* Low salary
* Lack of promotion

This model can help organizations improve employee retention strategies.

---

## Future Improvements

Possible improvements include:

* Random Forest Classifier
* XGBoost
* Flask/Django deployment
* Interactive HR dashboard
* Real-time predictions

---

## Author

Developed by Emmanuel Kiptim

GitHub: [https://github.com/kiptimemmanuel](https://github.com/kiptimemmanuel)
