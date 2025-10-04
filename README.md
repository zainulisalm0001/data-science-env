# 🏡 House Prices Prediction – Assignment 1  

This repository contains the work for **Data Science in Production – Assignment 1**.  
The goal is to build a simple machine learning pipeline to predict house prices using the [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  

---

## 📂 Repository Structure  

---

## 🚀 Project Setup  

1. Clone repository
   ```bash
   git clone https://github.com/zainulisalm0001/data-science-env.git
   cd data-science-env

 2. Create branch 
```bash
git checkout -b pw1

3. Install dependencies

pip install -r requirements.txt

4. Add dataset

Place train.csv inside the data/ folder (this file is ignored in git).

📘 Notebook Overview

The notebook house-prices-modeling.ipynb implements:
	•	Data loading (using pandas)
	•	Feature selection (2 continuous + 2 categorical features)
	•	Preprocessing
	•	Continuous: imputation + scaling
	•	Categorical: encoding
	•	Model Training (RandomForestRegressor)
	•	Evaluation using RMSLE metric (competition standard)

⸻

🛠️ Technologies Used
	•	Python 3.11
	•	pandas, numpy
	•	scikit-learn
	•	Jupyter Notebook
