# 🚗 Used Car Price Prediction ML System

This repository contains an end-to-end Machine Learning project designed to predict the market value of used cars based on various technical and historical features. This project was completed as part of the **Machine Learning Lab Final Course Project**.

## 👥 Group Members & Responsibilities

| Name | ID |
| :--- | :--- |
| **Efrata Wolde** | UGR/1245/15 | 
| **Fromsis Jafar** | UGR/0854/15 | 
| **Hasset Dejene** | UGR/7979/15 | 
| **Natnael Endale** | UGR/5583/15 | 
| **Yonas Tessema** | ATR/0419/14 | 

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                # Original used_cars.csv
│   └── processed/          # Encoded and cleaned data
├── models/
│   ├── model.pkl           # Trained Linear Regression/Random Forest model
│   ├── scaler.pkl          # Saved StandardScaler instance
│   └── features.pkl        # List of features for app consistency
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── app.py                  # Streamlit Web Application
├── requirements.txt        # Python dependencies

└── README.md
