# 🌦️ Predictive Analytics for Weather-Induced Infrastructure Failures

This project leverages machine learning and deep learning techniques to predict **infrastructure failures** caused by **extreme weather events**. By integrating historical weather, environmental, and infrastructure data, the model forecasts potential outages, their **probability, duration,** and **impact severity**, enabling proactive maintenance and disaster readiness.

---

## 🧩 Project Overview

Climate-driven infrastructure disruptions such as power outages, road damages, and equipment failures are increasingly common.  
This project aims to build a **data-driven prediction pipeline** that analyzes weather patterns and correlates them with past failure incidents to estimate the risk of future failures.

The workflow includes:
- Data ingestion from meteorological and infrastructure sources  
- Feature engineering and normalization  
- Model training using multi-modal inputs (temporal, numerical, categorical)  
- Prediction visualization for actionable insights  

---

## 🏗️ Architecture


### Key Components
- **Data Pipeline:** Cleans and merges historical weather and infrastructure datasets  
- **ML/DL Models:** Predicts failure likelihood and duration using Random Forests and Neural Networks  
- **Visualization:** Generates plots for feature importance, accuracy metrics, and predicted outage risk  

---

## 📊 Dataset

- **Weather Data:** Temperature, humidity, precipitation, wind speed, and extreme event logs  
- **Infrastructure Data:** Equipment age, material type, failure logs, and maintenance history  
- **Temporal Resolution:** Hourly/daily records aligned across multiple regions  

All datasets were preprocessed for missing values, scaling, and feature encoding before modeling.

---

## 🧠 Model Summary

- **Techniques Used:**  
  - Random Forest Classifier  
  - LSTM-based Neural Networks for time-series patterns  
- **Metrics Evaluated:** Accuracy, F1-score, ROC-AUC, and Mean Absolute Error (for duration prediction)

The best-performing model demonstrated robust predictive capability across unseen test sets, highlighting strong correlation between weather anomalies and outage occurrences.

---

## 📈 Results & Visualization

Key findings include:
- High correlation between **precipitation & wind speed** and **failure probability**  
- Temporal deep learning models outperformed baseline classifiers by a significant margin  
- Visualization dashboards provided interpretable insights into failure-prone regions  

---

## 🧰 Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Seaborn  
- **Version Control:** Git, GitHub  
- **Environment:** Jupyter Notebook / VS Code  

---

## 🚀 Future Enhancements

- Integrate real-time weather API feeds for live predictions  
- Deploy model via a Flask/Django web interface  
- Expand dataset to include geospatial infrastructure layers  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Lasya Ramachandruni**  
📧 swathisrilasyamayukha.ramachandruni@sjsu.edu  
🌐 [GitHub Profile](https://github.com/LasyaRamachandruni)

---

> “Using data to make our infrastructure resilient against nature’s unpredictability.”
