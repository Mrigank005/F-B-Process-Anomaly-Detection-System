# F&B Process Anomaly Detection System

This project implements an **Anomaly Detection System** tailored for the **Food & Beverage (F&B) industry**, using process data collected in `dataset.xlsx`.  
The notebook demonstrates how data-driven approaches can be applied to identify unusual operational behaviors, which may indicate inefficiencies, risks, or process deviations.

---

## 📂 Repository Structure
- **`F&B_Process_Anomaly_Detection.ipynb`**  
  Main Jupyter Notebook containing:
  - Data loading and preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature scaling and transformation
  - Model training & anomaly detection
  - Visualization of anomalies

- **`dataset.xlsx`**  
  Sample dataset used for training, testing, and anomaly detection.

---

## 🚀 Methods & Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Normalization/standardization of features
   - Converting categorical values (if any)

2. **Exploratory Data Analysis**
   - Correlation heatmaps
   - Distribution plots for process parameters
   - Time-series trends of key features

3. **Anomaly Detection Algorithms**
   - **Isolation Forest**  
     Detects anomalies by randomly partitioning data points. Points requiring fewer splits are marked anomalous.
   - **Local Outlier Factor (LOF)**  
     Measures the local deviation of a data point with respect to its neighbors.
   - **One-Class SVM**  
     Learns the frontier of normal data to separate anomalies.
   - **Z-Score Method** (Statistical baseline)  
     Identifies anomalies based on standard deviation thresholds.

4. **Evaluation**
   - Comparison of anomaly detection methods
   - Visualization of anomalies vs. normal data in 2D/feature space
   - Detection performance analysis

---

## 📊 Example Output

* Correlation heatmaps of process parameters
* Anomaly detection results using different algorithms
* Scatter plots highlighting anomalies vs. normal process behavior

---

## 📝 Notes

* The provided dataset is a **sample** for demonstration purposes.
* Models can be tuned or replaced with advanced methods (e.g., Autoencoders, LSTM) depending on the application.
* This workflow is extendable to **other manufacturing and process industries**.

---
