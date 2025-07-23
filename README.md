# 🎙️ Human Voice Clustering and Classification

This project builds a machine learning-based system to **cluster** and **classify** human voices based on audio-extracted features. It includes full **EDA**, **clustering (KMeans & DBSCAN)**, **classification (Random Forest, SVM, MLP)**, and a **Streamlit interface** for real-time gender prediction.

---

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA) with visualizations
- 🔍 Clustering using KMeans and DBSCAN
- 🤖 Voice classification using Random Forest, SVM, and MLP
- 🎤 Manual input for real-time gender prediction
- 🌐 User-friendly Streamlit web interface

---

## 📁 Dataset

The dataset includes extracted features from human voice recordings, such as:

- **Spectral Features**: Centroid, Bandwidth, Contrast, Flatness, Rolloff
- **Pitch Features**: Mean, Min, Max, Std
- **MFCCs**: MFCC_1_mean to MFCC_13_mean and their stds
- **Other Features**: Zero Crossing Rate, RMS Energy, Entropy
- **Label**: Gender (`0 = Female`, `1 = Male`)

---

## 🧠 Models Used

### Clustering
- **KMeans**
- **DBSCAN**

### Classification
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Multi-layer Perceptron (MLP)**

---

## 🛠 Tech Stack

- **Python 3.x**
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Streamlit**
- **Joblib**

---

## 📊 Evaluation Metrics

| Metric                 | Value               |
|------------------------|--------------------|
| Accuracy               | ~95%               |
| Precision              | High               |
| Recall                 | High               |
| F1 Score               | Excellent          |
| Silhouette Score       | ~0.45 (for KMeans) |

---

## 🧪 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/voice-gender-classification.git
cd voice-gender-classification
