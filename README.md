# 🎙️ Human Voice Clustering & Classification App

An interactive Streamlit web application that analyzes vocal features to classify human voices by gender and explores clustering patterns using machine learning models.

---

## 🚀 Features

- 📊 **Exploratory Data Analysis (EDA)**  
  Visualize feature distributions, detect outliers, and analyze gender-based patterns.

- 🔍 **Clustering with KMeans & DBSCAN**  
  Discover natural groupings in audio features using unsupervised learning.

- 🤖 **Gender Classification (RF, SVM, MLP)**  
  Classify male/female voice using Random Forest, Support Vector Machine, and Multi-layer Perceptron.

- 📈 **Performance Metrics**  
  View classification reports including accuracy, precision, recall, and F1-score.

---

## 🧠 Technologies Used

- **Frontend/UI:** Streamlit
- **Backend/ML:** scikit-learn, pandas, numpy
- **Visualization:** seaborn, matplotlib
- **Model Deployment:** joblib (for loading pre-trained models)

---

## 🗃️ Dataset

The application uses a CSV file (`vocal_gender_features_new.csv`) containing pre-extracted audio features and gender labels:
- **Features:** Various pitch, intensity, spectral, and formant-related statistics.
- **Label:** `0` for Female, `1` for Male.


