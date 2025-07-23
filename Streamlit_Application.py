import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, classification_report, accuracy_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("vocal_gender_features_new.csv")

# Load models
@st.cache_resource
def load_models():
    try:
        return {
            "RF": joblib.load("random_forest_model.pkl"),
            "SVM": joblib.load("svm_model.pkl"),
            "MLP": joblib.load("mlp_model.pkl"),
            "Scaler": joblib.load("scaler.pkl")
        }
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return {}

def main():
    st.set_page_config("Human Voice Clustering & Classification", layout="wide")
    st.title("🎙️ Human Voice Clustering & Classification")

    menu = ["Home", "EDA", "Clustering", "Classification", "Voice Prediction", "Conclusion"]
    choice = st.sidebar.radio("Go to", menu)

    data = load_data()
    models = load_models()

    if choice == "Home":
        st.subheader("🏠 Welcome to the Human Voice Analysis App")
        st.markdown("""
### 🔍 What You Can Do Here
- Explore the dataset and audio features using **EDA**
- Use **KMeans** or **DBSCAN** to explore natural clusters in the data
- Classify voices using **Random Forest, SVM, or MLP**
- View model **performance metrics and reports**
- Predict gender based on a few manually entered features
- Read about the project goals and insights in the **Conclusion**

🎧 This app uses audio feature extraction, machine learning models, and Streamlit UI to give you a full pipeline from data exploration to audio classification.
        """)

    elif choice == "EDA":
        st.subheader("📊 Exploratory Data Analysis")
        st.dataframe(data.head())

        st.write("### Summary Statistics")
        st.write(data.describe())

        st.write("### Gender Distribution")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='label', data=data)
        plt.title("Gender Distribution (0 = Female, 1 = Male)")
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### Correlation Heatmap")
        plt.figure(figsize=(12, 6))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### Outlier Detection with Boxplot")
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=data.drop(columns=["label"]))
        plt.xticks(rotation=90)
        plt.title("Boxplot for Outlier Detection Across Features")
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### KDE Plots for Pitch-Related Features")
        pitch_features = [col for col in data.columns if 'pitch' in col.lower()]
        if pitch_features:
            melted = data[pitch_features + ['label']].melt(id_vars='label')
            g = sns.FacetGrid(melted, col="variable", hue="label", sharex=False, sharey=False, height=4)
            g.map(sns.kdeplot, "value").add_legend()
            st.pyplot(g.fig)
            plt.clf()
        else:
            st.warning("No pitch-related features found in the dataset.")

    elif choice == "Clustering":
        st.subheader("🔍 Clustering (KMeans & DBSCAN)")
        features = data.drop(columns=["label"])
        scaler = models.get("Scaler")
        if scaler:
            X_scaled = scaler.transform(features)

            algo = st.selectbox("Choose Algorithm", ["KMeans", "DBSCAN"])

            if algo == "KMeans":
                k = st.slider("Choose K", 2, 10, 3)
                km = KMeans(n_clusters=k, random_state=42)
                labels = km.fit_predict(X_scaled)
            else:
                eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
                min_samp = st.slider("Min Samples", 2, 20, 5)
                db = DBSCAN(eps=eps, min_samples=min_samp)
                labels = db.fit_predict(X_scaled)

            st.write("### Silhouette Score")
            valid_mask = labels != -1
            if len(set(labels[valid_mask])) > 1:
                score = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
                st.success(f"Silhouette Score: {score:.2f}")
            else:
                st.warning("Not enough clusters for silhouette score.")

            st.write("### PCA Plot")
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            scatter = ax.scatter(components[:, 0], components[:, 1], c=labels, cmap='tab10')
            ax.set_title("PCA of Clusters")
            st.pyplot(fig)
        else:
            st.error("Scaler model not loaded.")

    elif choice == "Classification":
        st.subheader("🤖 Voice Classification (RF, SVM, MLP)")
        scaler = models.get("Scaler")
        if scaler:
            X = data.drop(columns=["label"])
            y = data["label"]
            X_scaled = scaler.transform(X)

            clf = st.selectbox("Choose Classifier", ["RF", "SVM", "MLP"])
            model = models.get(clf)

            if model:
                st.write("### Prediction Results")
                preds = model.predict(X_scaled)
                pred_df = pd.DataFrame({"Actual": y, "Predicted": preds})
                st.dataframe(pred_df.head())

                st.write("### Classification Report")
                report = classification_report(y, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.error("Classifier model not found.")
        else:
            st.error("Scaler not found.")

    elif choice == "Voice Prediction":
        st.subheader("🎤 Predict Gender from Voice Features")
        user_input = {}
        user_input["mean_spectral_centroid"] = st.number_input("Mean Spectral Centroid", value=2000.0)
        user_input["std_spectral_centroid"] = st.number_input("Std Spectral Centroid", value=100.0)
        user_input["mean_pitch"] = st.number_input("Mean Pitch", value=120.0)
        user_input["std_pitch"] = st.number_input("Std Pitch", value=30.0)
        user_input["zero_crossing_rate"] = st.number_input("Zero Crossing Rate", value=0.05)
        user_input["rms_energy"] = st.number_input("RMS Energy", value=0.1)
        user_input["mfcc_1_mean"] = st.number_input("MFCC 1 Mean", value=15.0)
        user_input["mfcc_1_std"] = st.number_input("MFCC 1 Std", value=5.0)

        if st.button("Predict Gender"):
            try:
                full_input = {}
                for col in data.columns:
                    if col != "label":
                        full_input[col] = user_input[col] if col in user_input else float(data[col].mean())
                input_df = pd.DataFrame([full_input])
                input_scaled = models["Scaler"].transform(input_df)
                prediction = models["RF"].predict(input_scaled)[0]
                gender = "Male" if prediction == 1 else "Female"
                st.success(f"🔮 Predicted Gender: **{gender}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    elif choice == "Conclusion":
        st.subheader("📌 Conclusion")
        st.markdown("""
### 📋 Project Summary
- Developed a **human voice classification** system using extracted **audio features**.
- Applied techniques including **Exploratory Data Analysis (EDA)**, **Clustering (KMeans, DBSCAN)**, and **Classification (Random Forest, SVM, MLP)**.
- Designed an interactive **Streamlit application** for real-time gender prediction using audio input.

### 💡 Key Insights
- **KMeans** clustering identified natural groupings when clusters were distinct.
- **Random Forest** showed superior classification accuracy and robustness.
- Real-time prediction was successfully achieved using pre-extracted features from audio files.

### 🚀 Future Enhancements
- Integrate **deep learning models** using spectrograms or raw audio waveforms.
- Expand the dataset to include **multilingual and cross-cultural** voice samples.
- Develop a **scalable backend API** to support deployment in production environments.
        """)

if __name__ == "__main__":
    main()
