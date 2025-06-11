import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import librosa.display
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

    menu = ["Home", "EDA", "Clustering", "Classification", "Audio Prediction", "Conclusion"]
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
- Upload an audio file and get instant **gender prediction**
- View model **performance metrics and reports**
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

    elif choice == "Audio Prediction":
        st.subheader("🎧 Upload an Audio File to Predict")
        uploaded_file = st.file_uploader("Upload a WAV File", type=["wav"])

        if uploaded_file:
            y, sr = librosa.load(uploaded_file, duration=3, mono=True)
            st.audio(uploaded_file)

            st.write("### Waveform")
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)

            features = {
                "meanfreq": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "sd": np.std(y),
                "Q25": np.percentile(y, 25),
                "Q75": np.percentile(y, 75),
                "IQR": np.percentile(y, 75) - np.percentile(y, 25),
                "skew": pd.Series(y).skew(),
                "kurt": pd.Series(y).kurtosis(),
                "sp.ent": -np.sum(librosa.feature.spectral_centroid(y=y, sr=sr) * np.log2(librosa.feature.spectral_centroid(y=y, sr=sr))),
                "sfm": np.mean(librosa.feature.spectral_flatness(y=y)),
                "mode": pd.Series(y).mode()[0] if len(pd.Series(y).mode()) > 0 else 0,
                "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "meanfun": np.mean(librosa.feature.zero_crossing_rate(y)),
                "minfun": np.min(librosa.feature.zero_crossing_rate(y)),
                "maxfun": np.max(librosa.feature.zero_crossing_rate(y)),
                "meandom": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "mindom": np.min(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "maxdom": np.max(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "dfrange": np.ptp(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "modindx": np.std(librosa.feature.zero_crossing_rate(y)) / np.mean(librosa.feature.zero_crossing_rate(y))
            }

            feat_df = pd.DataFrame([features])
            scaler = models.get("Scaler")
            if scaler:
                X_pred_scaled = scaler.transform(feat_df)
                chosen_model = st.selectbox("Select Model", ["RF", "SVM", "MLP"])
                model = models.get(chosen_model)
                if model:
                    pred = model.predict(X_pred_scaled)[0]
                    st.success(f"Predicted Gender: {'Male' if pred == 1 else 'Female'}")
                else:
                    st.error("Selected model not found.")
            else:
                st.error("Scaler model not loaded.")

    elif choice == "Conclusion":
        st.subheader("📌 Conclusion")
        st.markdown("""
### Project Summary
- Implemented **human voice classification** using extracted **audio features**.
- Used **EDA, clustering (KMeans, DBSCAN)**, and **classification (RF, SVM, MLP)** techniques.
- Built a real-time **Streamlit interface** for gender prediction using uploaded audio.

### Key Insights
- KMeans clustering revealed natural groupings when clusters were well defined.
- Random Forest delivered high classification accuracy and robustness.
- Live voice predictions proved feasible with extracted features.

### Future Work
- Add deep learning models with spectrogram input.
- Expand voice datasets across different languages.
- Build a scalable backend API for real-time prediction.
        """)

if __name__ == "__main__":
    main()
