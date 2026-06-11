import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import re
import os

from streamlit_webrtc import webrtc_streamer
from face_module.face_detection import EmotionDetector

from transformers import pipeline
from text_module.text_analysis import analyze_text
from voice_module.voice_analysis import analyze_voice

# ================= PAGE =================
st.set_page_config(page_title="Emotion AI", layout="wide")

st.markdown("""
<style>

.main {
    background-color: #F4F7FC;
}

.big-title {
    font-size: 40px;
    margin-bottom: 20px;

}


.subtitle{
    text-align:center;
    color:#64748B;
    margin-bottom:25px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    margin-top:15px;
}

.metric-card {
    background: #DBEAFE;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #93C5FD;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 style="
text-align:center;
color:#1E3A8A;
font-weight:700;
font-size:48px;
margin-bottom:5px;
">
🧠 Multimodal Emotion Detection System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
text-align:center;
color:#64748B;
font-size:18px;
">
Text • Face • Voice Emotion Recognition
</p>
""", unsafe_allow_html=True)
# ================= USER LOGIN =================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("## 👤 User Information")

    name = st.text_input("Enter Name")

    user_id = st.text_input("Enter User ID")

    if st.button("🚀 Start Session"):

        if name.strip() and user_id.strip():

            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.session_state.user_id = user_id

            st.rerun()

        else:

            st.error("Please enter Name and User ID")

    st.stop()

# ================= MENU =================


st.sidebar.markdown("## 🧠 Emotion AI")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navigation",
    [
        "🧠 Model Training",
        "📂 Bulk Analysis",
        "📝 Text Analysis",
        "📷 Face Detection",
        "🎤 Voice Analysis",
        "🔗 Combined Result",
        "📜 History",


    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):

    st.session_state.logged_in = False

    st.session_state.pop("user_name", None)
    st.session_state.pop("user_id", None)

    st.rerun()

st.sidebar.info("""
### About Project

This AI system combines:

- Text Emotion Analysis
- Sentiment Detection
- Facial Emotion Recognition
- Emotion Fusion

Built using:
- Streamlit
- Transformers
- FER
- Scikit-learn
""")

# ================= EMOJI =================
def get_emoji(emotion):

    emojis = {
        "joy": "😊",
        "happy": "😊",
        "sadness": "😢",
        "sad": "😢",
        "anger": "😠",
        "angry": "😠",
        "fear": "😨",
        "surprise": "😲",
        "love": "❤️",
        "neutral": "😐"
    }

    return emojis.get(str(emotion).lower(), "😐")

# ================= SENTIMENT =================
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment()

def predict_sentiment(text):

    result = sentiment_pipeline(text)[0]

    return result["label"]

# ================= GRAPH =================
def plot_emotions(emotions):

    df = pd.DataFrame(
        list(emotions.items()),
        columns=["Emotion", "Score"]
    )

    fig, ax = plt.subplots(figsize=(7,3))

    sns.barplot(
        x=df["Emotion"],
        y=df["Score"],
        ax=ax
    )

    plt.xticks(rotation=20)

    plt.tight_layout()

    st.pyplot(fig)
# ================= TEXT ANALYSIS =========================
if menu =="📝 Text Analysis":


    st.header("📝 Text Emotion Analysis")

    text = st.text_area("Enter Text")

    if st.button("Analyze"):

        if text.strip():

            sentiment = predict_sentiment(text)

            emotion, score, emotions = analyze_text(text)

            st.session_state["text_emotion"] = emotion
            st.session_state["text_sentiment"] = sentiment

            st.markdown(f"""
            <div class="card">

            <h3>📊 Analysis Result</h3>

            <p><b>Emotion:</b> {emotion.upper()} {get_emoji(emotion)}</p>

            <p><b>Sentiment:</b> {sentiment}</p>

            <p><b>Confidence:</b> {round(score*100,2)}%</p>

            </div>
            """, unsafe_allow_html=True)

            st.progress(int(score * 100))

            st.subheader("Emotion Scores")

            plot_emotions(emotions)

        else:

            st.warning("Please enter text")
# ================= BULK ANALYSIS =========================
elif menu == "📂 Bulk Analysis":

    st.header("📂 Bulk Text Analysis")

    file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if file:

        df = pd.read_csv(file)

        if "Sentence" not in df.columns:

            st.error("CSV must contain 'Sentence' column")

        else:

            df["Sentiment"] = df["Sentence"].apply(
                predict_sentiment
            )

            st.dataframe(df.head(20))

            st.subheader("Sentiment Distribution")

            fig, ax = plt.subplots(figsize=(5,3))

            sns.countplot(
                x=df["Sentiment"],
                ax=ax
            )

            st.pyplot(fig)

# ================= FACE DETECTION ========================
elif menu =="📷 Face Detection":

    st.header("📷 Face Emotion Detection")
    st.write("Turn on webcam and wait 2-3 seconds")

    # Clear old files button
    if st.button("🗑 Clear Results"):

        if os.path.exists("face_module/face_output.json"):
            os.remove("face_module/face_output.json")

        if os.path.exists("face_module/latest_face.jpg"):
            os.remove("face_module/latest_face.jpg")

        st.rerun()

    # Clear old results when page opens first time
    if "face_reset" not in st.session_state:

        if os.path.exists("face_module/face_output.json"):
            os.remove("face_module/face_output.json")

        if os.path.exists("face_module/latest_face.jpg"):
            os.remove("face_module/latest_face.jpg")

        st.session_state.face_reset = True

    col1, col2 = st.columns([2,1])

    # ================= LEFT SIDE =================
    with col1:

        webrtc_streamer(
            key="face_camera",
            video_transformer_factory=EmotionDetector
        )

        snapshot_path = "face_module/latest_face.jpg"

        if os.path.exists(snapshot_path):

            st.markdown("### 📸 Latest Detection")

            st.image(
                snapshot_path,
                use_container_width=True
            )

    # ================= RIGHT SIDE =================
    with col2:

        face_file = "face_module/face_output.json"

        if os.path.exists(face_file) and os.path.exists(snapshot_path):

            with open(face_file) as f:
                data = json.load(f)

            emotion = data["emotion"]
            score = data["score"]

            st.success(
                f"Face Emotion: {emotion.upper()} {get_emoji(emotion)}"
            )

            st.write(
                f"Confidence: {round(score * 100, 2)}%"
            )

            st.progress(score)

            if "all_emotions" in data:

                st.subheader("All Emotion Scores")

                emotions = data["all_emotions"]

                for emo, val in sorted(
                    emotions.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):

                    st.write(
                        f"{emo.upper()} : {round(val*100,1)}%"
                    )

                    st.progress(float(val))

        else:

            st.info("No face detected yet.")
# ================= VOICE ANALYSIS ========================
elif menu == "🎤 Voice Analysis":

    st.header("🎤 Voice Emotion Analysis")

    st.write("Click button and speak into microphone")

    if st.button("🎙 Start Voice Analysis"):

        with st.spinner("Listening..."):

            text, emotion, score = analyze_voice()

        if text:
            st.session_state["voice_emotion"] = emotion
            st.success("Voice Captured Successfully")
            st.write("### Recognized Speech")
            st.info(text)
            st.write(
                f"### Emotion: {emotion.upper()} {get_emoji(emotion)}"
            )
            st.progress(float(score))
        else:
            st.error("Could not recognize voice")
# ================= COMBINED RESULT =======================

elif menu =="🔗 Combined Result":
    st.header("🔗 Multimodal Emotion Fusion")
    text_emotion = st.session_state.get("text_emotion")
    text_sentiment = st.session_state.get("text_sentiment")
    voice_emotion = st.session_state.get("voice_emotion")
    face_file = "face_module/face_output.json"
 # ================= LOAD FACE RESULT =================
    if os.path.exists(face_file):

        with open(face_file) as f:

            data = json.load(f)

        face_emotion = data.get("emotion")

        face_score = data.get("score", 0)

    else:

        face_emotion = None
        face_score = 0

    # ================= TWO COLUMN LAYOUT =================
    col1, col2, col3 = st.columns(3)

    # ================= TEXT RESULT =================
    with col1:

        st.markdown(f"""
        <div class="card">

        <h3>📝 Text Emotion</h3>

        <h2>{text_emotion if text_emotion else "No Data"} 
        {get_emoji(text_emotion) if text_emotion else ""}</h2>

        <p><b>Sentiment:</b> 
        {text_sentiment if text_sentiment else "N/A"}</p>

        </div>
        """, unsafe_allow_html=True)

    # ================= FACE RESULT =================
    with col2:

        st.markdown(f"""
        <div class="card">

        <h3>📷 Face Emotion</h3>
        <h2>{face_emotion if face_emotion else "No Data"} 
        {get_emoji(face_emotion) if face_emotion else ""}</h2>
        <p><b>Confidence:</b> 
        {round(face_score*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:

        st.markdown(f"""
        <div class="card">

        <h3>🎤 Voice Emotion</h3>

        <h2>
        {voice_emotion if voice_emotion else "No Data"}
        {get_emoji(voice_emotion) if voice_emotion else ""}
        </h2>

        </div>
        """, unsafe_allow_html=True)


        with col1:
            st.metric("📝 Text", text_emotion if text_emotion else "N/A")

        with col2:
            st.metric("📷 Face", face_emotion if face_emotion else "N/A")

        with col3:
            st.metric("🎤 Voice", voice_emotion if voice_emotion else "N/A")
    # ================= FINAL FUSION =================
    st.markdown("---")
    emotions = []
    if text_emotion:
        emotions.append(text_emotion)
    if face_emotion:
        emotions.append(face_emotion)
    if voice_emotion:
        emotions.append(voice_emotion)
    if len(emotions) > 0:
        from collections import Counter
        counts = Counter(emotions)
        final_emotion = counts.most_common(1)[0][0]
        fusion_status = "3-Modal Fusion"
        st.markdown(f"""
        <div class="metric-card">

        <h2>🎯 Final Emotion</h2>

        <h1>
        {get_emoji(final_emotion)}
        </h1>

        <h2>
        {final_emotion.upper()}
        </h2>

        <p>
        Fusion Status: {fusion_status}
        </p>

        </div>
        """, unsafe_allow_html=True)
        if "history_saved" not in st.session_state:
            st.session_state["history_saved"] = False
        if not st.session_state["history_saved"]:
            if st.button("💾 Save To History"):
                from datetime import datetime
                import shutil
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_name = f"history_faces/{timestamp}.jpg"
                os.makedirs("history_faces", exist_ok=True)
                if os.path.exists("face_module/latest_face.jpg"):
                    snapshot_name = ""

                    if os.path.exists("face_module/latest_face.jpg"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        snapshot_name = f"history_faces/{timestamp}.jpg"

                        os.makedirs("history_faces", exist_ok=True)

                        shutil.copy(
                            "face_module/latest_face.jpg",
                            snapshot_name
                        )
                else:
                    snapshot_name = ""
                record = {
                    "name": st.session_state.user_name,
                    "user_id": st.session_state.user_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text_emotion": text_emotion,
                    "face_emotion": face_emotion,
                    "voice_emotion": voice_emotion,
                    "final_emotion": final_emotion,
                    "snapshot": snapshot_name
                }
                history_file = "emotion_history.json"
                if os.path.exists(history_file):
                    with open(history_file, "r") as f:
                        history = json.load(f)
                else:
                    history = []
                history.append(record)
                with open(history_file, "w") as f:
                    json.dump(history, f, indent=4)
                st.session_state["history_saved"] =False
                st.success("History Saved Successfully")
    else:
        st.warning(
            "Please run Text, Face or Voice Analysis first."
        )
elif menu == "📜 History":
    st.header("📜 Emotion History")
    history_file = "emotion_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
        if len(history) > 0:
            df = pd.DataFrame(history)
            st.dataframe(
                df,
                use_container_width=True
            )
            st.markdown("### 📸 Saved Snapshots")
            for item in history:
                st.markdown("---")
                st.write(f"Name: {item['name']}")
                st.write(f"User ID: {item['user_id']}")
                st.write(f"Emotion: {item['final_emotion']}")
                if item.get("snapshot") and os.path.exists(item["snapshot"]):
                    st.image(
                        item["snapshot"],
                        width=250
                    )
            st.session_state["history_saved"] = False
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇ Download History",
                csv,
                "emotion_history.csv",
                "text/csv"            )
        else:
            st.info("No history available.")
    else:
        st.info("No history available.")
# ================= MODEL TRAINING ========================
elif menu =="🧠 Model Training":

    st.header("🧠 Emotion Model Training")
    train_path = "data/train.txt"
    if not os.path.exists(train_path):
        st.error("Dataset not found!")
    else:
        st.success("Emotion dataset loaded successfully")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.linear_model import LogisticRegression
        # ================= LOAD DATA =================
        df = pd.read_csv(
            train_path,
            sep=';',
            names=["Sentence", "Emotion"]
        )
        # ================= CREATE SENTIMENT =================
        positive_emotions = [
            "joy",
            "love",
            "surprise"
        ]
        df["Sentiment"] = df["Emotion"].apply(
            lambda x:
            "POSITIVE"
            if x in positive_emotions
            else "NEGATIVE"
        )
        # ================= PREPROCESS =================
        def preprocess(text):
            text = str(text)
            text = re.sub(r"http\S+", " ", text)
            text = re.sub(r"@\w+", " ", text)
            text = re.sub(r"#\w+", " ", text)
            text = re.sub(r"[^a-zA-Z ]", " ", text)
            return text.lower()
        df["Clean"] = df["Sentence"].apply(preprocess)
        # ================= VECTORIZATION =================
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        X = vectorizer.fit_transform(df["Clean"])
        # ================= LABEL =================
        y = df["Sentiment"].map({
            "NEGATIVE": 0,
            "POSITIVE": 1
        })
        # ================= SPLIT =================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42        )
        # ================= MODEL =================
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # ================= SAVE =================
        os.makedirs("models", exist_ok=True)
        pickle.dump(
            model,
            open("models/emotion_model.pkl", "wb")
        )
        pickle.dump(
            vectorizer,
            open("models/vectorizer.pkl", "wb")
        )
        # ================= ACCURACY =================
        acc = accuracy_score(y_test, y_pred)
        st.success(
            f"Accuracy: {round(acc*100,2)}%"
        )
        # ================= CONFUSION MATRIX =================
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(2.8, 2.2))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["NEGATIVE", "POSITIVE"],
            yticklabels=["NEGATIVE", "POSITIVE"],
            cbar=False,
            annot_kws={"size": 12},
            ax=ax
        )

        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

        plt.tight_layout()
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
             st.pyplot(fig, use_container_width=False)
        # ================= EMOTION GRAPH =================
        st.subheader("📊 Emotion Distribution")

        emotion_counts = df["Emotion"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(3.5,2))

        sns.barplot(
            x=emotion_counts.index,
            y=emotion_counts.values,
            ax=ax2
        )

        ax2.set_xlabel("")
        ax2.set_ylabel("Count")

        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

        plt.xticks(rotation=25)

        # Center the chart
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.pyplot(fig2)

        plt.close(fig2)
        # ================= SAMPLE DATA =================
        st.subheader("📄 Dataset Sample")

        with st.expander("Click to View Sample Data"):

            sample_df = df[["Sentence", "Emotion", "Sentiment"]].head(5).copy()

            sample_df["Sentence"] = (
                    sample_df["Sentence"].str[:50] + "..."
            )

            st.dataframe(
                sample_df,
                use_container_width=True,
                height=220
            )

        st.success("✅ Model trained successfully")

        st.markdown("---")

        st.markdown(
            """
            <center>
            Developed by <b>Shalini Choudhary</b><br>
            AI • Machine Learning • Computer Vision
            </center>
            """,
            unsafe_allow_html=True
        )