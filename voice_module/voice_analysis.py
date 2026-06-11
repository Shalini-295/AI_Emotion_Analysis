import speech_recognition as sr
from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def analyze_voice():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Speak now...")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(audio)

        result = emotion_model(text)[0]

        emotion = result["label"]
        score = result["score"]

        return text, emotion, score

    except Exception as e:

        return None, None, None