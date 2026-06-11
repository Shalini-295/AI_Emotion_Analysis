from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def analyze_text(text):

    result = classifier(text)[0]

    emotions = {
        r["label"]: r["score"]
        for r in result
    }

    top_emotion = max(
        emotions,
        key=emotions.get
    )

    top_score = emotions[top_emotion]

    return top_emotion, top_score, emotions