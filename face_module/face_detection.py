import cv2
import json
import os

from fer import FER
from streamlit_webrtc import VideoTransformerBase
import streamlit as st


@st.cache_resource
def load_detector():
    return FER()


detector = load_detector()


class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        try:
            result = detector.detect_emotions(img)

            if result:
                emotions = result[0]["emotions"]

                emotion = max(emotions, key=emotions.get)

                score = emotions[emotion]

                x, y, w, h = result[0]["box"]

                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"{emotion.upper()} ({int(score * 100)}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                os.makedirs("face_module", exist_ok=True)

                cv2.imwrite(
                    "face_module/latest_face.jpg",
                    img
                )

                with open("face_module/face_output.json", "w") as f:
                    json.dump(
                        {
                            "emotion": emotion,
                            "score": float(score),
                            "all_emotions": emotions
                        },
                        f
                    )

        except Exception as e:
            print("Face Detection Error:", e)

        return img