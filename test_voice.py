from voice_module.voice_analysis import analyze_voice

text, emotion, score = analyze_voice()

print("Text:", text)
print("Emotion:", emotion)
print("Score:", score)