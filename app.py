from flask import Flask, request, jsonify, render_template
import os
import requests
from pydub import AudioSegment
import speech_recognition as sr

app = Flask(__name__)

# Hugging Face Inference API Endpoint & Token
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_API_TOKEN = "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"  # Replace with your Hugging Face API Token
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Convert speech to text
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded!"})

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file!"})

    # Save the uploaded file
    file_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(file_path)

    # Convert MP3 to WAV if necessary
    if file_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(file_path)
        file_path = file_path.replace('.mp3', '.wav')
        sound.export(file_path, format="wav")

    # Transcribe the audio
    try:
        transcription = transcribe_audio(file_path)
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"})

    # Send transcription to Hugging Face API for summarization
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": transcription})
        summary = response.json()[0]["summary_text"]
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"})

    return jsonify({"summary": summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=port, debug=True)
