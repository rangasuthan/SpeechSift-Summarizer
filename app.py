from flask import Flask, request, jsonify, render_template
import os
import requests
import speech_recognition as sr
from pydub import AudioSegment

app = Flask(__name__)

# Get Hugging Face API Key from environment variables (GitHub Secret)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face API URL for summarization
HF_SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

# Function to transcribe audio
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)

    try:
        with audio_file as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            return transcription
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None

# Split audio into 30-second chunks
def split_audio(file_path, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(file_path)
    total_length_ms = len(audio)
    chunks = []

    for i in range(0, total_length_ms, chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_filename = f"{file_path}_chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)

    return chunks

# Function to summarize text using Hugging Face API
def summarize_text(text):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": text, "parameters": {"max_length": 130, "min_length": 30, "do_sample": False}}

    response = requests.post(HF_SUMMARIZATION_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        print(f"Summarization API error: {response.text}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part"})

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"})

    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    # Convert MP3 to WAV if necessary
    if file_path.endswith('.mp3'):
        try:
            sound = AudioSegment.from_mp3(file_path)
            file_path = file_path.replace('.mp3', '.wav')
            sound.export(file_path, format="wav")
        except Exception as e:
            return jsonify({"error": f"Failed to convert audio file: {str(e)}"})

    # Split and transcribe audio
    try:
        audio_chunks = split_audio(file_path)
        combined_transcription = " ".join(filter(None, [transcribe_audio(chunk) for chunk in audio_chunks]))
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"})

    if not combined_transcription:
        return jsonify({"error": "No transcription available"})

    # Summarize text using Hugging Face API
    summary = summarize_text(combined_transcription)
    if not summary:
        return jsonify({"error": "Summarization failed"})

    return jsonify({"transcription": combined_transcription, "summary": summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=port, debug=True)
