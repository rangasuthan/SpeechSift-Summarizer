from flask import Flask, request, jsonify, render_template
import os
import speech_recognition as sr
import requests
from pydub import AudioSegment

app = Flask(__name__)

# Hugging Face API Key (Replace with your actual key)
HF_API_KEY = "your_huggingface_api_key"
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Transcribe each audio chunk
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
        raise

# Split audio into chunks
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
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text})
        response_json = response.json()
        
        if "error" in response_json:
            return f"API Error: {response_json['error']}"

        return response_json[0]["summary_text"]
    except Exception as e:
        return f"Summarization API error: {str(e)}"

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

    # Save the audio file to the 'uploads' folder
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

    # Split the audio into chunks
    try:
        audio_chunks = split_audio(file_path, chunk_length_ms=30000)  # Split into 30-second chunks
    except Exception as e:
        return jsonify({"error": f"Audio splitting failed: {str(e)}"})

    # Transcribe each audio chunk
    combined_transcription = ""
    try:
        for chunk_path in audio_chunks:
            transcription = transcribe_audio(chunk_path)
            combined_transcription += transcription + " "
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"})

    # Summarize the combined transcription using Hugging Face API
    summary = summarize_text(combined_transcription)

    # Return both transcription and summary
    return jsonify({"transcription": combined_transcription, "summary": summary})

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
