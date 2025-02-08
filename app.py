from flask import Flask, request, jsonify, render_template
import os
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment

app = Flask(__name__)

# Load Summarization Model Locally (No API Key Required)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Transcribe Audio Using SpeechRecognition
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)  # Google Speech-to-Text
            return transcription
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Speech Recognition Error: {e}"

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded"})

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

    # Split the audio into chunks
    try:
        audio_chunks = split_audio(file_path, chunk_length_ms=30000)  # 30-second chunks
    except Exception as e:
        return jsonify({"error": f"Audio splitting failed: {str(e)}"})

    # Transcribe each chunk
    combined_transcription = ""
    try:
        for chunk_path in audio_chunks:
            transcription = transcribe_audio(chunk_path)
            combined_transcription += transcription + " "
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"})

    # Ensure transcription isn't empty before summarization
    if not combined_transcription.strip():
        return jsonify({"error": "Transcription is empty, please check the audio quality."})

    # Summarize Transcription
    try:
        summary = summarizer(combined_transcription, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"})

    return jsonify({"transcription": combined_transcription, "summary": summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get the port from environment variable, default to 5000
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=port, debug=True)

