from flask import Flask, request, jsonify, render_template
import os
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
import math

app = Flask(__name__)

# Initialize the summarization pipeline with the specified model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

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
    
    # Divide audio into chunks of chunk_length_ms duration
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

    # Summarize the combined transcription
    try:
        summary = summarizer(combined_transcription, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"})

    # Return both transcription and summary
    return jsonify({"transcription": combined_transcription, "summary": summary})

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run()

