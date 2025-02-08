from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
from transformers import pipeline

app = Flask(__name__)

# Load Summarization Model (Optimized)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to transcribe audio using SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)  # Google Speech API (Free & No Token Limit)
            return text
        except sr.UnknownValueError:
            return "Speech was unclear. Unable to transcribe."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition."

# Function to split text into chunks (if text is too long)
def split_text(text, max_chunk_length=400):
    words = text.split()
    chunks = []
    chunk = []
    length = 0
    for word in words:
        chunk.append(word)
        length += len(word) + 1
        if length > max_chunk_length:
            chunks.append(" ".join(chunk))
            chunk = []
            length = 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if "audio_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded audio file
    audio_path = "uploaded_audio.wav"
    audio_file.save(audio_path)

    # Transcribe audio
    transcription = transcribe_audio(audio_path)

    # Handle empty transcription case
    if not transcription or "Unable to transcribe" in transcription:
        return jsonify({"error": "Transcription failed", "transcription": transcription})

    # Split long transcriptions into smaller chunks
    text_chunks = split_text(transcription)

    # Summarize each chunk and combine
    summarized_text = " ".join(
        summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        for chunk in text_chunks
    )

    return jsonify({"transcription": transcription, "summary": summarized_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get the port from environment variable, default to 5000
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host="0.0.0.0", port=port, debug=True)

