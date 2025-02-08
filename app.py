import os
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)


MODEL_NAME = "facebook/bart-large-cnn"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Hide API Key (Set this in Git Repository Secrets or .env)
HF_API_KEY = os.getenv("HF_API_KEY", "your-readonly-key")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if "text_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["text_file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read and decode text file
        text = file.read().decode("utf-8")

        # Tokenization and Summarization
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
