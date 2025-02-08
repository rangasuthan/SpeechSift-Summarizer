import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the summarization model
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def summarize_text(text):
    """Summarizes the given text using the Hugging Face model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        text = request.form.get('text')  # Get the text input from the form
        if not text:
            return jsonify({'error': 'No text provided for summarization'}), 400
        
        summary = summarize_text(text)
        return jsonify({'summary': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
