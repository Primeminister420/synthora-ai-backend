import openai
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
import whisper

# Initialize Firebase
cred = credentials.Certificate("path/to/firebase/credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask App
app = Flask(__name__)

# Load OpenAI Whisper model
model = whisper.load_model("base")

# OpenAI API Key (Replace with actual key)
OPENAI_API_KEY = "your_openai_api_key"

def summarize_text(text):
    """Uses GPT-4 to summarize text notes."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the following text into key bullet points."},
            {"role": "user", "content": text}
        ],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"]

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Handles speech-to-text conversion using Whisper."""
    file = request.files["audio"]
    result = model.transcribe(file)
    return jsonify({"text": result["text"]})

@app.route("/summarize", methods=["POST"])
def summarize():
    """Summarizes a text note using GPT-4."""
    data = request.json
    text = data.get("text", "")
    summary = summarize_text(text)
    return jsonify({"summary": summary})

@app.route("/save_note", methods=["POST"])
def save_note():
    """Saves a note to Firebase Firestore."""
    data = request.json
    doc_ref = db.collection("notes").add(data)
    return jsonify({"message": "Note saved", "id": doc_ref[1].id})

@app.route("/search_notes", methods=["POST"])
def search_notes():
    """Searches notes using keywords or semantic similarity."""
    data = request.json
    query = data.get("query", "")
    notes_ref = db.collection("notes").stream()
    results = [note.to_dict() for note in notes_ref if query.lower() in note.to_dict().get("text", "").lower()]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
