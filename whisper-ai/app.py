from flask import Flask, request, jsonify
import whisper
import os
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# ✅ Make sure ffmpeg is available in PATH for Whisper
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
else:
    raise RuntimeError("❌ FFmpeg not found. Please install it and add to PATH.")

# ✅ Load Whisper model (you can change "base" to "small", "medium", or "large")
model = whisper.load_model("base")

# ✅ Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Whisper API! The server is running.'})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 🧠 Whisper transcribes the audio file
        result = model.transcribe(filepath)
        return jsonify({'text': result['text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 🧹 Always clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)