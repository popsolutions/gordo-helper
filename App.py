from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from timeout_middleware import TimeoutMiddleware
import os
import subprocess
import whisper

app = Flask(__name__)
app.secret_key = 'eos5dmk2'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.wsgi_app = TimeoutMiddleware(app.wsgi_app, timeout=300)

def convert_mp4_to_mp3(input_file, output_file):
    try:
        print("Starting conversion from MP4 to MP3...")
        command = ['ffmpeg', '-i', input_file, '-q:a', '0', '-map', 'a', output_file]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr}")
        raise

def transcribe_audio(mp3_file, language):
    print("Starting transcription...")
    model = whisper.load_model("base")
    result = model.transcribe(mp3_file, language=language)
    transcription = result['text']
    print("Transcription completed.")
    
    transcription_file = mp3_file.replace('.mp3', '.txt')
    with open(transcription_file, 'w') as f:
        f.write(transcription)
    print(f"Transcription saved to {transcription_file}")
    
    return transcription_file

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        language = request.form['language']
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            try:
                output_file = file_path.rsplit('.', 1)[0] + '.mp3'
                convert_mp4_to_mp3(file_path, output_file)
                transcription_file = transcribe_audio(output_file, language)
                flash(f'Transcription saved to https://gordo-helper.pop.coop/{transcription_file}', 'success')
                return redirect(url_for('download_file', filename=os.path.basename(transcription_file)))
            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'danger')
                return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
