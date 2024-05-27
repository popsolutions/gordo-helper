from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import subprocess
import whisper
import requests
import concurrent.futures
import datetime
import socket

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Timeout settings
REQUEST_TIMEOUT = 120

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def generate_unique_filename(extension):
    local_ip = get_local_ip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{local_ip}_{timestamp}.{extension}"

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
    
    return transcription_file, transcription

def process_with_ollama(transcription):
    ollama_url = 'http://localhost:1234/v1/chat/completions'
    payload = {
        "model": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "messages": [
            {"role": "system", "content": "Always answer in rhymes."},
            {"role": "user", "content": transcription}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(ollama_url, json=payload, headers=headers)
        response.raise_for_status()
        processed_text = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"Processed text: {processed_text}")
        return processed_text
    except requests.RequestException as e:
        print(f"Error processing with Ollama: {e}")
        raise

def execute_with_timeout(func, *args, timeout=REQUEST_TIMEOUT, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        language = request.form['language']
        file = request.files['file']
        if file:
            extension = file.filename.rsplit('.', 1)[1]
            unique_filename = generate_unique_filename(extension)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            try:
                mp3_filename = generate_unique_filename('mp3')
                mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
                execute_with_timeout(convert_mp4_to_mp3, file_path, mp3_path)
                transcription_file, transcription = execute_with_timeout(transcribe_audio, mp3_path, language)
                processed_text = execute_with_timeout(process_with_ollama, transcription)
                flash('Transcription and processing completed.', 'success')
                return render_template('result.html', transcription=transcription, processed_text=processed_text)
            except concurrent.futures.TimeoutError:
                flash('An error occurred: Request timed out.', 'danger')
                return redirect(url_for('index'))
            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'danger')
                return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
