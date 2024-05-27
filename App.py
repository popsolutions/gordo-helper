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
app.config['TRANSCRIBED_FOLDER'] = 'transcribed'
app.config['SUMMARIZED_FOLDER'] = 'summarized'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIBED_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUMMARIZED_FOLDER'], exist_ok=True)

# Timeout settings
REQUEST_TIMEOUT = 1200

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def generate_unique_filename(extension, custom_name=None):
    if custom_name:
        return f"{custom_name}.{extension}"
    local_ip = get_local_ip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")  # Include microseconds for more uniqueness
    return f"{local_ip}_{timestamp}.{extension}"

def convert_mp4_to_mp3(input_file, output_file):
    try:
        print("Starting conversion from MP4 to MP3...")
        command = ['ffmpeg', '-y', '-i', input_file, '-q:a', '0', '-map', 'a', output_file]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr}")
        raise

def transcribe_audio(mp3_file, language, custom_name=None):
    print("Starting transcription...")
    model = whisper.load_model("base")
    result = model.transcribe(mp3_file, language=language)
    transcription = result['text']
    print("Transcription completed.")
    
    transcription_file = os.path.join(app.config['TRANSCRIBED_FOLDER'], generate_unique_filename('txt', custom_name))
    with open(transcription_file, 'w') as f:
        f.write(transcription)
    print(f"Transcription saved to {transcription_file}")
    
    return transcription_file, transcription

def process_with_ollama(transcription, language, content, custom_name=None):
    ollama_url = 'http://localhost:1234/v1/chat/completions'
    payload = {
        "model": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": transcription}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Accept-Language": language
    }
    try:
        response = requests.post(ollama_url, json=payload, headers=headers)
        response.raise_for_status()
        processed_text = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"Processed text: {processed_text}")
        
        summary_file = os.path.join(app.config['SUMMARIZED_FOLDER'], generate_unique_filename('txt', custom_name))
        with open(summary_file, 'w') as f:
            f.write(processed_text)
        print(f"Summary saved to {summary_file}")

        return summary_file, processed_text
    except requests.RequestException as e:
        print(f"Error processing with Ollama: {e}")
        raise

def execute_with_timeout(func, *args, timeout=REQUEST_TIMEOUT, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

@app.route('/', methods=['GET', 'POST'])
def index():
    transcribed_files = os.listdir(app.config['TRANSCRIBED_FOLDER'])
    summarized_files = os.listdir(app.config['SUMMARIZED_FOLDER'])
    if request.method == 'POST':
        language = request.form['language']
        custom_name = request.form['custom_name']
        file = request.files['file']
        if 'transcribe' in request.form:
            if file:
                extension = file.filename.rsplit('.', 1)[1]
                unique_filename = generate_unique_filename(extension, custom_name)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                try:
                    mp3_filename = generate_unique_filename('mp3', custom_name)
                    mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
                    execute_with_timeout(convert_mp4_to_mp3, file_path, mp3_path)
                    transcription_file, transcription = execute_with_timeout(transcribe_audio, mp3_path, language, custom_name)
                    flash('Transcription started. The file will be available in the "Transcribed Files" section soon.', 'success')
                    return redirect(url_for('index'))
                except concurrent.futures.TimeoutError:
                    flash('An error occurred: Request timed out.', 'danger')
                    return redirect(url_for('index'))
                except Exception as e:
                    flash(f'An error occurred: {str(e)}', 'danger')
                    return redirect(url_for('index'))
        elif 'transcribe_and_resume' in request.form:
            if file:
                extension = file.filename.rsplit('.', 1)[1]
                unique_filename = generate_unique_filename(extension, custom_name)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                try:
                    mp3_filename = generate_unique_filename('mp3', custom_name)
                    mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
                    execute_with_timeout(convert_mp4_to_mp3, file_path, mp3_path)
                    transcription_file, transcription = execute_with_timeout(transcribe_audio, mp3_path, language, custom_name)
                    summary_file, summary = execute_with_timeout(process_with_ollama, transcription, language, "Resume the Key points of this audio", custom_name)
                    flash('Transcription and summary started. The files will be available in the "Transcribed Files" and "Summarized Files" sections soon.', 'success')
                    return redirect(url_for('index'))
                except concurrent.futures.TimeoutError:
                    flash('An error occurred: Request timed out.', 'danger')
                    return redirect(url_for('index'))
                except Exception as e:
                    flash(f'An error occurred: {str(e)}', 'danger')
                    return redirect(url_for('index'))
    return render_template('index.html', transcribed_files=transcribed_files, summarized_files=summarized_files)

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['TRANSCRIBED_FOLDER'], filename)

@app.route('/summarized/<filename>')
def download_summary(filename):
    return send_from_directory(app.config['SUMMARIZED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
