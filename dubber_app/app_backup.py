import os
import sys
import types
import time
import tempfile
import threading
import traceback
import re
import asyncio
import numpy as np
import librosa
from langdetect import detect

# ── Monkey-patch transformers.onnx ──
onnx_module = types.ModuleType('transformers.onnx')
onnx_module.OnnxConfig = object
onnx_module.OnnxSeq2SeqConfigWithPast = object
onnx_module.__path__ = []
utils_module = types.ModuleType('transformers.onnx.utils')
utils_module.compute_effective_axis_dimension = lambda *args, **kwargs: 0
sys.modules['transformers.onnx'] = onnx_module
sys.modules['transformers.onnx.utils'] = utils_module

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import speech_recognition as sr
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
import edge_tts
import ollama

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ── Job store ──────────────────────────────────────────────────────────────────
jobs = {}

# ── Multi-Gender Voice Mapping ────────────────────────────────────────────────
VOICE_MAP = {
    'kn': {'male': 'kn-IN-GaganNeural', 'female': 'kn-IN-SapnaNeural'},
    'hi': {'male': 'hi-IN-MadhurNeural', 'female': 'hi-IN-SwaraNeural'},
    'ta': {'male': 'ta-IN-ValluvarNeural', 'female': 'ta-IN-PallaviNeural'},
    'te': {'male': 'te-IN-MohanNeural', 'female': 'te-IN-ShrutiNeural'},
    'ml': {'male': 'ml-IN-MidhunNeural', 'female': 'ml-IN-SobhanaNeural'},
    'en': {'male': 'en-IN-PrabhatNeural', 'female': 'en-IN-NeerjaNeural'}
}

# ── AI Logic ──────────────────────────────────────────────────────────────────
def ai_refine_text(text: str) -> str:
    if not text.strip(): return ""
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': 'You are a transcript editor. Correct grammar, punctuation, and typos. Return ONLY the corrected text.'},
            {'role': 'user', 'content': text}
        ])
        return response['message']['content'].strip()
    except Exception: return text

def ai_summarize_text(text: str) -> str:
    if not text.strip(): return ""
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': 'Summarize in 3 concise bullet points. Return ONLY the points.'},
            {'role': 'user', 'content': text}
        ])
        return response['message']['content'].strip()
    except Exception: return "Summary unavailable."

# ── Audio Analysis (Auto-Detection) ───────────────────────────────────────────
def detect_gender(audio_path: str) -> str:
    try:
        y, sr_rate = librosa.load(audio_path, sr=16000, duration=15)
        harmonic = librosa.effects.harmonic(y)
        f0 = librosa.yin(harmonic, fmin=75, fmax=300)
        valid_f0 = f0[(f0 > 70) & (f0 < 300)]
        if len(valid_f0) == 0: return "male"
        median_pitch = np.median(valid_f0)
        return "female" if median_pitch > 140 else "male"
    except Exception: return "male"

def translate_blocks(text: str, target_lang: str) -> str:
    if not text.strip(): return ""
    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences: sentences = [text]
        blocks = []
        for i in range(0, len(sentences), 3):
            blocks.append(" ".join(sentences[i:i+3]))
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_blocks = [translator.translate(b) for b in blocks]
        return " ".join(translated_blocks)
    except Exception: return text

# ── Background job ─────────────────────────────────────────────────────────────
def process_job(job_id, video_path, target_lang, voice_gender):
    try:
        def upd(status, progress, log):
            jobs[job_id].update({'status': status, 'progress': progress, 'log': log})

        upd('analyzing', 10, 'Preparing audio...')
        video = VideoFileClip(video_path)
        tmp_wav = tempfile.mktemp(suffix='.wav')
        video.audio.write_audiofile(tmp_wav, codec='pcm_s16le', logger=None)
        video.close()

        # Voice Selection
        if voice_gender == 'auto':
            gender = detect_gender(tmp_wav)
        else:
            gender = voice_gender
        
        jobs[job_id]['gender'] = gender
        upd('transcribing', 25, f'Using {gender.upper()} voice. Transcribing...')

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = recognizer.record(source)

        try:
            raw_text = recognizer.recognize_google(audio_data)
        except Exception:
            jobs[job_id].update({'status': 'error', 'error': 'Speech recognition failed.'})
            return
        finally:
            if os.path.exists(tmp_wav): os.remove(tmp_wav)

        upd('refining', 45, 'Refining with AI...')
        full_src = ai_refine_text(raw_text)

        upd('summarizing', 55, 'Generating summary...')
        summary = ai_summarize_text(full_src)
        jobs[job_id]['summary'] = summary

        upd('translating', 65, f'Translating to {target_lang}...')
        full_tgt = translate_blocks(full_src, target_lang)
        
        upd('dubbing', 80, f'Generating {gender} AI voice...')
        tmp_mp3 = tempfile.mktemp(suffix='.mp3')
        voice_set = VOICE_MAP.get(target_lang, VOICE_MAP['kn'])
        VOICE = voice_set.get(gender, voice_set['male'])
        
        async def generate_tts():
            communicate = edge_tts.Communicate(full_tgt, VOICE)
            await communicate.save(tmp_mp3)
        
        try:
            asyncio.run(asyncio.wait_for(generate_tts(), timeout=40))
        except Exception:
            jobs[job_id].update({'status': 'error', 'error': 'Voice synthesis failed.'})
            return

        upd('dubbing', 90, 'Building final video...')
        out_filename = f"dubbed_{job_id}_{os.path.basename(video_path)}"
        out_path = os.path.join(os.path.abspath(app.config['OUTPUT_FOLDER']), out_filename)
        video2 = VideoFileClip(video_path)
        tgt_audio = AudioFileClip(tmp_mp3)
        final = video2.with_audio(tgt_audio)
        final.write_videofile(out_path, codec='libx264', audio_codec='aac', logger=None, threads=4, preset='ultrafast')
        video2.close(); tgt_audio.close()
        if os.path.exists(tmp_mp3): os.remove(tmp_mp3)

        jobs[job_id].update({
            'status': 'done',
            'progress': 100,
            'log': 'Done!',
            'english': full_src,
            'kannada': full_tgt,
            'summary': summary,
            'video_url': f'/outputs/{out_filename}',
        })

    except Exception as e:
        jobs[job_id].update({'status': 'error', 'error': str(e)})
        traceback.print_exc()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    target_lang = request.form.get('target_lang', 'kn')
    voice_gender = request.form.get('voice_gender', 'male')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    job_id = str(int(time.time() * 1000))
    jobs[job_id] = {'status': 'queued', 'progress': 0}
    threading.Thread(target=process_job, args=(job_id, filepath, target_lang, voice_gender), daemon=True).start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id); return jsonify(job) if job else (jsonify({'error': 'Not found'}), 404)

@app.route('/translate_text', methods=['POST'])
def translate_text_route():
    data = request.json
    translated = GoogleTranslator(source='auto', target=data.get('target_lang', 'en')).translate(data.get('text'))
    return jsonify({'translated': translated})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    job = jobs.get(data.get('job_id'))
    if not job or 'english' not in job: return jsonify({'error': 'Not ready'}), 400
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': f'Answer based on transcript: {job["english"]}'},
            {'role': 'user', 'content': data.get('question')}
        ])
        ans_en = response['message']['content']
        ans_lang = data.get('ans_lang', 'en')
        final_ans = GoogleTranslator(source='en', target=ans_lang).translate(ans_en) if ans_lang != 'en' else ans_en
        return jsonify({'answer': final_ans})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def serve_output(filename): return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__': app.run(debug=False, port=5001, threaded=True)
