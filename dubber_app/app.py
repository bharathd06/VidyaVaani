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
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import json
import io
import datetime
from fpdf import FPDF

# ── Monkey-patch transformers.onnx ──
onnx_module = types.ModuleType('transformers.onnx')
onnx_module.OnnxConfig = object
onnx_module.OnnxSeq2SeqConfigWithPast = object
onnx_module.OnnxConfigWithPast = object
onnx_module.__path__ = []
utils_module = types.ModuleType('transformers.onnx.utils')
utils_module.compute_effective_axis_dimension = lambda *args, **kwargs: 0
sys.modules['transformers.onnx'] = onnx_module
sys.modules['transformers.onnx.utils'] = utils_module

from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
import edge_tts
import ollama
from rag_engine import LocalVectorIndex, chunk_text

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global RAG Vector Index ───────────────────────────────────────────────────
VECTOR_INDEX_FILE = os.path.join(BASE_DIR, 'vector_index.json')
global_vector_index = LocalVectorIndex()
if os.path.exists(VECTOR_INDEX_FILE):
    global_vector_index.load(VECTOR_INDEX_FILE)

# ── VidyaVaani — AI-Powered Multilingual Education Platform ───────────────────
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)



# ── Job store & History ────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_to_history(job_id, job_data):
    history = load_history()
    history[job_id] = job_data
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")

jobs = load_history()

# ── Multi-Gender Voice Mapping ────────────────────────────────────────────────
VOICE_MAP = {
    'kn': {'male': 'kn-IN-GaganNeural', 'female': 'kn-IN-SapnaNeural'},
    'hi': {'male': 'hi-IN-MadhurNeural', 'female': 'hi-IN-SwaraNeural'},
    'ta': {'male': 'ta-IN-ValluvarNeural', 'female': 'ta-IN-PallaviNeural'},
    'te': {'male': 'te-IN-MohanNeural', 'female': 'te-IN-ShrutiNeural'},
    'ml': {'male': 'ml-IN-MidhunNeural', 'female': 'ml-IN-SobhanaNeural'},
    'en': {'male': 'en-IN-PrabhatNeural', 'female': 'en-IN-NeerjaNeural'}
}

# ── IndicTrans2 Model Loading ─────────────────────────────────────────────────
print("Loading AI4Bharat IndicTrans2 model into GPU... This will take a moment.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for translation: {DEVICE}")

HF_TOKEN = "YOUR HUGGING FACE TOKEN"
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

xliterator = UnicodeIndicTransliterator()
print("IndicTrans2 model loaded successfully!")

# ── Whisper STT Model (CPU int8 — fast, offline, handles foreign accents) ──────
print("Loading Whisper 'small' model for speech recognition (CPU mode)...")
whisper_model = WhisperModel(
    "small",
    device="cpu",          # CPU avoids cublas/CUDA version conflicts
    compute_type="int8"    # int8 is fast on CPU, no GPU cuBLAS needed
)
print("Whisper model loaded!")

INDIC_LANG_MAP = {
    'kn': 'kan_Knda',
    'hi': 'hin_Deva',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'ml': 'mal_Mlym',
    'en': 'eng_Latn'
}

# ── Domain-Specific Glossaries & AI Prompts ──────────────────────────────────
DOMAIN_GLOSSARIES = {
    'general': {
        'whisper_prompt': "Namaste, welcome to this educational tutorial. Let's learn about some key concepts.",
        'ai_refine_instructions': "You are an educational transcript editor. Correct grammar, punctuation, and typos. Retain standard academic terms.",
        'ai_summary_instructions': "Summarize this educational content in 3 concise, clear bullet points."
    },
    'cs': {
        'whisper_prompt': "TCP/IP, DHCP, DNS, LAN, WAN, router, subnet, packet, protocol, OSPF, BGP, SQL, API, database, latency, bandwidth, CPU, GPU, RAM, compiler, algorithm, data structure, array, binary search, Big O, class, object, stack, queue.",
        'ai_refine_instructions': "You are a Computer Science and IT lecture transcript editor. Correct grammar, punctuation, typos, and technical terminology (like network protocols, code syntax, SQL, APIs, etc.). Keep technical acronyms in English uppercase.",
        'ai_summary_instructions': "Summarize this computer science/tech lecture in 3 concise, clear bullet points. Use standard terminology and highlight key concepts."
    },
    'physics_chem': {
        'whisper_prompt': "quantum, thermodynamics, entropy, enthalpy, molecule, atom, proton, neutron, electron, stoichiometry, catalysis, relativity, gravity, force, velocity, acceleration, speed of light, photon, isotope, covalent bond.",
        'ai_refine_instructions': "You are a Physics and Chemistry lecture transcript editor. Correct grammar, punctuation, typos, and scientific terms/formulas. Ensure elements, units, and theories are spelled correctly.",
        'ai_summary_instructions': "Summarize this physics/chemistry content in 3 concise, clear bullet points. Highlight key equations, laws, or chemical reactions."
    },
    'math': {
        'whisper_prompt': "calculus, derivative, integral, matrix, linear algebra, vector, equation, theorem, lemma, geometry, trigonometry, sine, cosine, tangent, probability, statistic, distribution, standard deviation, variance.",
        'ai_refine_instructions': "You are a Mathematics lecture transcript editor. Correct grammar, punctuation, typos, and mathematical terminology (calculus terms, equations, functions, geometric formulas, etc.).",
        'ai_summary_instructions': "Summarize this mathematical lecture in 3 concise, clear bullet points. Break down the core mathematical theorems, formulas, or methods discussed."
    },
    'biology': {
        'whisper_prompt': "photosynthesis, cellular, DNA, RNA, gene, chromosome, protein, enzyme, mitochondria, nucleus, cytoplasm, ecosystem, species, evolution, anatomy, neuron, cardiovascular, respiration, photosynthesis, homeostasis.",
        'ai_refine_instructions': "You are a Biology and Life Sciences lecture transcript editor. Correct grammar, punctuation, typos, and complex biological terms (organelles, genetics, species names, physiological systems).",
        'ai_summary_instructions': "Summarize this biological lecture in 3 concise, clear bullet points. Focus on biological processes, functions, or anatomical systems."
    }
}

def translate_text_indic(text: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    if target_lang == 'en':
        return text
    try:
        tgt_tag = INDIC_LANG_MAP.get(target_lang, 'kan_Knda')
        tagged_text = f"eng_Latn {tgt_tag} {text.strip()}"
        
        inputs = tokenizer([tagged_text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        
        with torch.inference_mode():
            generated_tokens = model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=1,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
            
        generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # Transliterate script from Devanagari (hi) to native language script if necessary
        if target_lang != 'hi':
            final_text = xliterator.transliterate(generated_text, "hi", target_lang)
        else:
            final_text = generated_text
        return final_text
    except Exception as e:
        print(f"[IndicTrans2 Error: {e}]. Falling back to Google Translate.")
        try:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception:
            return text
    finally:
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

# ── AI Logic ──────────────────────────────────────────────────────────────────
def ai_refine_text(text: str, domain: str = 'general') -> str:
    if not text.strip(): return ""
    try:
        sys_prompt = DOMAIN_GLOSSARIES.get(domain, DOMAIN_GLOSSARIES['general'])['ai_refine_instructions']
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': f"{sys_prompt} Return ONLY the corrected text."},
            {'role': 'user', 'content': text}
        ])
        return response['message']['content'].strip()
    except Exception: return text

def ai_summarize_text(text: str, domain: str = 'general') -> str:
    if not text.strip(): return ""
    try:
        sys_prompt = DOMAIN_GLOSSARIES.get(domain, DOMAIN_GLOSSARIES['general'])['ai_summary_instructions']
        # Split text into chunks
        chunks = chunk_text(text, chunk_size=3000, overlap=300)
        if not chunks:
            return "Summary unavailable."
        
        # Map step: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"[RAG Summary] Mapping chunk {i+1}/{len(chunks)}")
            response = ollama.chat(model='mistral', messages=[
                {'role': 'system', 'content': "You are an educational assistant. Summarize this section of a lecture in 2-3 short bullet points. Be technical and precise."},
                {'role': 'user', 'content': chunk}
            ])
            summary_part = response['message']['content'].strip()
            chunk_summaries.append(summary_part)
            
        if len(chunks) == 1:
            # Single chunk: apply direct reduce
            response = ollama.chat(model='mistral', messages=[
                {'role': 'system', 'content': f"{sys_prompt} Return ONLY the points."},
                {'role': 'user', 'content': chunk_summaries[0]}
            ])
            return response['message']['content'].strip()
            
        # Reduce step: Combine maps and synthesize final summary
        combined_summaries = "\n".join(chunk_summaries)
        print(f"[RAG Summary] Reducing {len(chunks)} summaries into global summary...")
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': f"{sys_prompt} Return ONLY the points."},
            {'role': 'user', 'content': f"Here are summaries of different parts of the lecture:\n{combined_summaries}"}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"[RAG Summary Error]: {e}")
        return "Summary unavailable."

def translate_blocks(text: str, target_lang: str) -> str:
    if not text.strip(): return ""
    try:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences: sentences = [text]
        blocks = []
        for i in range(0, len(sentences), 3):
            blocks.append(" ".join(sentences[i:i+3]))
        translated_blocks = [translate_text_indic(b, target_lang) for b in blocks]
        return " ".join(translated_blocks)
    except Exception: return text

# ── Background job (Batch Processing) ──────────────────────────────────────────
def process_job(job_id, video_path, target_lang, voice_gender, domain='general'):
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
            gender = 'male'
        else:
            gender = voice_gender
        
        jobs[job_id]['gender'] = gender
        upd('transcribing', 25, f'Using {gender.upper()} voice. Transcribing with Whisper AI...')

        # ── Whisper transcription (handles foreign accents, long audio, offline) ──
        try:
            whisper_prompt = DOMAIN_GLOSSARIES.get(domain, DOMAIN_GLOSSARIES['general'])['whisper_prompt']
            segments, info = whisper_model.transcribe(
                tmp_wav,
                beam_size=5,
                language="en",          # force English; change to None for auto-detect
                initial_prompt=whisper_prompt,
                vad_filter=True,        # skip silence automatically
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            raw_text = " ".join(seg.text.strip() for seg in segments).strip()
            if not raw_text:
                jobs[job_id].update({'status': 'error', 'error': 'No speech detected in the video. Make sure the video has clear audio.'})
                return
        except Exception as e:
            jobs[job_id].update({'status': 'error', 'error': f'Transcription failed: {e}'})
            return
        finally:
            if os.path.exists(tmp_wav): os.remove(tmp_wav)

        upd('refining', 45, 'Refining with AI...')
        full_src = ai_refine_text(raw_text, domain)

        # Build Vector Index for RAG
        try:
            print(f"[RAG Indexing] Adding video {job_id} to vector index...")
            global_vector_index.add_document(job_id, full_src, domain)
            global_vector_index.build_index()
            global_vector_index.save(VECTOR_INDEX_FILE)
            print("[RAG Indexing] Vector index saved successfully.")
        except Exception as ve:
            print(f"[RAG Indexing Error]: {ve}")

        upd('summarizing', 55, 'Generating summary...')
        summary = ai_summarize_text(full_src, domain)
        jobs[job_id]['summary'] = summary

        upd('translating', 65, f'Translating to {target_lang}...')
        full_tgt = translate_blocks(full_src, target_lang)
        
        upd('translating', 80, f'Generating {gender} AI voice...')
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

        upd('translating', 90, 'Building final video...')
        out_filename = f"translated_{job_id}_{os.path.basename(video_path)}"
        out_path = os.path.join(os.path.abspath(app.config['OUTPUT_FOLDER']), out_filename)
        video2 = VideoFileClip(video_path)
        tgt_audio = AudioFileClip(tmp_mp3)
        final = video2.with_audio(tgt_audio)
        final.write_videofile(out_path, codec='libx264', audio_codec='aac', logger=None, threads=4, preset='ultrafast')
        video2.close(); tgt_audio.close()
        if os.path.exists(tmp_mp3): os.remove(tmp_mp3)

        # Extract topics / tags using Mistral
        topics = []
        try:
            topic_prompt = f"""From this educational video transcript, identify exactly 3 key specific topics/concepts discussed.
Return ONLY a JSON array of 3 short topic strings (each under 6 words).
Example: ["TCP/IP Protocol Stack", "Subnet Masking", "OSI Reference Model"]

Transcript:
{full_src[:1500]}

Return ONLY the JSON array, no extra text."""

            topic_response = ollama.chat(model='mistral', messages=[
                {'role': 'user', 'content': topic_prompt}
            ])
            topic_content = topic_response['message']['content'].strip()
            if topic_content.startswith('```'):
                topic_content = '\n'.join(topic_content.split('\n')[1:-1]).strip()

            import json as _json
            topics = _json.loads(topic_content)
            if not isinstance(topics, list):
                topics = [str(topic_content)]
            topics = [t.strip() for t in topics[:3] if t.strip()]
        except Exception:
            # Fallback based on domain
            fallbacks = {
                'cs': ["TCP/IP Model", "Network Layers", "Data Communication"],
                'physics_chem': ["Organic Chemistry", "Atomic Physics", "Thermodynamics"],
                'math': ["Calculus", "Linear Algebra", "Probability & Stats"],
                'biology': ["Cell Structure", "Genetics", "Human Anatomy"],
                'general': ["Critical Thinking", "Study Skills", "Personal Development"]
            }
            topics = fallbacks.get(domain, fallbacks['general'])

        clean_title = os.path.basename(video_path).rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
        jobs[job_id].update({
            'status': 'done',
            'progress': 100,
            'log': 'Done!',
            'title': clean_title,
            'target_lang': target_lang,
            'domain': domain,
            'timestamp': datetime.datetime.now().strftime("%d %b %Y, %I:%M %p"),
            'english': full_src,
            'kannada': full_tgt,
            'summary': summary,
            'video_url': f'/outputs/{out_filename}',
            'topics': topics
        })
        save_to_history(job_id, jobs[job_id])

    except Exception as e:
        jobs[job_id].update({'status': 'error', 'error': str(e)})
        traceback.print_exc()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): 
    return render_template('index.html')

# ── Core App Routes ────────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    target_lang = request.form.get('target_lang', 'kn')
    voice_gender = request.form.get('voice_gender', 'male')
    domain = request.form.get('domain', 'general')
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    job_id = str(int(time.time() * 1000))

    jobs[job_id] = {'status': 'queued', 'progress': 0}
    threading.Thread(target=process_job, args=(job_id, filepath, target_lang, voice_gender, domain), daemon=True).start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if job:
        j_copy = job.copy()
        j_copy['job_id'] = job_id
        return jsonify(j_copy)
    return jsonify({'error': 'Not found'}), 404

@app.route('/history_list', methods=['GET'])
def history_list():
    history = load_history()
    list_data = []
    for jid, job in history.items():
        if job.get('status') == 'done':
            list_data.append({
                'job_id': jid,
                'title': job.get('title', 'Untitled Video'),
                'target_lang': job.get('target_lang', 'kn'),
                'domain': job.get('domain', 'general'),
                'timestamp': job.get('timestamp', ''),
                'video_url': job.get('video_url', '')
            })
    list_data.sort(key=lambda x: x['job_id'], reverse=True)
    return jsonify(list_data)

@app.route('/translate_text', methods=['POST'])
def translate_text_route():
    data = request.json
    translated = translate_text_indic(data.get('text'), data.get('target_lang', 'en'))
    return jsonify({'translated': translated})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    job_id = data.get('job_id')
    ans_lang = data.get('ans_lang', 'en')
    question = data.get('question')
    
    if job_id == 'general':
        try:
            # Global RAG Search across ALL transcripts in the LangChain Chroma index
            print(f"[LangChain RAG Chat] Querying global Chroma index for: {question}")
            relevant_chunks = global_vector_index.query(question, top_k=4)
            
            history = load_history()
            sources = []
            if relevant_chunks:
                context_parts = []
                for rc in relevant_chunks:
                    jid = rc['metadata'].get('job_id')
                    video_title = history.get(jid, {}).get('title', 'Unknown Lecture')
                    context_parts.append(f"[Lecture: {video_title}] {rc['chunk']}")
                    sources.append({
                        'lecture': video_title,
                        'chunk': rc['chunk'][:140] + ('...' if len(rc['chunk']) > 140 else ''),
                        'score': round(rc.get('score', 0.0), 3)
                    })
                context_text = "\n\n".join(context_parts)
                sys_content = (
                    f"You are VidyaVaani, an expert multilingual academic tutor. "
                    f"Answer the student's question clearly, concisely, and educationally using the following retrieved lecture transcript excerpts. "
                    f"Treat these excerpts as the lecture video content itself. Speak naturally and authoritatively without meta-disclaimers like 'the context does not specify a video':\n\n{context_text}"
                )
            else:
                sys_content = 'You are VidyaVaani, an expert multilingual academic tutor. Answer the student\'s question clearly, concisely, and educationally. Support and encourage scientific inquiry and critical thinking.'
                
            response = ollama.chat(model='mistral', messages=[
                {'role': 'system', 'content': sys_content},
                {'role': 'user', 'content': question}
            ])
            ans_en = response['message']['content']
            final_ans = translate_text_indic(ans_en, ans_lang) if ans_lang != 'en' else ans_en
            return jsonify({'answer': final_ans, 'sources': sources})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    job = jobs.get(job_id)
    if not job or 'english' not in job: return jsonify({'error': 'Not ready'}), 400
    try:
        # Check if this specific video is indexed; if not, index it on the fly into Chroma
        has_chunks = any(meta.get('job_id') == job_id for meta in global_vector_index.metadata)
        if not has_chunks:
            print(f"[LangChain RAG Chat] Job {job_id} not indexed. Indexing into Chroma on the fly...")
            global_vector_index.add_document(job_id, job['english'], job.get('domain', 'general'))
            global_vector_index.build_index()
            global_vector_index.save(VECTOR_INDEX_FILE)
            
        # Local RAG search within this specific video
        relevant_chunks = global_vector_index.query(question, filter_job_id=job_id, top_k=3)
        context_text = "\n\n".join([rc['chunk'] for rc in relevant_chunks])
        lec_title = job.get('title', 'this lecture')
        sources = [{
            'lecture': lec_title,
            'chunk': rc['chunk'][:140] + ('...' if len(rc['chunk']) > 140 else ''),
            'score': round(rc.get('score', 0.0), 3)
        } for rc in relevant_chunks]
        
        sys_content = (
            f"You are VidyaVaani, an expert multilingual academic tutor. "
            f"Answer the student's question directly about the lecture video '{lec_title}' based on the following transcript content. "
            f"Treat this transcript as the actual lecture video content. Speak naturally and authoritatively without meta-disclaimers like 'the context does not specify a video':\n\n{context_text}"
        )
        
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': sys_content},
            {'role': 'user', 'content': question}
        ])
        ans_en = response['message']['content']
        final_ans = translate_text_indic(ans_en, ans_lang) if ans_lang != 'en' else ans_en
        return jsonify({'answer': final_ans, 'sources': sources})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/rag_status', methods=['GET'])
def rag_status():
    """Telemetry endpoint reporting LangChain RAG index status."""
    try:
        return jsonify(global_vector_index.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    job_id = data.get('job_id')
    job = jobs.get(job_id)
    if not job or 'english' not in job:
        return jsonify({'error': 'Job not ready or not found'}), 400
    
    transcript = job['english']
    if not transcript.strip():
        return jsonify({'quiz': []})
    
    try:
        # Check if indexed; if not, index it on the fly
        has_chunks = any(meta['job_id'] == job_id for meta in global_vector_index.metadata)
        if not has_chunks:
            print(f"[RAG Quiz] Job {job_id} not indexed. Indexing on the fly...")
            global_vector_index.add_document(job_id, transcript, job.get('domain', 'general'))
            global_vector_index.build_index()
            global_vector_index.save(VECTOR_INDEX_FILE)
            
        # Get chunks for this job
        video_chunks = [chunk for chunk, meta in zip(global_vector_index.chunks, global_vector_index.metadata) if meta['job_id'] == job_id]
        
        if not video_chunks:
            video_chunks = chunk_text(transcript)
            
        # Select up to 3 chunks distributed across the transcript (first, middle, last) to ensure coverage
        selected_chunks = []
        num_chunks = len(video_chunks)
        if num_chunks >= 3:
            indices = [0, num_chunks // 2, num_chunks - 1]
            selected_chunks = [video_chunks[i] for i in indices]
        elif num_chunks > 0:
            selected_chunks = video_chunks
            # Pad to 3 chunks if we have fewer
            while len(selected_chunks) < 3:
                selected_chunks.append(video_chunks[len(selected_chunks) % len(video_chunks)])
        else:
            selected_chunks = [transcript]
            
        quiz = []
        for i, chunk in enumerate(selected_chunks[:3]):
            print(f"[RAG Quiz] Generating MCQ {i+1} from chunk segment...")
            prompt = f"""
            Analyze this lecture snippet:
            {chunk}

            Based on the snippet, generate exactly 1 multiple-choice question (MCQ) to test the viewer's understanding.
            Return ONLY a valid JSON object. Do NOT wrap it in backticks, markdown code blocks, or include any extra text.

            The object must have these exact keys:
            1. "question": The question text.
            2. "options": A list of exactly 4 choices (strings).
            3. "answer_index": The index (0 to 3) of the correct choice.
            4. "explanation": A brief explanation of why this answer is correct based on the snippet.

            Format:
            {{"question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "..."}}
            """
            response = ollama.chat(model='mistral', messages=[
                {'role': 'user', 'content': prompt}
            ])
            content = response['message']['content'].strip()
            if content.startswith("```"):
                lines = content.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            import json
            q_obj = json.loads(content)
            # Add citation metadata snippet from transcript
            q_obj['citation'] = chunk[:120] + "..."
            quiz.append(q_obj)
            
        return jsonify({'quiz': quiz})
    except Exception as e:
        print(f"Quiz generation error: {e}")
        fallback_quiz = [
            {
                "question": "What is the primary topic of the video?",
                "options": ["General Discussion", "Speech and Language Translation", "Machine Learning Optimization", "Continuous Integration"],
                "answer_index": 1,
                "explanation": "The video transcript focuses on speech processing, transcription, and translation.",
                "citation": "Speech translation systems enable multilingual learning..."
            }
        ]
        return jsonify({'quiz': fallback_quiz})

@app.route('/outputs/<filename>')
def serve_output(filename): 
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# ── Notes Generation & PDF Routes ─────────────────────────────────────────────

LANG_NAMES = {
    'en': 'English', 'hi': 'Hindi (हिन्दी)', 'kn': 'Kannada (ಕನ್ನಡ)',
    'ta': 'Tamil (தமிழ்)', 'te': 'Telugu (తెలుగు)', 'ml': 'Malayalam (മലയാളം)'
}

# Nirmala.ttc supports Devanagari (Hindi), Kannada, Tamil, Telugu, Malayalam
NIRMALA_FONT = 'C:/Windows/Fonts/Nirmala.ttc'
ARIAL_FONT   = 'C:/Windows/Fonts/arial.ttf'

def get_font_path(lang):
    if lang == 'en':
        return ARIAL_FONT if os.path.exists(ARIAL_FONT) else None
    return NIRMALA_FONT if os.path.exists(NIRMALA_FONT) else None

def ai_generate_notes(transcript: str) -> str:
    """Use local Ollama/Mistral to produce structured markdown study notes."""
    if not transcript.strip():
        return ""
    prompt = f"""You are an expert academic note-taker. Based on the following video transcript, create highly detailed, comprehensive, and exhaustive study notes for a student. Make sure to elaborate on every technical concept mentioned in the transcript without leaving out technical specifics.

Transcript:
{transcript}

Create study notes with EXACTLY this structure (use these exact headings):

## 📌 Topic
[Provide a clear, detailed explanation of the overall topic, its educational context, and its significance]

## 🧠 Key Concepts
[Identify and thoroughly explain all key concepts, technical terms, and core ideas in detail as bullet points. Provide detailed descriptions for each concept rather than brief summaries]

## 📝 Detailed Notes
[Write an in-depth, highly structured analysis covering the main content in full detail, organized logically into multiple descriptive paragraphs or sections. Incorporate any specific definitions, theories, architectures, or step-by-step processes discussed in the transcript]

## 💡 Key Takeaways
[List all important lessons, summaries, and critical insights to remember as highly descriptive bullet points]

## ❓ Review Questions
[Provide 4-5 deep conceptual review questions that test high-level understanding of the material]

Return ONLY the notes. Be extremely thorough, educational, detailed, and clear."""
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'system', 'content': 'You are an expert academic note-taker creating study notes for students.'},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Notes generation error: {e}")
        return ""

def build_notes_pdf(notes_en: str, notes_translated: str, lang: str, video_name: str) -> bytes:
    """Build a styled PDF with the notes content. Returns PDF bytes."""
    font_path = get_font_path(lang)
    use_unicode = font_path is not None and lang != 'en'
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # Register fonts
    try:
        pdf.add_font('Arial', fname=ARIAL_FONT)
        if use_unicode and os.path.exists(NIRMALA_FONT):
            pdf.add_font('Nirmala', fname=NIRMALA_FONT)
        else:
            use_unicode = False
    except Exception:
        use_unicode = False
    
    body_font = 'Nirmala' if use_unicode else 'Arial'
    
    # ── Header Banner ──
    pdf.set_fill_color(30, 20, 60)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_font('Arial', size=18)
    pdf.set_text_color(180, 150, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, 'SMART MULTILINGUAL TRANSLATOR', ln=True, align='C')
    pdf.set_font('Arial', size=10)
    pdf.set_text_color(150, 150, 200)
    pdf.set_xy(10, 20)
    pdf.cell(0, 8, 'AI-Generated Study Notes', ln=True, align='C')
    pdf.set_xy(10, 29)
    pdf.cell(0, 7, f'Language: {LANG_NAMES.get(lang, lang)}   |   Generated: {datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")}', ln=True, align='C')
    
    pdf.ln(12)
    
    # ── Video Title ──
    pdf.set_fill_color(45, 35, 80)
    pdf.set_draw_color(100, 70, 200)
    pdf.set_text_color(220, 200, 255)
    pdf.set_font('Arial', size=11)
    clean_name = video_name[:70] + ('...' if len(video_name) > 70 else '')
    pdf.set_x(10)
    pdf.cell(190, 10, f'  Video: {clean_name}', border='B', ln=True, fill=True)
    pdf.ln(5)
    
    # ── Notes content ──
    notes_to_render = notes_translated if lang != 'en' else notes_en
    
    SECTION_COLORS = {
        'topic':     (139, 92,  246),
        'concepts':  (6,   182, 212),
        'notes':     (16,  185, 129),
        'takeaways': (245, 158, 11),
        'questions': (236, 72,  153),
        'default':   (100, 116, 139),
    }
    
    def section_color(heading_lower):
        if 'topic' in heading_lower:    return SECTION_COLORS['topic']
        if 'concept' in heading_lower:  return SECTION_COLORS['concepts']
        if 'detail' in heading_lower or 'note' in heading_lower: return SECTION_COLORS['notes']
        if 'takeaway' in heading_lower: return SECTION_COLORS['takeaways']
        if 'review' in heading_lower or 'question' in heading_lower: return SECTION_COLORS['questions']
        return SECTION_COLORS['default']
    
    lines = notes_to_render.split('\n')
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(3)
            continue
        
        # Section headings: ## or #
        if stripped.startswith('##') or stripped.startswith('#'):
            heading_text = stripped.lstrip('#').strip()
            r, g, b = section_color(heading_text.lower())
            pdf.ln(4)
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Arial', size=12)
            pdf.set_x(10)
            pdf.cell(190, 9, f'  {heading_text}', ln=True, fill=True)
            pdf.set_draw_color(r, g, b)
            pdf.ln(3)
        
        # Bullet points
        elif stripped.startswith('- ') or stripped.startswith('• ') or stripped.startswith('* '):
            content = stripped[2:].strip()
            pdf.set_text_color(40, 40, 40)
            pdf.set_font(body_font, size=10)
            pdf.set_x(15)
            # Bullet symbol
            pdf.set_text_color(109, 40, 217)
            pdf.cell(5, 7, '▸')
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(20)
            try:
                pdf.multi_cell(175, 7, content, ln=True)
            except Exception:
                pdf.multi_cell(175, 7, content.encode('latin-1', errors='replace').decode('latin-1'), ln=True)
        
        # Numbered lines
        elif re.match(r'^\d+\.', stripped):
            pdf.set_text_color(40, 40, 40)
            pdf.set_font(body_font, size=10)
            pdf.set_x(15)
            try:
                pdf.multi_cell(180, 7, stripped, ln=True)
            except Exception:
                pdf.multi_cell(180, 7, stripped.encode('latin-1', errors='replace').decode('latin-1'), ln=True)
        
        # Normal paragraph text
        else:
            pdf.set_text_color(50, 50, 50)
            pdf.set_font(body_font, size=10)
            pdf.set_x(10)
            try:
                pdf.multi_cell(190, 7, stripped, ln=True)
            except Exception:
                pdf.multi_cell(190, 7, stripped.encode('latin-1', errors='replace').decode('latin-1'), ln=True)
        
        pdf.set_draw_color(50, 50, 80)
    
    # ── Footer on each page ──
    pdf.set_y(-15)
    pdf.set_font('Arial', size=8)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 10, f'Smart Multilingual Translator  •  AI Study Notes  •  Page {pdf.page_no()}', align='C')
    
    return bytes(pdf.output())


@app.route('/generate_notes', methods=['POST'])
def generate_notes():
    data = request.json
    job_id = data.get('job_id')
    language = data.get('language', 'en')
    
    job = jobs.get(job_id)
    if not job or 'english' not in job:
        return jsonify({'error': 'Job not ready or not found'}), 400
    
    transcript = job['english']
    if not transcript.strip():
        return jsonify({'error': 'No transcript available'}), 400
    
    try:
        # Step 1: Generate notes in English using Ollama
        notes_en = ai_generate_notes(transcript)
        if not notes_en:
            return jsonify({'error': 'Failed to generate notes. Is Ollama running?'}), 500
        
        # Step 2: Translate notes to target language if needed
        if language != 'en':
            notes_translated = translate_blocks(notes_en, language)
        else:
            notes_translated = notes_en
        
        # Cache notes in job for PDF download
        jobs[job_id]['notes_en'] = notes_en
        jobs[job_id]['notes_translated'] = notes_translated
        jobs[job_id]['notes_lang'] = language
        
        return jsonify({
            'notes': notes_translated,
            'notes_en': notes_en,
            'language': language,
            'lang_name': LANG_NAMES.get(language, language)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download_notes_pdf', methods=['POST'])
def download_notes_pdf():
    data = request.json
    job_id = data.get('job_id')
    language = data.get('language', 'en')
    video_name = data.get('video_name', 'Video')
    
    job = jobs.get(job_id) if job_id else {}
    notes_en = job.get('notes_en', data.get('notes_en', ''))
    notes_translated = job.get('notes_translated', data.get('notes', notes_en))
    
    if not notes_en:
        return jsonify({'error': 'No notes to download. Generate notes first.'}), 400
    
    try:
        pdf_bytes = build_notes_pdf(notes_en, notes_translated, language, video_name)
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        safe_name = re.sub(r'[^\w\s-]', '', video_name.replace(' ', '_'))[:40]
        filename = f'study_notes_{safe_name}_{language}.pdf'
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True, download_name=filename)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/references/<job_id>', methods=['GET'])
def get_references(job_id):
    """Use Mistral to extract key topics from the transcript and return
    curated textbook recommendations and article links for further reading."""
    job = jobs.get(job_id)
    if not job or 'english' not in job:
        return jsonify({'error': 'Job not ready or not found'}), 400

    transcript = job['english']
    domain = job.get('domain', 'general')

    # Return cached references if already generated
    if 'references' in job:
        return jsonify(job['references'])

    # Domain → standard textbook suggestions (curated fallbacks)
    DOMAIN_BOOKS = {
        'cs': [
            {'title': 'Computer Networks', 'authors': 'Andrew S. Tanenbaum & David J. Wetherall', 'edition': '5th Edition', 'publisher': 'Pearson', 'icon': '💻', 'search': 'Computer+Networks+Tanenbaum'},
            {'title': 'Introduction to Algorithms (CLRS)', 'authors': 'Cormen, Leiserson, Rivest, Stein', 'edition': '4th Edition', 'publisher': 'MIT Press', 'icon': '📐', 'search': 'Introduction+to+Algorithms+CLRS'},
            {'title': 'Operating System Concepts', 'authors': 'Silberschatz, Galvin, Gagne', 'edition': '10th Edition', 'publisher': 'Wiley', 'icon': '⚙️', 'search': 'Operating+System+Concepts+Silberschatz'},
            {'title': 'Database System Concepts', 'authors': 'Silberschatz, Korth, Sudarshan', 'edition': '7th Edition', 'publisher': 'McGraw-Hill', 'icon': '🗄️', 'search': 'Database+System+Concepts'},
        ],
        'physics_chem': [
            {'title': 'University Physics', 'authors': 'Young & Freedman', 'edition': '15th Edition', 'publisher': 'Pearson', 'icon': '⚛️', 'search': 'University+Physics+Young+Freedman'},
            {'title': 'Organic Chemistry', 'authors': 'Paula Yurkanis Bruice', 'edition': '8th Edition', 'publisher': 'Pearson', 'icon': '🧪', 'search': 'Organic+Chemistry+Bruice'},
            {'title': 'Physical Chemistry', 'authors': 'Atkins & de Paula', 'edition': '11th Edition', 'publisher': 'Oxford University Press', 'icon': '🔬', 'search': 'Physical+Chemistry+Atkins'},
            {'title': 'Concepts of Modern Physics', 'authors': 'Arthur Beiser', 'edition': '6th Edition', 'publisher': 'McGraw-Hill', 'icon': '🌌', 'search': 'Concepts+of+Modern+Physics+Beiser'},
        ],
        'math': [
            {'title': 'Calculus: Early Transcendentals', 'authors': 'James Stewart', 'edition': '9th Edition', 'publisher': 'Cengage', 'icon': '∫', 'search': 'Stewart+Calculus+Early+Transcendentals'},
            {'title': 'Linear Algebra and Its Applications', 'authors': 'Gilbert Strang', 'edition': '5th Edition', 'publisher': 'Cengage', 'icon': '📊', 'search': 'Linear+Algebra+Its+Applications+Strang'},
            {'title': 'Discrete Mathematics and Its Applications', 'authors': 'Kenneth H. Rosen', 'edition': '8th Edition', 'publisher': 'McGraw-Hill', 'icon': '🔣', 'search': 'Discrete+Mathematics+Applications+Rosen'},
            {'title': 'Introduction to Probability', 'authors': 'Bertsekas & Tsitsiklis', 'edition': '2nd Edition', 'publisher': 'Athena Scientific', 'icon': '🎲', 'search': 'Introduction+to+Probability+Bertsekas'},
        ],
        'biology': [
            {'title': 'Campbell Biology', 'authors': 'Urry, Cain, Wasserman et al.', 'edition': '12th Edition', 'publisher': 'Pearson', 'icon': '🧬', 'search': 'Campbell+Biology+12th+Edition'},
            {'title': "Guyton and Hall Medical Physiology", 'authors': 'John E. Hall', 'edition': '14th Edition', 'publisher': 'Elsevier', 'icon': '🫀', 'search': 'Guyton+Hall+Medical+Physiology'},
            {'title': 'Molecular Biology of the Cell', 'authors': 'Alberts, Johnson, Lewis et al.', 'edition': '7th Edition', 'publisher': 'Norton', 'icon': '🔬', 'search': 'Molecular+Biology+Cell+Alberts'},
            {'title': 'Biochemistry', 'authors': 'Jeremy M. Berg, John L. Tymoczko', 'edition': '9th Edition', 'publisher': 'Macmillan', 'icon': '⚗️', 'search': 'Biochemistry+Berg+Tymoczko'},
        ],
        'general': [
            {'title': 'Thinking, Fast and Slow', 'authors': 'Daniel Kahneman', 'edition': '1st Edition', 'publisher': 'Farrar, Straus and Giroux', 'icon': '🧠', 'search': 'Thinking+Fast+Slow+Kahneman'},
            {'title': 'The Art of Problem Solving (Vol 1)', 'authors': 'Sandor Lehoczky & Richard Rusczyk', 'edition': '7th Edition', 'publisher': 'AoPS', 'icon': '📚', 'search': 'Art+of+Problem+Solving+Vol1'},
            {'title': 'A Mind for Numbers', 'authors': 'Barbara Oakley', 'edition': '1st Edition', 'publisher': 'Tarcher Perigee', 'icon': '📖', 'search': 'Mind+for+Numbers+Oakley'},
        ]
    }

    try:
        # Re-use pre-extracted topics if present
        if 'topics' in job:
            topics = job['topics']
        else:
            # Ask Mistral to extract the 3 most specific topics from the transcript
            topic_prompt = f"""From this educational video transcript, identify exactly 3 key specific topics/concepts discussed.
Return ONLY a JSON array of 3 short topic strings (each under 6 words).
Example: ["TCP/IP Protocol Stack", "Subnet Masking", "OSI Reference Model"]

Transcript:
{transcript[:1500]}

Return ONLY the JSON array, no extra text."""

            topic_response = ollama.chat(model='mistral', messages=[
                {'role': 'user', 'content': topic_prompt}
            ])
            topic_content = topic_response['message']['content'].strip()

            # Clean up any markdown wrapping
            if topic_content.startswith('```'):
                topic_content = '\n'.join(topic_content.split('\n')[1:-1]).strip()

            import json as _json
            try:
                topics = _json.loads(topic_content)
                if not isinstance(topics, list):
                    topics = [str(topic_content)]
            except Exception:
                # Fallback: extract quoted strings
                topics = re.findall(r'"([^"]+)"', topic_content)[:3]
                if not topics:
                    topics = ['Key Concept 1', 'Key Concept 2', 'Key Concept 3']

            topics = [t.strip() for t in topics[:3] if t.strip()]
            job['topics'] = topics

        # Build article links (Wikipedia + Khan Academy + NPTEL based on topics)
        articles = []
        for topic in topics:
            encoded = topic.replace(' ', '_')
            encoded_q = topic.replace(' ', '+')
            articles.append({
                'topic': topic,
                'links': [
                    {
                        'source': 'Wikipedia',
                        'icon': '📖',
                        'color': '#3366cc',
                        'url': f'https://en.wikipedia.org/wiki/{encoded}',
                        'label': f'Wikipedia — {topic}'
                    },
                    {
                        'source': 'Khan Academy',
                        'icon': '🎓',
                        'color': '#14BF96',
                        'url': f'https://www.khanacademy.org/search?page_search_query={encoded_q}',
                        'label': f'Khan Academy — {topic}'
                    },
                    {
                        'source': 'NPTEL',
                        'icon': '🏛️',
                        'color': '#E07B39',
                        'url': f'https://nptel.ac.in/search?query={encoded_q}',
                        'label': f'NPTEL Lectures — {topic}'
                    },
                    {
                        'source': 'Google Scholar',
                        'icon': '🔍',
                        'color': '#4285F4',
                        'url': f'https://scholar.google.com/scholar?q={encoded_q}',
                        'label': f'Research Papers — {topic}'
                    }
                ]
            })

        # Get curated books for this domain
        books = DOMAIN_BOOKS.get(domain, DOMAIN_BOOKS['general'])

        result = {
            'topics': topics,
            'books': books,
            'articles': articles,
            'domain': domain
        }

        # Cache result in job store
        jobs[job_id]['references'] = result
        save_to_history(job_id, jobs[job_id])

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        # Return domain-based fallback without AI topics
        return jsonify({
            'topics': ['Key Topic 1', 'Key Topic 2', 'Key Topic 3'],
            'books': DOMAIN_BOOKS.get(domain, DOMAIN_BOOKS['general']),
            'articles': [],
            'domain': domain,
            'error': str(e)
        })


@app.route('/all_tags', methods=['GET'])
def get_all_tags():
    """Scan history.json for done items and return unique topics/tags mapping to videos."""
    try:
        history = load_history()
        all_tags = {}
        for jid, job in history.items():
            if job.get('status') == 'done':
                # Try to get topics from job structure
                topics = job.get('topics') or job.get('references', {}).get('topics')
                if not topics:
                    # Fallback based on domain to avoid empty list
                    domain = job.get('domain', 'general')
                    fallbacks = {
                        'cs': ["TCP/IP Model", "Network Layers", "Data Communication"],
                        'physics_chem': ["Organic Chemistry", "Atomic Physics", "Thermodynamics"],
                        'math': ["Calculus", "Linear Algebra", "Probability & Stats"],
                        'biology': ["Cell Structure", "Genetics", "Human Anatomy"],
                        'general': ["Critical Thinking", "Study Skills", "Personal Development"]
                    }
                    topics = fallbacks.get(domain, fallbacks['general'])
                
                for t in topics:
                    t_clean = t.strip()
                    if not t_clean:
                        continue
                    if t_clean not in all_tags:
                        all_tags[t_clean] = []
                    all_tags[t_clean].append({
                        'job_id': jid,
                        'title': job.get('title')
                    })
        return jsonify(all_tags)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_subjective_test', methods=['POST'])
def generate_subjective_test():
    """Use Mistral to generate short-answer conceptual questions grounded in lecture chunks."""
    try:
        data = request.json or {}
        topics = data.get('topics', [])
        num_questions = int(data.get('num_questions', 3))

        if not topics:
            return jsonify({'error': 'No topics selected for the test.'}), 400

        questions = []
        for i in range(num_questions):
            topic = topics[i % len(topics)]
            print(f"[RAG Subjective] Querying context for topic: {topic}")
            relevant = global_vector_index.query(topic, top_k=1)
            
            if relevant:
                ref_chunk = relevant[0]['chunk']
            else:
                ref_chunk = f"Core principles and applications of {topic} in this academic course."

            prompt = f"""You are an elite academic professor. Generate exactly ONE challenging subjective/conceptual short-answer question based on this lecture reference snippet about the topic "{topic}".
            
            Reference Snippet:
            {ref_chunk}
            
            The question must test the student's conceptual understanding of the snippet.
            Return the result ONLY as a valid JSON object. Do NOT wrap it in backticks, markdown code blocks, or include any extra text.
            The JSON object must have these exact keys:
            1. "topic": "{topic}"
            2. "question_text": "The subjective question text"
            
            Format:
            {{"topic": "{topic}", "question_text": "..."}}
            """
            try:
                response = ollama.chat(model='mistral', messages=[
                    {'role': 'user', 'content': prompt}
                ])
                content = response['message']['content'].strip()
                if content.startswith('```'):
                    content = '\n'.join(content.split('\n')[1:-1]).strip()
                import json as _json
                q_data = _json.loads(content)
                questions.append({
                    "id": i + 1,
                    "topic": q_data.get("topic", topic),
                    "question_text": q_data.get("question_text", f"Explain the core concept of {topic}.")
                })
            except Exception as qe:
                print(f"Error generating question {i} for topic {topic}: {qe}")
                questions.append({
                    "id": i + 1,
                    "topic": topic,
                    "question_text": f"Explain the core concept and practical application of {topic} in your own words."
                })

        return jsonify(questions[:num_questions])

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_subjective_test', methods=['POST'])
def evaluate_subjective_test():
    """Use Mistral to grade the typed answers grounded in retrieved lecture chunks."""
    try:
        data = request.json or {}
        answers = data.get('answers', []) # list of { topic, question_text, student_answer }

        if not answers:
            return jsonify({'error': 'No answers provided for evaluation.'}), 400

        evaluated_results = []
        for ans in answers:
            student_ans = ans.get('student_answer', '').strip()
            if not student_ans:
                evaluated_results.append({
                    'topic': ans.get('topic'),
                    'question_text': ans.get('question_text'),
                    'student_answer': '',
                    'score': 0,
                    'feedback': 'No answer was provided. Please write a detailed response to receive a grade.',
                    'model_answer': f'A standard explanation of {ans.get("topic")} should address its core principles, structure, and academic significance.'
                })
                continue

            # RAG Search: Retrieve matching context for this specific question
            print(f"[RAG Evaluation] Retrieving context for question: {ans.get('question_text')}")
            relevant = global_vector_index.query(ans.get('question_text'), top_k=2)
            context_text = "\n\n".join([rc['chunk'] for rc in relevant]) if relevant else f"Information related to {ans.get('topic')}"

            prompt = f"""You are an academic evaluator. Evaluate the student's answer to this question based on the retrieved lecture context.
            
            Lecture Context:
            {context_text}
            
            Question: {ans.get('question_text')}
            Topic: {ans.get('topic')}
            Student's Answer: {ans.get('student_answer', '')}
            
            Provide your evaluation in a JSON structure containing:
            1. "score": an integer from 0 to 10.
            2. "feedback": 2-3 sentences of feedback indicating what was correct, incorrect, and what key points they missed based on the lecture context.
            3. "model_answer": a short model answer (2-3 sentences) showing how to answer this question comprehensively using the lecture context.
            
            Return ONLY the raw JSON object, no backticks, markdown wrapping, or explanations.
            """
            try:
                response = ollama.chat(model='mistral', messages=[
                    {'role': 'user', 'content': prompt}
                ])
                content = response['message']['content'].strip()

                if content.startswith('```'):
                    content = '\n'.join(content.split('\n')[1:-1]).strip()

                import json as _json
                eval_data = _json.loads(content)
                score = int(eval_data.get('score', 5))
                feedback = eval_data.get('feedback', 'Thank you for your answer. Good attempt.')
                model_ans = eval_data.get('model_answer', 'A comprehensive answer should describe the core theory and practical implementation of the concept.')
            except Exception:
                # Fallback if parsing fails
                ans_len = len(ans.get('student_answer', '').split())
                score = 7 if ans_len > 15 else (4 if ans_len > 0 else 0)
                feedback = "Good conceptual summary. Ensure to expand on edge cases." if ans_len > 15 else "The answer is too brief. Try to explain with examples."
                model_ans = f"A standard explanation of {ans.get('topic')} should address its core principles, structure, and academic significance."

            evaluated_results.append({
                'topic': ans.get('topic'),
                'question_text': ans.get('question_text'),
                'student_answer': ans.get('student_answer'),
                'score': score,
                'feedback': feedback,
                'model_answer': model_ans
            })

        return jsonify(evaluated_results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__': 
    app.run(debug=False, port=5001, threaded=True)

