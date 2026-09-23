import os
import sys
import types
import torch
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Monkey-patch transformers.onnx
onnx_module = types.ModuleType('transformers.onnx')
onnx_module.OnnxConfig = object
onnx_module.OnnxSeq2SeqConfigWithPast = object
onnx_module.__path__ = []
utils_module = types.ModuleType('transformers.onnx.utils')
utils_module.compute_effective_axis_dimension = lambda *args, **kwargs: 0
sys.modules['transformers.onnx'] = onnx_module
sys.modules['transformers.onnx.utils'] = utils_module

# Configuration
HF_TOKEN = "TOKEN"
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"
TGT_LANG = "kan_Knda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    token=HF_TOKEN,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
xliterator = UnicodeIndicTransliterator()

def translate_sentence(text):
    if not text.strip(): return ""
    tagged_text = f"{SRC_LANG} {TGT_LANG} {text.strip()}"
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
    return xliterator.transliterate(generated_text, "hi", "kn")

text_to_translate = """
this is a serious mistake a whole generations of people have done that if you study how should you study hard you must study hard if you work how should you work you must work hard why didn't they tell you you must study joyfully why didn't I tell you you must work lovingly no you must do everything hot and then you complain you complain about everything in life because you're doing everything hot very substation Medical and scientific evidence to show you only when you are in pleasant states of experience does your body and your brain work at their best is that important for you to perform any activity in life well that your body and your brains are working well hello is it important
"""

# Split into manageable chunks by common phrases/pauses since punctuation is missing
# I'll manually split this one for the best result
chunks = [
    "this is a serious mistake a whole generations of people have done that",
    "if you study how should you study hard you must study hard",
    "if you work how should you work you must work hard",
    "why didn't they tell you you must study joyfully",
    "why didn't I tell you you must work lovingly",
    "no you must do everything hot and then you complain",
    "you complain about everything in life because you're doing everything hot",
    "very substation Medical and scientific evidence to show you",
    "only when you are in pleasant states of experience does your body and your brain work at their best",
    "is that important for you to perform any activity in life well",
    "that your body and your brains are working well hello is it important"
]

print("\n--- Translating Chunks ---")
results = []
for chunk in chunks:
    print(f"Translating: {chunk[:30]}...")
    results.append(translate_sentence(chunk))

print("\n--- Final Translation Result ---")
print(" ".join(results))
print("--------------------------")
