import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import pygame
import time
import threading
import queue

# Queue to hold audio chunks so we can process them in parallel with recording
audio_queue = queue.Queue()

def speak_audio(text, lang='kn'):
    """Generate and play audio for the translated text."""
    if not text:
        return
        
    audio_file = f"temp_dictation_{int(time.time()*1000)}.mp3"
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(audio_file)
        
        # Initialize and play the audio using pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait until pygame finishes playing the clip
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        pygame.mixer.music.unload()
        pygame.mixer.quit()
    except Exception as e:
        print(f"Audio playback error: {e}")
    finally:
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except:
                pass

def process_audio():
    """
    Background worker that constantly pulls audio from the queue, 
    transcribes it, and translates it WITHOUT stopping the microphone.
    """
    translator_kn = GoogleTranslator(source='en', target='kn')
    # We need a new recognizer instance for the worker thread since it handles transcription
    worker_recognizer = sr.Recognizer()
    
    while True:
        # Get pending audio recording
        audio = audio_queue.get()
        if audio is None:
            break
            
        try:
            # 1. Speech to Text
            english_text = worker_recognizer.recognize_google(audio, language="en-US")
            print(f"\n[YOU] (English): {english_text}")
            
            # 2. Text to Text
            kannada_text = translator_kn.translate(english_text)
            print(f"[TRANSLATOR] (Kannada): {kannada_text}")
            
            # 3. Text to Speech
            speak_audio(kannada_text, lang='kn')
            
        except sr.UnknownValueError:
            # Failed to recognize words, silently skip to avoid spamming the terminal
            pass
        except sr.RequestError as e:
            print(f"Network error; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        audio_queue.task_done()

def callback(recognizer, audio):
    """
    This function is triggered by speech_recognition INSTANTLY 
    whenever the microphone hears a phrase and you pause for a second.
    """
    # Send the audio to the queue to be translated immediately
    audio_queue.put(audio)
    print(" [Recording captured! Translating in parallel...]")

def live_speech_continuous():
    main_recognizer = sr.Recognizer()
    
    # Start the translation/speaking pipeline on a separate background thread
    worker_thread = threading.Thread(target=process_audio, daemon=True)
    worker_thread.start()
    
    # Define our source, but do NOT wrap the entire script in a "with" block manually
    # because listen_in_background() automatically handles its own "with" block!
    source = sr.Microphone()
    
    print("\n" + "="*50)
    print("CONTINUOUS REAL-TIME SPEECH TO KANNADA")
    print("⚠️ IMPORTANT: PLEASE WEAR HEADPHONES! ⚠️")
    print("If you use speakers, the microphone might pick up the Kannada")
    print("translation audio and try to translate it in an infinite loop!")
    print("="*50)
    
    print("\nAdjusting for background noise... Please wait.")
    
    # We only use the "with" block temporarily to adjust for ambient noise
    with source as s:
        main_recognizer.adjust_for_ambient_noise(s, duration=1)
        
    print("\nReady! Speak normally and continuously. You don't need to wait!")
    print("Press Ctrl+C in the terminal to stop at any time.\n")
    
    # listen_in_background spawns its own thread and safely manages the microphone stream!
    stop_listening = main_recognizer.listen_in_background(source, callback)
    
    try:
        # Keep the main program running to allow background threads to work
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping translation system...")
        stop_listening(wait_for_stop=False)
        print("Exiting peacefully.")

if __name__ == "__main__":
    live_speech_continuous()
