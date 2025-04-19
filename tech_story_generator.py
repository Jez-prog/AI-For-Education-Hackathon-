import requests
import random
import time
import os
import json
import re
import logging
from gtts import gTTS
import wave
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import speech_recognition as sr  # For speech-to-text functionality

# === Configuration ===
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llama3.2:1b"
BAD_WORDS = ["kill", "blood", "hate", "stupid", "die", "weapon", "murder"]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Story Elements ===
heroes = ["curious AI bot", "friendly robot", "tech-savvy kid", "digital cat", "inventor teenager"]
settings = ["futuristic city", "underground tech lab", "space station", "robot school", "virtual reality world"]
goals = ["build a supercomputer", "stop a rogue virus", "win a coding challenge", "invent a teleportation device", "solve a space mystery"]

quiz_questions = [
    {"question": "What is the main character of the story?", "choices": ["a) AI bot", "b) Robot", "c) Kid", "d) Cat"], "answer": "a"},
    {"question": "Where does the story take place?", "choices": ["a) City", "b) Lab", "c) Space", "d) School"], "answer": "b"},
    # Add 8 more questions here...
]

# === Utility Functions ===
def play_audio(filename):
    """Play an audio file using sounddevice."""
    try:
        samplerate, data = wavfile.read(filename)
        if data.dtype != np.float32:
            data = data / np.max(np.abs(data), axis=0)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        logging.error(f"Error during audio playback: {e}")

def record_audio(duration, samplerate=44100):
    """Record audio using sounddevice."""
    try:
        print("Recording...")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        print("Recording complete.")
        return audio_data
    except Exception as e:
        logging.error(f"Error during audio recording: {e}")
        return None

def save_audio(filename, audio_data, samplerate=44100):
    """Save recorded audio to a WAV file."""
    try:
        wavfile.write(filename, samplerate, (audio_data * 32767).astype(np.int16))
    except Exception as e:
        logging.error(f"Error saving audio file: {e}")

# === Main Functions ===
def build_prompt(hero, setting, goal):
    return f"Tell a safe, exciting short story for kids about a {hero} in a {setting} who wants to {goal}. Keep it imaginative and friendly for ages 6-10."

def is_safe(text):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in BAD_WORDS) + r')\b', re.IGNORECASE)
    return not pattern.search(text)

def ask_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=10)
        response.raise_for_status()
        story_parts = []

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    story_parts.append(content)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decoding error: {e}")
                    continue

        return ''.join(story_parts)

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        return None

def narrate(text, story_count):
    try:
        filename = f"story_{story_count + 1}.wav"
        tts = gTTS(text)
        tts.save("temp.mp3")

        # Convert MP3 to WAV
        os.system(f"ffmpeg -i temp.mp3 -ar 44100 -ac 1 {filename} -y")
        os.remove("temp.mp3")

        # Play the audio using sounddevice
        play_audio(filename)
    except Exception as e:
        logging.error(f"Error during narration: {e}")

def ask_question(question_data):
    question = question_data["question"]
    choices = question_data["choices"]
    correct_answer = question_data["answer"]

    tts = gTTS(f"{question}. Your choices are: {', '.join(choices)}. Please answer with a, b, c, or d.")
    tts.save("question.wav")
    play_audio("question.wav")

    attempts = 0
    while attempts < 3:
        try:
            print("Listening for your answer...")
            audio_data = record_audio(duration=5)
            save_audio("response.wav", audio_data)

            recognizer = sr.Recognizer()
            with sr.AudioFile("response.wav") as source:
                audio = recognizer.record(source)
            user_response = recognizer.recognize_google(audio).lower()

            if user_response in ["a", "b", "c", "d"]:
                if user_response == correct_answer:
                    print("Correct!")
                    tts = gTTS("Correct!")
                    tts.save("response.wav")
                    play_audio("response.wav")
                    return
                else:
                    attempts += 1
                    if attempts < 3:
                        print("Your answer is incorrect! Please try again.")
                        tts = gTTS("Your answer is incorrect! Please try again.")
                        tts.save("retry.wav")
                        play_audio("retry.wav")
                    else:
                        print(f"Incorrect! The correct answer is {correct_answer}.")
                        tts = gTTS(f"Incorrect! The correct answer is {correct_answer}.")
                        tts.save("response.wav")
                        play_audio("response.wav")
            else:
                print("Invalid response. Please answer with a, b, c, or d.")
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your response.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            break

def ask_feedback():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    tts = gTTS("What part of the story did you not understand?")
    tts.save("feedback.wav")
    play_audio("feedback.wav")

    try:
        with mic as source:
            print("Listening for your feedback...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            user_feedback = recognizer.recognize_google(audio)

            if is_safe(user_feedback):
                explanation = f"You mentioned: {user_feedback}. Let me explain in simple terms."
                print(explanation)
                tts = gTTS(explanation)
                tts.save("explanation.wav")
                play_audio("explanation.wav")
            else:
                print("Inappropriate feedback detected. Please try again.")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand your feedback.")
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")

def cleanup():
    for file in os.listdir():
        if file.endswith(".wav"):
            try:
                os.remove(file)
            except Exception as e:
                logging.warning(f"Failed to remove file {file}: {e}")

# Register cleanup function
import atexit
atexit.register(cleanup)

# === Interactive Story Loop ===
if __name__ == "__main__":
    print("🤖 Welcome! Let's enjoy a tech story together.\nPress Enter after each one to continue...\n")

    story_count = 0
    try:
        while story_count < 10:
            hero = random.choice(heroes)
            setting = random.choice(settings)
            goal = random.choice(goals)

            prompt = build_prompt(hero, setting, goal)
            story = ask_ollama(prompt)

            print(f"\n🔹 STORY {story_count + 1}: {hero} in {setting} wants to {goal}")
            if story and is_safe(story):
                print("📖", story.strip())
                narrate(story, story_count)
            else:
                logging.warning("Skipped due to unsafe content or an error.")
                time.sleep(1)
                continue

            print("\n📝 Let's take a quiz!")
            for question_data in quiz_questions:
                ask_question(question_data)

            print("\n💬 Let's discuss the story!")
            ask_feedback()

            story_count += 1
            input("\n👉 Press Enter for the next story...")

    except KeyboardInterrupt:
        print("\n👋 Exiting early. See you next time!")