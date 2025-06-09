import speech_recognition as sr
import pyttsx3
import requests
import json
import re
import os
import subprocess
import webbrowser
import time
import threading
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import base64
from io import BytesIO
import win32gui
import win32con

# Configuration
API_KEY = "Your ai api key"  # Your provided API key
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Different models for different purposes
CHAT_MODEL = "deepseek/deepseek-r1-zero:free"  # For regular chat
VISION_MODEL = "openai/chatgpt-4o-latest"  # For screen analysis with your API key

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://jarvis-assistant.com",
    "X-Title": "Jarvis Voice Assistant"
}

# Configure PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Configure Tesseract path (adjust this path based on your installation)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize
engine = pyttsx3.init()
engine.setProperty("rate", 180)

recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Global variables for session management
session_active = False
session_timeout = 30  # seconds
last_interaction = 0


def callback(recognizer, audio):
    try:
        command = recognizer.recognize_google(audio)
        if command.lower() == "stop":
            engine.stop()
            print("Speech stopped by user")
    except:
        pass


stop_listening = recognizer.listen_in_background(microphone, callback)


def speak(text, allow_interruption=False):
    global last_interaction
    last_interaction = time.time()

    if allow_interruption:
        stop_listening(wait_for_stop=False)
        stop_listening_new = recognizer.listen_in_background(microphone, callback)

    print(f"🗣️ Jarvis: {text}")
    engine.say(text)
    engine.runAndWait()

    if allow_interruption:
        stop_listening_new(wait_for_stop=False)


def listen(timeout=5):
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("🎤 Listening...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

        print("🔍 Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except sr.WaitTimeoutError:
        print("⏰ Listening timeout")
        return None
    except sr.UnknownValueError:
        print("❓ Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return None


def clean_response(text):
    # Remove markdown formatting and clean up text
    text = re.sub(r'\\n|\n|\r', ' ', text)
    text = re.sub(r'[*_#>\[\]{}|`]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ORIGINAL CHAT FUNCTION - RESTORED
def chat_with_deepseek(prompt):
    try:
        data = {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Jarvis, an intelligent AI assistant. Speak clearly and in a natural tone. Keep responses concise and conversational. Respond in the same language the user speaks."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,  # Limit response length for speech
            "temperature": 0.7
        }

        print("🌐 Sending request to AI...")
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data), timeout=10)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                raw_answer = result["choices"][0]["message"]["content"]
                print(f"🧠 AI Raw Response: {raw_answer}")
                return clean_response(raw_answer)
            else:
                print("❌ No choices in API response")
                return "I received an empty response from the AI."
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return "Sorry, I couldn't get a response from the AI service."

    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return "The AI service is taking too long to respond."
    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return "I can't connect to the AI service right now."
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return "There was a problem connecting to the AI."


# NEW SCREEN ANALYSIS FUNCTIONS
def capture_screen():
    """Capture the entire screen"""
    try:
        screenshot = pyautogui.screenshot()
        return screenshot
    except Exception as e:
        print(f"❌ Error capturing screen: {e}")
        return None


def capture_area_around_cursor(radius=200):
    """Capture area around cursor position"""
    try:
        x, y = pyautogui.position()

        # Calculate capture area
        left = max(0, x - radius)
        top = max(0, y - radius)
        width = radius * 2
        height = radius * 2

        # Adjust if going beyond screen boundaries
        screen_width, screen_height = pyautogui.size()
        if left + width > screen_width:
            width = screen_width - left
        if top + height > screen_height:
            height = screen_height - top

        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        # Draw cursor position indicator
        draw = ImageDraw.Draw(screenshot)
        cursor_x = x - left
        cursor_y = y - top

        # Draw crosshair at cursor position
        draw.line([(cursor_x - 10, cursor_y), (cursor_x + 10, cursor_y)], fill='red', width=3)
        draw.line([(cursor_x, cursor_y - 10), (cursor_x, cursor_y + 10)], fill='red', width=3)
        draw.ellipse([(cursor_x - 5, cursor_y - 5), (cursor_x + 5, cursor_y + 5)], outline='red', width=2)

        return screenshot, (x, y)
    except Exception as e:
        print(f"❌ Error capturing cursor area: {e}")
        return None, None


def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Use Tesseract to extract text
        text = pytesseract.image_to_string(opencv_image)
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return ""


def get_active_window_info():
    """Get information about the active window"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)
        return window_title
    except Exception as e:
        print(f"❌ Error getting window info: {e}")
        return "Unknown Application"


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    try:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"❌ Error converting image to base64: {e}")
        return None


def analyze_screen_with_chatgpt(image, question, context_text=""):
    """Send screen image to ChatGPT for analysis using your API key"""
    try:
        base64_image = image_to_base64(image)
        if not base64_image:
            return "Sorry, I couldn't process the screen image."

        # Prepare the message with image using ChatGPT 4o
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are Jarvis, an AI assistant analyzing screen content. Question: {question}\n\nScreen text context: {context_text}\n\nPlease analyze the image and provide a clear, concise answer."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        data = {
            "model": VISION_MODEL,  # Using ChatGPT 4o latest
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7
        }

        print("🔍 Analyzing screen with ChatGPT Vision...")
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data), timeout=15)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                answer = result["choices"][0]["message"]["content"]
                return clean_response(answer)
            else:
                return "I couldn't analyze the screen content."
        else:
            print(f"❌ Vision API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return "Sorry, I couldn't analyze the screen right now."

    except Exception as e:
        print(f"❌ Error analyzing screen: {e}")
        return "There was an error analyzing the screen."


def search_google_with_context(query, screen_text=""):
    """Search Google with additional context from screen"""
    search_query = query
    if screen_text:
        # Add relevant screen text to search query
        words = screen_text.split()[:10]  # First 10 words as context
        context = " ".join(words)
        search_query = f"{query} {context}"

    speak(f"Searching Google for {query}")
    url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
    webbrowser.open(url)


def handle_screen_question(question):
    """Handle questions about screen content"""
    try:
        # Determine if user wants full screen or cursor area
        if any(word in question.lower() for word in ["cursor", "pointing", "here", "this"]):
            print("📸 Capturing area around cursor...")
            image, cursor_pos = capture_area_around_cursor()
            if not image:
                speak("Sorry, I couldn't capture the screen area.")
                return
            context_type = f"area around cursor at position {cursor_pos}"
        else:
            print("📸 Capturing full screen...")
            image = capture_screen()
            if not image:
                speak("Sorry, I couldn't capture the screen.")
                return
            context_type = "full screen"

        # Extract text from image
        print("🔤 Extracting text from screen...")
        screen_text = extract_text_from_image(image)

        # Get active window info
        window_title = get_active_window_info()

        print(f"📱 Active window: {window_title}")
        print(f"📝 Extracted text preview: {screen_text[:100]}...")

        # Try to answer with ChatGPT vision first
        print("🤖 Analyzing with ChatGPT Vision...")
        ai_response = analyze_screen_with_chatgpt(image, question, screen_text)

        if ai_response and "sorry" not in ai_response.lower() and "couldn't" not in ai_response.lower():
            speak(ai_response)
        else:
            # Fallback to text-based analysis with DeepSeek
            if screen_text:
                fallback_prompt = f"Based on this screen content: '{screen_text[:500]}' and the active window '{window_title}', please answer: {question}"
                fallback_response = chat_with_deepseek(fallback_prompt)

                if fallback_response:
                    speak(fallback_response)
                else:
                    # Final fallback - search Google
                    speak("Let me search for that information online.")
                    search_google_with_context(question, screen_text)
            else:
                speak("I couldn't extract any text from the screen. Let me search online for you.")
                search_google_with_context(question)

    except Exception as e:
        print(f"❌ Error handling screen question: {e}")
        speak("Sorry, I had trouble analyzing the screen.")


def check_session_timeout():
    """Check if the session should timeout"""
    global session_active, last_interaction
    while True:
        if session_active and time.time() - last_interaction > session_timeout:
            session_active = False
            print("💤 Session timed out. Say 'Jarvis' to reactivate.")
        time.sleep(5)


# 💻 ORIGINAL CONTROL FUNCTIONS - ALL RESTORED
def shutdown():
    speak("Shutting down the system.")
    os.system("shutdown /s /t 1")


def open_chrome():
    speak("Opening Chrome.")
    try:
        chrome_paths = [
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
        ]

        for path in chrome_paths:
            if os.path.exists(path):
                subprocess.Popen([path])
                return

        # If Chrome not found, try opening default browser
        webbrowser.open("https://www.google.com")
    except Exception as e:
        speak("Sorry, I couldn't open Chrome.")
        print(f"Error opening Chrome: {e}")


def search_google(query):
    speak(f"Searching Google for {query}")
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    webbrowser.open(url)


def open_folder(folder_name):
    speak(f"Opening folder {folder_name}")

    # Common folder locations
    user_profile = os.path.expanduser("~")
    common_folders = {
        "desktop": os.path.join(user_profile, "Desktop"),
        "documents": os.path.join(user_profile, "Documents"),
        "downloads": os.path.join(user_profile, "Downloads"),
        "pictures": os.path.join(user_profile, "Pictures"),
        "music": os.path.join(user_profile, "Music"),
        "videos": os.path.join(user_profile, "Videos")
    }

    folder_name_lower = folder_name.lower()

    if folder_name_lower in common_folders:
        folder_path = common_folders[folder_name_lower]
    else:
        folder_path = os.path.join(user_profile, folder_name)

    if os.path.exists(folder_path):
        os.startfile(folder_path)
    else:
        speak(f"Sorry, I can't find the {folder_name} folder.")


def process_command(command):
    """Process user commands with ALL original features + screen analysis"""
    global session_active

    command_lower = command.lower()

    # NEW: Check for screen-related questions FIRST
    screen_keywords = [
        "what's on", "what is on", "what's this", "what is this",
        "what's that", "what is that", "screen", "display",
        "see", "look", "show", "cursor", "pointing", "here"
    ]

    question_words = ["what", "how", "why", "where", "when", "who", "which"]

    # Check if this is a screen-related question
    is_screen_question = (
            any(keyword in command_lower for keyword in screen_keywords) or
            (any(qword in command_lower for qword in question_words) and
             any(keyword in command_lower for keyword in ["screen", "this", "that", "here", "cursor"]))
    )

    if is_screen_question:
        handle_screen_question(command)
        return "continue"

    # ORIGINAL COMMANDS - ALL RESTORED
    if any(word in command_lower for word in ["exit", "quit", "stop", "bye"]):
        speak("Goodbye! Have a great day.", allow_interruption=False)
        return "exit"

    elif "shutdown" in command_lower:
        shutdown()
        return "exit"

    elif "open chrome" in command_lower:
        open_chrome()

    elif "search for" in command_lower or "google" in command_lower:
        search_query = command_lower.replace("search for", "").replace("google", "").strip()
        if search_query:
            search_google(search_query)
        else:
            speak("What should I search for?")

    elif "open folder" in command_lower:
        folder = command_lower.replace("open folder", "").strip()
        open_folder(folder)

    # NEW: Screen capture commands
    elif "take screenshot" in command_lower or "capture screen" in command_lower:
        screenshot = capture_screen()
        if screenshot:
            screenshot.save("screenshot.png")
            speak("Screenshot saved as screenshot.png")
        else:
            speak("Sorry, I couldn't take a screenshot.")

    # Session control
    elif any(word in command_lower for word in ["sleep", "deactivate", "rest"]):
        session_active = False
        speak("Going to sleep. Say Jarvis to wake me up.")
        return "continue"

    else:
        # ORIGINAL: Send to DeepSeek for regular chat
        response = chat_with_deepseek(command)
        if response:
            speak(response, allow_interruption=True)
        else:
            speak("I couldn't understand the response.", allow_interruption=False)

    return "continue"


# 🔁 ORIGINAL MAIN LOOP - RESTORED
if __name__ == "__main__":
    print("🤖 Complete Enhanced Jarvis Assistant Starting...")
    print("📋 New Features: Screen analysis, OCR, ChatGPT Vision")
    print("📋 Required packages: pip install pyautogui opencv-python pillow pytesseract pywin32")
    print("📋 Also install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")

    # Start session timeout checker in background
    timeout_thread = threading.Thread(target=check_session_timeout, daemon=True)
    timeout_thread.start()

    speak(
        "Hello, I am Jarvis. I can now see your screen and answer questions about what you're looking at. Say 'Jarvis' to activate me.",
        allow_interruption=False)

    while True:
        try:
            if not session_active:
                print("🕒 Waiting for wake word 'Jarvis'...")
                wake_input = listen(timeout=10)

                if wake_input and "jarvis" in wake_input.lower():
                    session_active = True
                    last_interaction = time.time()
                    speak("Yes? What would you like me to do?", allow_interruption=False)
                    continue
                else:
                    continue

            else:
                # Session is active, listen for commands
                print("🎯 Session active - listening for commands...")
                command = listen(timeout=8)

                if command:
                    last_interaction = time.time()
                    result = process_command(command)

                    if result == "exit":
                        break
                    elif result == "continue":
                        continue
                else:
                    # No command received, but session is still active
                    print("⏳ No command received, session still active...")

        except KeyboardInterrupt:
            print("\n👋 Exiting Jarvis...")
            speak("Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            speak("Sorry, something went wrong.")
            continue

    print("🔚 Jarvis Assistant stopped.")
