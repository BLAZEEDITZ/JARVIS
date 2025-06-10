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
import sys
import logging
import configparser
from datetime import datetime
from pathlib import Path

import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

# Set up logging - FIXED: Removed emoji characters to avoid encoding issues
log_dir = os.path.join(os.path.expanduser("~"), "JarvisLogs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"jarvis_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # FIXED: Added UTF-8 encoding
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Jarvis")

# Try to import screen analysis libraries (optional)
try:
    import pyautogui
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import pytesseract
    import win32gui
    import win32con
    import win32api
    import win32com.client

    SCREEN_ANALYSIS_AVAILABLE = True
    logger.info("[OK] Screen analysis libraries loaded successfully")  # FIXED: Removed emoji
except ImportError as e:
    SCREEN_ANALYSIS_AVAILABLE = False
    logger.warning(f"[WARNING] Screen analysis not available: {e}")
    logger.info(
        "[INFO] To enable screen analysis, install: pip install pyautogui opencv-python pillow pytesseract pywin32")

# Configuration file setup
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "jarvis_config.ini")


def load_config():
    """Load configuration from file or create default"""
    config = configparser.ConfigParser()

    if os.path.exists(CONFIG_FILE):
        try:
            config.read(CONFIG_FILE)
            logger.info(f"Configuration loaded from {CONFIG_FILE}")

            # FIXED: Check if required sections and keys exist
            if not config.has_section('API'):
                logger.warning("API section missing in config, creating default")
                config.add_section('API')

            # Check for required API keys
            required_api_keys = ['deepseek_key', 'base_url', 'primary_model', 'fallback_models', 'site_url',
                                 'site_name', 'temperature', 'max_tokens']
            for key in required_api_keys:
                if not config.has_option('API', key):
                    logger.warning(f"Missing config key: API.{key}, adding default")
                    if key == 'deepseek_key':
                        config['API'][key] = 'your api key'
                    elif key == 'base_url':
                        config['API'][key] = 'https://openrouter.ai/api/v1/chat/completions'
                    elif key == 'primary_model':
                        config['API'][key] = 'deepseek/deepseek-chat:free'
                    elif key == 'fallback_models':
                        config['API'][key] = 'deepseek/deepseek-coder:free,meta-llama/llama-3.1-8b-instruct:free'
                    elif key == 'site_url':
                        config['API'][key] = 'https://jarvis-assistant.local'
                    elif key == 'site_name':
                        config['API'][key] = 'Jarvis Voice Assistant'
                    elif key == 'temperature':
                        config['API'][key] = '0.7'
                    elif key == 'max_tokens':
                        config['API'][key] = '2048'

            # Check for SETTINGS section
            if not config.has_section('SETTINGS'):
                logger.warning("SETTINGS section missing in config, creating default")
                config.add_section('SETTINGS')

            # Check for required settings
            required_settings = ['session_timeout', 'max_retries', 'retry_delay', 'voice_rate', 'search_depth']
            for key in required_settings:
                if not config.has_option('SETTINGS', key):
                    logger.warning(f"Missing config key: SETTINGS.{key}, adding default")
                    if key == 'session_timeout':
                        config['SETTINGS'][key] = '30'
                    elif key == 'max_retries':
                        config['SETTINGS'][key] = '3'
                    elif key == 'retry_delay':
                        config['SETTINGS'][key] = '2'
                    elif key == 'voice_rate':
                        config['SETTINGS'][key] = '180'
                    elif key == 'search_depth':
                        config['SETTINGS'][key] = '2'

            # Save any changes made
            save_config(config)

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            create_default_config(config)
    else:
        create_default_config(config)

    return config


def create_default_config(config):
    """Create default configuration"""
    # FIXED: Make sure sections exist
    if not config.has_section('API'):
        config.add_section('API')
    if not config.has_section('SETTINGS'):
        config.add_section('SETTINGS')

    config['API'] = {
        'deepseek_key': 'your api key',
        'base_url': 'https://openrouter.ai/api/v1/chat/completions',
        'primary_model': 'deepseek/deepseek-chat:free',
        'fallback_models': 'deepseek/deepseek-coder:free,meta-llama/llama-3.1-8b-instruct:free',
        'site_url': 'https://jarvis-assistant.local',
        'site_name': 'Jarvis Voice Assistant',
        'temperature': '0.7',
        'max_tokens': '2048'
    }

    config['SETTINGS'] = {
        'session_timeout': '30',
        'max_retries': '3',
        'retry_delay': '2',
        'voice_rate': '180',
        'search_depth': '2'
    }

    save_config(config)
    logger.info("Default configuration created")


class AppLauncherConfig:
    """Class to handle application launcher configuration"""

    def __init__(self, config):
        self.config = config
        self.ensure_launcher_section()

    def ensure_launcher_section(self):
        """Ensure launcher section exists in config"""
        if not self.config.has_section('LAUNCHER'):
            self.config.add_section('LAUNCHER')
            # Default launcher settings
            self.config['LAUNCHER'] = {
                'enabled': 'true',
                'scan_interval': '5',
                'max_retries': '3',
                'retry_delay': '2'
            }
            save_config(self.config)
            logger.info("Added LAUNCHER section to configuration")

        if not self.config.has_section('LAUNCHER_APPS'):
            self.config.add_section('LAUNCHER_APPS')
            # Example app configuration
            self.config['LAUNCHER_APPS']['example'] = json.dumps({
                'executable': 'example.exe',
                'directories': [
                    os.path.join(os.path.expanduser("~"), "Downloads"),
                    "C:\\Program Files\\Example"
                ],
                'launch_params': '--start',
                'enabled': 'true'
            })
            save_config(self.config)
            logger.info("Added LAUNCHER_APPS section to configuration")

    def get_app_configs(self):
        """Get all app configurations"""
        app_configs = {}

        if self.config.has_section('LAUNCHER_APPS'):
            for app_name, app_config_str in self.config.items('LAUNCHER_APPS'):
                try:
                    app_config = json.loads(app_config_str)
                    app_configs[app_name] = app_config
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in config for app {app_name}")

        return app_configs

    def add_app_config(self, app_name, executable, directories, launch_params="", enabled=True):
        """Add or update app configuration"""
        if not self.config.has_section('LAUNCHER_APPS'):
            self.config.add_section('LAUNCHER_APPS')

        app_config = {
            'executable': executable,
            'directories': directories,
            'launch_params': launch_params,
            'enabled': str(enabled).lower()
        }

        self.config['LAUNCHER_APPS'][app_name] = json.dumps(app_config)
        save_config(self.config)
        logger.info(f"Added/updated launcher configuration for app: {app_name}")

    def remove_app_config(self, app_name):
        """Remove app configuration"""
        if self.config.has_section('LAUNCHER_APPS') and app_name in self.config['LAUNCHER_APPS']:
            self.config.remove_option('LAUNCHER_APPS', app_name)
            save_config(self.config)
            logger.info(f"Removed launcher configuration for app: {app_name}")
            return True

        logger.warning(f"App {app_name} not found in launcher configuration")
        return False

    def is_enabled(self):
        """Check if launcher is enabled"""
        if self.config.has_section('LAUNCHER'):
            return self.config['LAUNCHER'].get('enabled', 'true').lower() == 'true'
        return False


class AppLauncherEventHandler(FileSystemEventHandler):
    """Event handler for file system events"""

    def __init__(self, app_launcher):
        self.app_launcher = app_launcher
        super().__init__()

    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self.app_launcher.check_and_launch_app(event.src_path)

    def on_moved(self, event):
        """Handle file move events"""
        if not event.is_directory:
            self.app_launcher.check_and_launch_app(event.dest_path)


class AppLauncher:
    """Application launcher class integrated with Jarvis"""

    def __init__(self, config):
        self.config = config
        self.launcher_config = AppLauncherConfig(config)
        self.observers = []
        self.running = False
        self.launched_apps = set()

    def start(self):
        """Start monitoring directories"""
        if not self.launcher_config.is_enabled():
            logger.info("App launcher is disabled in configuration")
            return False

        if self.running:
            logger.warning("App launcher is already running")
            return False

        self.running = True
        self.setup_observers()

        # Initial scan of directories
        self.scan_directories()

        logger.info("App launcher started")
        return True

    def stop(self):
        """Stop monitoring directories"""
        self.running = False

        for observer in self.observers:
            observer.stop()

        for observer in self.observers:
            observer.join()

        self.observers = []
        logger.info("App launcher stopped")

    def setup_observers(self):
        """Set up file system observers for all directories"""
        # Stop existing observers
        for observer in self.observers:
            observer.stop()
            observer.join()

        self.observers = []

        # Get all unique directories from app configs
        directories = set()
        for app_name, app_config in self.launcher_config.get_app_configs().items():
            if app_config.get('enabled', 'true').lower() == 'true':
                for directory in app_config.get('directories', []):
                    if os.path.exists(directory) and os.path.isdir(directory):
                        directories.add(directory)
                    else:
                        logger.warning(f"Directory not found: {directory} for app {app_name}")

        # Set up observers for each directory
        if directories:
            event_handler = AppLauncherEventHandler(self)
            for directory in directories:
                try:
                    observer = Observer()
                    observer.schedule(event_handler, directory, recursive=False)
                    observer.start()
                    self.observers.append(observer)
                    logger.info(f"Monitoring directory: {directory}")
                except Exception as e:
                    logger.error(f"Error setting up observer for directory {directory}: {e}")

    def scan_directories(self):
        """Scan all directories for executable files"""
        app_configs = self.launcher_config.get_app_configs()

        for app_name, app_config in app_configs.items():
            if app_config.get('enabled', 'true').lower() != 'true':
                continue

            executable = app_config.get('executable')
            if not executable:
                logger.warning(f"No executable specified for app {app_name}")
                continue

            directories = app_config.get('directories', [])
            for directory in directories:
                if not os.path.exists(directory) or not os.path.isdir(directory):
                    continue

                exe_path = os.path.join(directory, executable)
                if os.path.exists(exe_path) and os.path.isfile(exe_path):
                    self.launch_app(app_name, exe_path, app_config.get('launch_params', ''))
                    break

    def check_and_launch_app(self, file_path):
        """Check if the file is a monitored executable and launch if it is"""
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return

        file_name = os.path.basename(file_path)
        directory = os.path.dirname(file_path)

        app_configs = self.launcher_config.get_app_configs()
        for app_name, app_config in app_configs.items():
            if app_config.get('enabled', 'true').lower() != 'true':
                continue

            executable = app_config.get('executable')
            if not executable or executable != file_name:
                continue

            directories = app_config.get('directories', [])
            if directory in directories:
                self.launch_app(app_name, file_path, app_config.get('launch_params', ''))
                break

    def launch_app(self, app_name, exe_path, launch_params):
        """Launch the application"""
        # Check if app was already launched
        if app_name in self.launched_apps:
            logger.info(f"App {app_name} was already launched, skipping")
            return False

        logger.info(f"Launching app: {app_name} from {exe_path}")

        max_retries = int(self.config.get('LAUNCHER', 'max_retries', fallback='3'))
        retry_delay = int(self.config.get('LAUNCHER', 'retry_delay', fallback='2'))

        cmd = f'"{exe_path}" {launch_params}'.strip()

        for attempt in range(max_retries):
            try:
                process = subprocess.Popen(cmd, shell=True)
                logger.info(f"App {app_name} launched successfully (PID: {process.pid})")
                self.launched_apps.add(app_name)

                # Announce the launch via TTS if available
                if TTS_AVAILABLE:
                    speak(f"Launched {app_name}")

                return True
            except Exception as e:
                logger.error(f"Error launching app {app_name} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        logger.error(f"Failed to launch app {app_name} after {max_retries} attempts")
        if TTS_AVAILABLE:
            speak(f"Failed to launch {app_name}")
        return False

    def reset_launched_apps(self):
        """Reset the list of launched apps"""
        self.launched_apps = set()
        logger.info("Reset launched apps list")

    def get_status(self):
        """Get launcher status"""
        return {
            'running': self.running,
            'enabled': self.launcher_config.is_enabled(),
            'monitored_apps': len(self.launcher_config.get_app_configs()),
            'launched_apps': len(self.launched_apps),
            'observers': len(self.observers)
        }


def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")


# Load configuration
config = load_config()

# API Configuration - FIXED: Added error handling for missing keys
try:
    DEEPSEEK_API_KEY = config['API']['deepseek_key']
    BASE_URL = config['API']['base_url']
    PRIMARY_MODEL = config['API']['primary_model']
    FALLBACK_MODELS = config['API']['fallback_models'].split(',')
    SITE_URL = config['API']['site_url']
    SITE_NAME = config['API']['site_name']
    TEMPERATURE = float(config['API']['temperature'])
    MAX_TOKENS = int(config['API']['max_tokens'])
except (KeyError, ValueError) as e:
    logger.error(f"Error loading configuration: {e}")
    logger.info("Recreating configuration with defaults")
    create_default_config(config)
    # Try again after recreating
    DEEPSEEK_API_KEY = config['API']['deepseek_key']
    BASE_URL = config['API']['base_url']
    PRIMARY_MODEL = config['API']['primary_model']
    FALLBACK_MODELS = config['API']['fallback_models'].split(',')
    SITE_URL = config['API']['site_url']
    SITE_NAME = config['API']['site_name']
    TEMPERATURE = float(config['API']['temperature'])
    MAX_TOKENS = int(config['API']['max_tokens'])

# All available models
AVAILABLE_MODELS = [PRIMARY_MODEL] + FALLBACK_MODELS

# Settings - FIXED: Added error handling for missing keys
try:
    SESSION_TIMEOUT = int(config['SETTINGS']['session_timeout'])
    MAX_RETRIES = int(config['SETTINGS']['max_retries'])
    RETRY_DELAY = int(config['SETTINGS']['retry_delay'])
    VOICE_RATE = int(config['SETTINGS']['voice_rate'])
    SEARCH_DEPTH = int(config['SETTINGS']['search_depth'])
except (KeyError, ValueError) as e:
    logger.error(f"Error loading settings: {e}")
    logger.info("Using default settings")
    SESSION_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    VOICE_RATE = 180
    SEARCH_DEPTH = 2

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": SITE_URL,
    "X-Title": SITE_NAME
}

# Configure screen analysis if available
if SCREEN_ANALYSIS_AVAILABLE:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1

    # Try to find Tesseract installation
    TESSERACT_PATHS = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
    ]

    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"[OK] Found Tesseract at: {path}")  # FIXED: Removed emoji
            break
    else:
        logger.warning("[WARNING] Tesseract not found. OCR features will be limited.")  # FIXED: Removed emoji

# Initialize TTS
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", VOICE_RATE)
    TTS_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    logger.error(f"[ERROR] Text-to-speech not available: {e}")  # FIXED: Removed emoji

# Initialize Speech Recognition
try:
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    SPEECH_RECOGNITION_AVAILABLE = True
except Exception as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.error(f"[ERROR] Speech recognition not available: {e}")  # FIXED: Removed emoji

# Global variables for session management
session_active = False
last_interaction = 0
current_model = AVAILABLE_MODELS[0]
api_health = {
    "status": "unknown",
    "last_check": 0,
    "working_models": [],
    "failed_models": [],
    "error_details": None
}

# Initialize app launcher
app_launcher = None


class APIError(Exception):
    """Custom exception for API errors with detailed information"""

    def __init__(self, status_code, message, model=None, details=None):
        self.status_code = status_code
        self.message = message
        self.model = model
        self.details = details
        super().__init__(self.message)


def test_deepseek_connection(force=False):
    """Test DeepSeek API connection with detailed diagnostics"""
    global api_health, current_model

    # Skip if tested recently (within last 10 minutes) unless forced
    if not force and time.time() - api_health["last_check"] < 600:
        return api_health["status"] == "healthy"

    logger.info("[TESTING] Testing DeepSeek API connection...")  # FIXED: Removed emoji
    api_health["last_check"] = time.time()
    api_health["working_models"] = []
    api_health["failed_models"] = []
    api_health["error_details"] = None

    connection_test_passed = False

    # First, test basic internet connectivity
    try:
        internet_test = requests.get("https://www.google.com", timeout=5)
        if internet_test.status_code == 200:
            logger.info("[OK] Internet connection is working")  # FIXED: Removed emoji
            connection_test_passed = True
        else:
            logger.error(
                f"[ERROR] Internet connection test failed: {internet_test.status_code}")  # FIXED: Removed emoji
    except Exception as e:
        logger.error(f"[ERROR] Internet connection test failed: {e}")  # FIXED: Removed emoji

    if not connection_test_passed:
        api_health["status"] = "no_internet"
        return False

    # Test each model
    for model in AVAILABLE_MODELS:
        try:
            logger.info(f"Testing model: {model}")

            test_data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Hello, can you hear me? Please respond with just 'Yes, I can hear you.'"
                    }
                ],
                "max_tokens": 20,
                "temperature": 0.1
            }

            response = requests.post(BASE_URL, headers=HEADERS, data=json.dumps(test_data), timeout=15)

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    response_text = result['choices'][0]['message']['content']
                    if response_text and len(response_text.strip()) > 0:
                        api_health["working_models"].append(model)
                        logger.info(f"[OK] Model {model} is working")  # FIXED: Removed emoji
                    else:
                        api_health["failed_models"].append((model, "Empty response"))
                        logger.warning(f"[WARNING] Model {model} returned empty response")  # FIXED: Removed emoji
                else:
                    api_health["failed_models"].append((model, "No choices in response"))
                    logger.warning(f"[WARNING] Model {model} returned no choices")  # FIXED: Removed emoji
            else:
                error_details = f"Status: {response.status_code}"
                try:
                    error_json = response.json()
                    if 'error' in error_json:
                        error_details += f", Message: {error_json['error'].get('message', 'Unknown')}"
                except:
                    error_details += f", Raw: {response.text[:100]}"

                api_health["failed_models"].append((model, error_details))
                logger.error(f"[ERROR] Model {model} failed: {error_details}")  # FIXED: Removed emoji

        except Exception as e:
            error_msg = str(e)
            api_health["failed_models"].append((model, error_msg))
            logger.error(f"[ERROR] Error testing {model}: {error_msg}")  # FIXED: Removed emoji
            if api_health["error_details"] is None:
                api_health["error_details"] = error_msg

    # Update API health status
    if len(api_health["working_models"]) > 0:
        api_health["status"] = "healthy"
        current_model = api_health["working_models"][0]  # Use first working model
        logger.info(
            f"[OK] DeepSeek API connection successful with {len(api_health['working_models'])} models")  # FIXED: Removed emoji
        return True
    else:
        api_health["status"] = "unhealthy"
        logger.error("[ERROR] All DeepSeek models failed")  # FIXED: Removed emoji
        return False


def get_api_diagnostics():
    """Get detailed API diagnostics information"""
    diagnostics = []

    # Check API key
    if not DEEPSEEK_API_KEY or len(DEEPSEEK_API_KEY) < 20:
        diagnostics.append("DeepSeek API key is missing or invalid")
    else:
        diagnostics.append(f"DeepSeek API key looks valid (starts with {DEEPSEEK_API_KEY[:10]}...)")

    # Check internet connection
    try:
        requests.get("https://www.google.com", timeout=3)
        diagnostics.append("Internet connection is working")
    except:
        diagnostics.append("Internet connection is not working")

    # Check base URL
    diagnostics.append(f"Base URL: {BASE_URL}")

    # Check working models
    if api_health["working_models"]:
        diagnostics.append(f"Working models: {', '.join(api_health['working_models'])}")
    else:
        diagnostics.append("No working models found")

    # Check failed models
    if api_health["failed_models"]:
        for model, reason in api_health["failed_models"]:
            diagnostics.append(f"Model {model} failed: {reason}")

    # Check error details
    if api_health["error_details"]:
        diagnostics.append(f"Error details: {api_health['error_details']}")

    return diagnostics


def callback(recognizer, audio):
    try:
        command = recognizer.recognize_google(audio)
        if command.lower() == "stop":
            if TTS_AVAILABLE:
                engine.stop()
            logger.info("Speech stopped by user")
    except:
        pass


# Initialize background listening if available
if SPEECH_RECOGNITION_AVAILABLE:
    try:
        stop_listening = recognizer.listen_in_background(microphone, callback)
    except Exception as e:
        stop_listening = None
        logger.error(f"[ERROR] Background listening not available: {e}")  # FIXED: Removed emoji


def speak(text, allow_interruption=False):
    global last_interaction
    last_interaction = time.time()

    logger.info(f"[SPEAK] Jarvis: {text}")  # FIXED: Removed emoji

    if TTS_AVAILABLE:
        try:
            if allow_interruption and stop_listening:
                stop_listening(wait_for_stop=False)
                stop_listening_new = recognizer.listen_in_background(microphone, callback)

            engine.say(text)
            engine.runAndWait()

            if allow_interruption and stop_listening:
                stop_listening_new(wait_for_stop=False)
        except Exception as e:
            logger.error(f"[ERROR] TTS error: {e}")  # FIXED: Removed emoji
    else:
        print(f"Jarvis: {text}")


def listen(timeout=5):
    if not SPEECH_RECOGNITION_AVAILABLE:
        # Fallback to text input
        try:
            return input("Type your command: ").strip()
        except:
            return None

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info("[LISTENING] Listening...")  # FIXED: Removed emoji
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

        logger.info("[PROCESSING] Recognizing...")  # FIXED: Removed emoji
        query = recognizer.recognize_google(audio)
        logger.info(f"You said: {query}")
        return query
    except sr.WaitTimeoutError:
        logger.info("[TIMEOUT] Listening timeout")  # FIXED: Removed emoji
        return None
    except sr.UnknownValueError:
        logger.info("[UNKNOWN] Could not understand audio")  # FIXED: Removed emoji
        return None
    except sr.RequestError as e:
        logger.error(f"[ERROR] Speech recognition error: {e}")  # FIXED: Removed emoji
        return None


def clean_response(text):
    text = re.sub(r'\\n|\n|\r', ' ', text)
    text = re.sub(r'[*_#>\[\]{}|`]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def chat_with_deepseek(prompt, retries=0):
    """Enhanced chat function with robust error handling and automatic retries"""
    global current_model

    if retries >= MAX_RETRIES:
        logger.error(f"[ERROR] Maximum retries ({MAX_RETRIES}) reached")  # FIXED: Removed emoji
        return "I'm having trouble connecting to the DeepSeek API after multiple attempts. Please check your internet connection or try again later."

    try:
        logger.info(f"[API] Sending request to DeepSeek using model: {current_model}")  # FIXED: Removed emoji

        data = {
            "model": current_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Jarvis, an intelligent AI assistant. Speak clearly and naturally. Keep responses concise and conversational."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }

        response = requests.post(BASE_URL, headers=HEADERS, data=json.dumps(data), timeout=20)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                response_text = result['choices'][0]['message']['content']
                if response_text and len(response_text.strip()) > 0:
                    logger.info(
                        f"[RESPONSE] DeepSeek Response received ({len(response_text)} chars)")  # FIXED: Removed emoji
                    return clean_response(response_text)
                else:
                    logger.warning("[WARNING] Empty response from DeepSeek")  # FIXED: Removed emoji
                    raise APIError(200, "Empty response", current_model)
            else:
                logger.warning("[WARNING] No choices in DeepSeek response")  # FIXED: Removed emoji
                raise APIError(200, "No choices in response", current_model)

        elif response.status_code == 401:
            logger.error(f"[ERROR] Authentication failed (401)")  # FIXED: Removed emoji
            error_message = "Authentication failed"
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_message = error_json['error'].get('message', 'Authentication failed')
            except:
                pass

            # Print detailed debug info for auth errors
            logger.error(f"API Key: {DEEPSEEK_API_KEY[:10]}...")
            logger.error(f"Headers: {json.dumps(HEADERS, indent=2)}")

            raise APIError(401, error_message, current_model)

        elif response.status_code == 429:
            logger.warning(f"[WARNING] Rate limit exceeded (429)")  # FIXED: Removed emoji

            # Try next available model
            if current_model in api_health["working_models"]:
                current_index = api_health["working_models"].index(current_model)
                if current_index < len(api_health["working_models"]) - 1:
                    current_model = api_health["working_models"][current_index + 1]
                    logger.info(f"Switching to model: {current_model}")
                    return chat_with_deepseek(prompt, retries)  # Don't increment retries for model switch

            # If no other model available or already using last model, wait and retry
            logger.info(f"Waiting {RETRY_DELAY * (retries + 1)}s before retry {retries + 1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY * (retries + 1))
            return chat_with_deepseek(prompt, retries + 1)

        else:
            error_message = f"API error: {response.status_code}"
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_message = error_json['error'].get('message', error_message)
            except:
                error_message += f" - {response.text[:100]}"

            logger.error(f"[ERROR] API Error: {error_message}")  # FIXED: Removed emoji
            raise APIError(response.status_code, error_message, current_model)

    except requests.exceptions.Timeout:
        logger.error("[ERROR] Request timeout")  # FIXED: Removed emoji
        time.sleep(RETRY_DELAY * (retries + 1))
        return chat_with_deepseek(prompt, retries + 1)

    except requests.exceptions.ConnectionError:
        logger.error("[ERROR] Connection error")  # FIXED: Removed emoji
        time.sleep(RETRY_DELAY * (retries + 1))
        return chat_with_deepseek(prompt, retries + 1)

    except APIError as e:
        # For authentication errors, don't retry
        if e.status_code == 401:
            return f"I'm having trouble with my DeepSeek API authentication. Please check your API key. Error: {e.message}"

        # For other API errors, retry with different model
        if retries < MAX_RETRIES - 1:
            # Try to switch models
            if current_model in AVAILABLE_MODELS:
                current_index = AVAILABLE_MODELS.index(current_model)
                if current_index < len(AVAILABLE_MODELS) - 1:
                    current_model = AVAILABLE_MODELS[current_index + 1]
                    logger.info(f"Switching to model: {current_model} after API error")
                    time.sleep(RETRY_DELAY * (retries + 1))
                    return chat_with_deepseek(prompt, retries + 1)

        return f"I encountered an error with the DeepSeek API. Error code: {e.status_code}. Please try again later."

    except Exception as e:
        logger.error(f"[ERROR] Exception: {str(e)}")  # FIXED: Removed emoji
        time.sleep(RETRY_DELAY * (retries + 1))
        return chat_with_deepseek(prompt, retries + 1)


# Screen analysis functions (only if libraries available)
if SCREEN_ANALYSIS_AVAILABLE:
    def capture_screen():
        try:
            screenshot = pyautogui.screenshot()
            return screenshot
        except Exception as e:
            logger.error(f"[ERROR] Error capturing screen: {e}")  # FIXED: Removed emoji
            return None


    def capture_area_around_cursor(radius=200):
        try:
            x, y = pyautogui.position()
            left = max(0, x - radius)
            top = max(0, y - radius)
            width = radius * 2
            height = radius * 2

            screen_width, screen_height = pyautogui.size()
            if left + width > screen_width:
                width = screen_width - left
            if top + height > screen_height:
                height = screen_height - top

            screenshot = pyautogui.screenshot(region=(left, top, width, height))

            # Draw cursor indicator
            draw = ImageDraw.Draw(screenshot)
            cursor_x = x - left
            cursor_y = y - top
            draw.line([(cursor_x - 10, cursor_y), (cursor_x + 10, cursor_y)], fill='red', width=3)
            draw.line([(cursor_x, cursor_y - 10), (cursor_x, cursor_y + 10)], fill='red', width=3)

            return screenshot, (x, y)
        except Exception as e:
            logger.error(f"[ERROR] Error capturing cursor area: {e}")  # FIXED: Removed emoji
            return None, None


    def extract_text_from_image(image):
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            text = pytesseract.image_to_string(opencv_image)
            return text.strip()
        except Exception as e:
            logger.error(f"[ERROR] Error extracting text: {e}")  # FIXED: Removed emoji
            return ""


    def handle_screen_question(question):
        try:
            if any(word in question.lower() for word in ["cursor", "pointing", "here", "this"]):
                logger.info("[SCREEN] Capturing cursor area...")  # FIXED: Removed emoji
                image, cursor_pos = capture_area_around_cursor()
                if not image:
                    speak("Sorry, I couldn't capture the screen area.")
                    return
                context_type = "area around cursor"
            else:
                logger.info("[SCREEN] Capturing full screen...")  # FIXED: Removed emoji
                image = capture_screen()
                if not image:
                    speak("Sorry, I couldn't capture the screen.")
                    return
                context_type = "full screen"

            # Extract text from image
            screen_text = extract_text_from_image(image)
            logger.info(f"[OCR] Extracted text: {screen_text[:100]}...")  # FIXED: Removed emoji

            # Use DeepSeek with the extracted text
            if screen_text:
                logger.info("[ANALYSIS] Using extracted text with DeepSeek...")  # FIXED: Removed emoji
                text_prompt = f"I'm looking at my screen and I see this text: '{screen_text[:1000]}'. Based on this text, {question}"
                text_response = chat_with_deepseek(text_prompt)
                speak(text_response)
            else:
                speak("I couldn't extract any text from the screen. Let me search online for you.")
                search_google(question)

        except Exception as e:
            logger.error(f"[ERROR] Error handling screen question: {e}")  # FIXED: Removed emoji
            speak("Sorry, I had trouble analyzing the screen.")


def check_session_timeout():
    global session_active, last_interaction
    while True:
        if session_active and time.time() - last_interaction > SESSION_TIMEOUT:
            session_active = False
            logger.info("[TIMEOUT] Session timed out. Say 'Jarvis' to reactivate.")  # FIXED: Removed emoji
        time.sleep(5)


# Windows search functions
def search_windows(query, search_type="all"):
    """Search for files, applications, or settings on Windows"""
    try:
        logger.info(f"[SEARCH] Searching Windows for: {query}")  # FIXED: Removed emoji

        results = []
        query_lower = query.lower()

        # Search for applications in Start Menu
        if search_type in ["all", "app", "application", "program"]:
            app_results = search_applications(query_lower)
            results.extend(app_results)

        # Search for files
        if search_type in ["all", "file", "document"]:
            file_results = search_files(query_lower)
            results.extend(file_results)

        # Search for settings
        if search_type in ["all", "setting", "settings", "control"]:
            setting_results = search_settings(query_lower)
            results.extend(setting_results)

        logger.info(f"Found {len(results)} results for '{query}'")
        return results
    except Exception as e:
        logger.error(f"[ERROR] Error searching Windows: {e}")  # FIXED: Removed emoji
        return []


def search_applications(query):
    """Search for applications in Start Menu and Program Files"""
    results = []

    try:
        # Common application locations
        app_locations = [
            os.path.join(os.environ["ProgramData"], "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ["ProgramFiles"]),
            os.path.join(os.environ["ProgramFiles(x86)"]) if "ProgramFiles(x86)" in os.environ else None
        ]

        # Search for .lnk and .exe files
        for location in app_locations:
            if location and os.path.exists(location):
                for root, dirs, files in os.walk(location):
                    # Limit search depth to avoid excessive searching
                    if root.count(os.sep) - location.count(os.sep) > SEARCH_DEPTH:
                        continue

                    for file in files:
                        if file.lower().endswith((".lnk", ".exe")):
                            file_lower = file.lower()
                            if query in file_lower:
                                score = calculate_match_score(query, file_lower)
                                full_path = os.path.join(root, file)
                                results.append({
                                    "name": file,
                                    "path": full_path,
                                    "type": "application",
                                    "score": score
                                })
    except Exception as e:
        logger.error(f"[ERROR] Error searching applications: {e}")  # FIXED: Removed emoji

    # Sort by score (higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]  # Return top 5 results


def search_files(query):
    """Search for files in common locations"""
    results = []

    try:
        # Common file locations
        file_locations = [
            os.path.join(os.environ["USERPROFILE"], "Desktop"),
            os.path.join(os.environ["USERPROFILE"], "Documents"),
            os.path.join(os.environ["USERPROFILE"], "Downloads"),
            os.path.join(os.environ["USERPROFILE"], "Pictures"),
            os.path.join(os.environ["USERPROFILE"], "Videos"),
            os.path.join(os.environ["USERPROFILE"], "Music")
        ]

        # Common file extensions
        common_extensions = [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".jpg", ".png", ".mp3", ".mp4"]

        for location in file_locations:
            if os.path.exists(location):
                for root, dirs, files in os.walk(location):
                    # Limit search depth
                    if root.count(os.sep) - location.count(os.sep) > SEARCH_DEPTH:
                        continue

                    for file in files:
                        file_lower = file.lower()
                        # Prioritize files with common extensions
                        if any(file_lower.endswith(ext) for ext in common_extensions) and query in file_lower:
                            score = calculate_match_score(query, file_lower)
                            full_path = os.path.join(root, file)
                            results.append({
                                "name": file,
                                "path": full_path,
                                "type": "file",
                                "score": score
                            })
    except Exception as e:
        logger.error(f"[ERROR] Error searching files: {e}")  # FIXED: Removed emoji

    # Sort by score (higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]  # Return top 5 results


def search_settings(query):
    """Search for Windows settings"""
    results = []

    # Common Windows settings
    settings_map = {
        "display": "ms-settings:display",
        "bluetooth": "ms-settings:bluetooth",
        "wifi": "ms-settings:network-wifi",
        "network": "ms-settings:network",
        "sound": "ms-settings:sound",
        "notification": "ms-settings:notifications",
        "privacy": "ms-settings:privacy",
        "update": "ms-settings:windowsupdate",
        "account": "ms-settings:yourinfo",
        "time": "ms-settings:dateandtime",
        "language": "ms-settings:regionlanguage",
        "gaming": "ms-settings:gaming",
        "device": "ms-settings:devicemanager"
    }

    for setting_name, setting_uri in settings_map.items():
        if query in setting_name:
            score = calculate_match_score(query, setting_name)
            results.append({
                "name": f"{setting_name.capitalize()} Settings",
                "path": setting_uri,
                "type": "setting",
                "score": score
            })

    # Sort by score (higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]  # Return top 3 results


def calculate_match_score(query, target):
    """Calculate how well the query matches the target"""
    # Exact match gets highest score
    if query == target:
        return 100

    # Starting with query gets high score
    if target.startswith(query):
        return 90

    # Word boundary match gets good score
    if f" {query}" in target:
        return 80

    # Contains query gets medium score
    if query in target:
        return 70

    # Partial word match gets lower score
    query_parts = query.split()
    if any(part in target for part in query_parts):
        return 60

    # Low score for other matches
    return 50


def open_search_result(result):
    """Open a search result based on its type"""
    try:
        if result["type"] == "application" or result["type"] == "file":
            logger.info(f"[ACTION] Opening: {result['path']}")  # FIXED: Removed emoji
            os.startfile(result["path"])
            return True
        elif result["type"] == "setting":
            logger.info(f"[ACTION] Opening setting: {result['path']}")  # FIXED: Removed emoji
            os.system(f"start {result['path']}")
            return True
        return False
    except Exception as e:
        logger.error(f"[ERROR] Error opening {result['name']}: {e}")  # FIXED: Removed emoji
        return False


def handle_open_command(command):
    """Handle commands to open files, applications, or settings"""
    # Extract what to open from the command
    open_query = command.lower().replace("open", "").replace("launch", "").replace("start", "").replace("run",
                                                                                                        "").strip()

    if not open_query:
        speak("What would you like me to open?")
        return

    # Determine search type
    search_type = "all"
    if any(word in command.lower() for word in ["application", "program", "app"]):
        search_type = "application"
    elif any(word in command.lower() for word in ["file", "document"]):
        search_type = "file"
    elif any(word in command.lower() for word in ["setting", "control panel", "preference"]):
        search_type = "setting"

    # Search Windows
    speak(f"Searching for {open_query}...")
    results = search_windows(open_query, search_type)

    if not results:
        speak(f"I couldn't find anything matching '{open_query}'.")
        return

    # If we have exactly one result or a very high-scoring top result, open it directly
    if len(results) == 1 or (
            len(results) > 1 and results[0]["score"] >= 90 and results[0]["score"] - results[1]["score"] > 20):
        top_result = results[0]
        speak(f"Opening {top_result['name']}")
        if open_search_result(top_result):
            return
        else:
            speak(f"Sorry, I couldn't open {top_result['name']}.")
            return

    # Otherwise, offer choices
    speak(f"I found {len(results)} matches. Which one would you like to open?")
    for i, result in enumerate(results[:3], 1):  # Limit to top 3
        speak(f"{i}: {result['name']}")

    # Get user choice
    choice = listen(timeout=5)
    if not choice:
        speak("I didn't hear your choice. Please try again.")
        return

    # Try to parse the choice
    try:
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            selected = results[int(choice) - 1]
            speak(f"Opening {selected['name']}")
            if open_search_result(selected):
                return
            else:
                speak(f"Sorry, I couldn't open {selected['name']}.")
                return
        else:
            # Try to match the spoken choice with a result name
            for result in results:
                if choice.lower() in result["name"].lower():
                    speak(f"Opening {result['name']}")
                    if open_search_result(result):
                        return
                    else:
                        speak(f"Sorry, I couldn't open {result['name']}.")
                        return

            speak("I didn't understand your choice. Please try again.")
    except Exception as e:
        logger.error(f"[ERROR] Error processing choice: {e}")  # FIXED: Removed emoji
        speak("Sorry, I had trouble with that request.")


def handle_launcher_command(command):
    """Handle app launcher related commands"""
    global app_launcher

    command_lower = command.lower()

    if "start launcher" in command_lower or "enable launcher" in command_lower:
        if not app_launcher:
            app_launcher = AppLauncher(config)

        if app_launcher.start():
            speak("App launcher started successfully")
        else:
            speak("App launcher is already running or disabled")

    elif "stop launcher" in command_lower or "disable launcher" in command_lower:
        if app_launcher and app_launcher.running:
            app_launcher.stop()
            speak("App launcher stopped")
        else:
            speak("App launcher is not running")

    elif "launcher status" in command_lower:
        if not app_launcher:
            app_launcher = AppLauncher(config)

        status = app_launcher.get_status()
        status_msg = f"Launcher is {'running' if status['running'] else 'stopped'}. "
        status_msg += f"Monitoring {status['monitored_apps']} apps. "
        status_msg += f"Launched {status['launched_apps']} apps so far."
        speak(status_msg)

    elif "add launcher app" in command_lower:
        speak("What's the name of the application?")
        app_name = listen(timeout=10)
        if not app_name:
            speak("I didn't hear the app name")
            return

        speak("What's the executable file name?")
        executable = listen(timeout=10)
        if not executable:
            speak("I didn't hear the executable name")
            return

        # Use common directories as default
        directories = [
            os.path.join(os.path.expanduser("~"), "Downloads"),
            os.path.join(os.path.expanduser("~"), "Desktop")
        ]

        if not app_launcher:
            app_launcher = AppLauncher(config)

        app_launcher.launcher_config.add_app_config(app_name, executable, directories)
        speak(f"Added {app_name} to launcher configuration")

        # Restart launcher if it was running
        if app_launcher.running:
            app_launcher.stop()
            app_launcher.start()

    elif "remove launcher app" in command_lower:
        speak("Which application should I remove from the launcher?")
        app_name = listen(timeout=10)
        if not app_name:
            speak("I didn't hear the app name")
            return

        if not app_launcher:
            app_launcher = AppLauncher(config)

        if app_launcher.launcher_config.remove_app_config(app_name):
            speak(f"Removed {app_name} from launcher configuration")

            # Restart launcher if it was running
            if app_launcher.running:
                app_launcher.stop()
                app_launcher.start()
        else:
            speak(f"Could not find {app_name} in launcher configuration")

    elif "list launcher apps" in command_lower:
        if not app_launcher:
            app_launcher = AppLauncher(config)

        app_configs = app_launcher.launcher_config.get_app_configs()
        if not app_configs:
            speak("No applications configured for launcher")
        else:
            speak(f"I'm monitoring {len(app_configs)} applications:")
            for app_name, app_config in app_configs.items():
                enabled = "enabled" if app_config.get('enabled', 'true').lower() == 'true' else "disabled"
                speak(f"{app_name} is {enabled}")

    elif "reset launcher" in command_lower:
        if not app_launcher:
            app_launcher = AppLauncher(config)

        app_launcher.reset_launched_apps()
        speak("Reset launcher - applications can be launched again")

    else:
        speak(
            "I didn't understand that launcher command. You can say start launcher, stop launcher, launcher status, add launcher app, remove launcher app, list launcher apps, or reset launcher.")


def process_command(command):
    global session_active
    command_lower = command.lower()

    # Screen analysis (if available)
    if SCREEN_ANALYSIS_AVAILABLE:
        screen_keywords = ["what's on", "what is on", "what's this", "screen", "cursor", "pointing", "here"]
        if any(keyword in command_lower for keyword in screen_keywords):
            handle_screen_question(command)
            return "continue"

    # App launcher commands
    launcher_keywords = ["launcher", "auto launch", "monitor app", "watch app"]
    if any(keyword in command_lower for keyword in launcher_keywords):
        handle_launcher_command(command)
        return "continue"

    # Open commands - use Windows search
    if command_lower.startswith(("open ", "launch ", "start ", "run ")):
        handle_open_command(command)
        return "continue"

    # System commands
    if any(word in command_lower for word in ["exit", "quit", "stop", "bye"]):
        speak("Goodbye! Have a great day.")
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
        if folder:
            open_folder(folder)
        else:
            speak("Which folder should I open?")
    elif "sleep" in command_lower:
        session_active = False
        speak("Going to sleep. Say Jarvis to wake me up.")
        return "continue"
    elif "diagnose" in command_lower or "troubleshoot" in command_lower:
        # Run API diagnostics
        speak("Running diagnostics on my DeepSeek API connection.")
        test_deepseek_connection(force=True)
        diagnostics = get_api_diagnostics()
        for line in diagnostics:
            speak(line)
    elif "update api key" in command_lower or "change api key" in command_lower:
        # Allow user to update API key
        speak("Please enter your new DeepSeek API key:")
        new_key = input("Enter new DeepSeek API key: ").strip()
        if new_key and len(new_key) > 20:
            config['API']['deepseek_key'] = new_key
            save_config(config)
            global DEEPSEEK_API_KEY, HEADERS
            DEEPSEEK_API_KEY = new_key
            HEADERS["Authorization"] = f"Bearer {DEEPSEEK_API_KEY}"
            speak("API key updated. Testing connection...")
            test_deepseek_connection(force=True)
        else:
            speak("Invalid API key. No changes made.")
    else:
        # Regular AI chat with DeepSeek
        response = chat_with_deepseek(command)
        if response:
            speak(response, allow_interruption=True)
        else:
            speak("I couldn't get a response right now.")

    return "continue"


def create_startup_script():
    """Create a batch file to auto-start Jarvis on Windows startup"""
    try:
        startup_folder = os.path.join(os.getenv('APPDATA'), 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
        script_path = os.path.abspath(__file__)
        python_path = sys.executable

        batch_content = f'''@echo off
cd /d "{os.path.dirname(script_path)}"
start /min cmd /c "{python_path}" "{script_path}"
'''

        batch_file = os.path.join(startup_folder, 'Jarvis_Assistant.bat')
        with open(batch_file, 'w') as f:
            f.write(batch_content)

        logger.info(f"[OK] Startup script created: {batch_file}")  # FIXED: Removed emoji
        return True
    except Exception as e:
        logger.error(f"[ERROR] Could not create startup script: {e}")  # FIXED: Removed emoji
        return False


# Main execution
if __name__ == "__main__":
    logger.info("[STARTUP] DeepSeek Direct API Jarvis Assistant Starting...")  # FIXED: Removed emoji
    logger.info(f"[CONFIG] DeepSeek API Key: {DEEPSEEK_API_KEY[:10]}...")  # FIXED: Removed emoji

    # Test DeepSeek API connection first
    if not test_deepseek_connection():
        logger.warning("[WARNING] DeepSeek API connection failed. Please check:")  # FIXED: Removed emoji
        logger.warning("1. Your internet connection")
        logger.warning("2. Your API key is correct")
        logger.warning("3. The model names are correct")

        # Continue anyway for offline features
        speak("DeepSeek API connection failed, but I can still help with system commands.")

    # Ask about startup if not already set up
    startup_script = os.path.join(os.getenv('APPDATA'), 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup',
                                  'Jarvis_Assistant.bat')
    if not os.path.exists(startup_script):
        try:
            startup_choice = input("Would you like Jarvis to start automatically when Windows starts? (y/n): ").lower()
            if startup_choice == 'y':
                create_startup_script()
        except:
            pass

    # Initialize and start app launcher if enabled
    try:
        app_launcher = AppLauncher(config)
        if app_launcher.launcher_config.is_enabled():
            launcher_thread = threading.Thread(target=app_launcher.start, daemon=True)
            launcher_thread.start()
            logger.info("App launcher initialized and started")
    except Exception as e:
        logger.error(f"Error initializing app launcher: {e}")

    # Start session timeout checker
    timeout_thread = threading.Thread(target=check_session_timeout, daemon=True)
    timeout_thread.start()

    speak(
        "Hello, I am Jarvis. I'm now powered by DeepSeek using direct API calls. How can i assist you today. Say 'Jarvis' to activate me.")

    while True:
        try:
            if not session_active:
                logger.info("[WAITING] Waiting for wake word 'Jarvis'...")  # FIXED: Removed emoji
                wake_input = listen(timeout=10)

                if wake_input and "jarvis" in wake_input.lower():
                    session_active = True
                    last_interaction = time.time()
                    speak("Yes? What would you like me to do?")
                    continue
                else:
                    continue
            else:
                logger.info("[ACTIVE] Session active - listening for commands...")  # FIXED: Removed emoji
                command = listen(timeout=8)

                if command:
                    last_interaction = time.time()
                    result = process_command(command)

                    if result == "exit":
                        break
                else:
                    logger.info("[WAITING] No command received, session still active...")  # FIXED: Removed emoji

        except KeyboardInterrupt:
            logger.info("\n[EXIT] Exiting Jarvis...")  # FIXED: Removed emoji
            speak("Goodbye!")
            break
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error: {e}")  # FIXED: Removed emoji
            continue

    logger.info("[EXIT] Jarvis Assistant stopped.")  # FIXED: Removed emoji
