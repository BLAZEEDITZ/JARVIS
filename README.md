🧠 Jarvis prototype - AI Voice Assistant for Desktop
Jarvis is a smart, voice-controlled AI assistant for your desktop that combines natural conversation, system control, and advanced screen analysis. Designed for productivity, accessibility, and convenience, Jarvis listens to your voice, understands your screen, and acts intelligently.

🎤 Voice Control
Wake Word Activation: Say "Jarvis" to activate voice listening.

Session Management: Jarvis listens for commands with a 30-second timeout.

Speech Recognition: Converts your voice into text using powerful STT engines.

Text-to-Speech: Replies in a natural, human-like voice.

Interruption Handling: Smart pause/resume logic ensures fluid conversations.

💻 System Control
Jarvis can manage your system with voice commands:

🔌 Shutdown your computer

🌐 Open Chrome browser

📂 Open folders: Desktop, Documents, Downloads, etc.

🔍 Perform Google searches

🤖 AI Chat
Jarvis integrates with powerful language models to chat naturally:

🧠 DeepSeek AI for general conversations

🔍 NLP-powered understanding of your intent

💬 Conversational responses that feel intuitive and contextual

🆕 NEW SCREEN ANALYSIS FEATURES
📸 Screen Capture
🖥️ Full Screen Capture

🎯 Cursor Area Capture with crosshair selection

📝 OCR Text Extraction from captured images

🪟 Active Window Detection

🧠 ChatGPT Vision Integration
🔑 Uses your ChatGPT-4o API key

🖼️ Understands your screen content via AI vision

❓ Answers questions about visual elements

🔁 Smart fallback to text/OCR and web search

🎯 Smart Question Detection
Ask questions naturally, and Jarvis will figure out what to analyze:

🧾 "What's on my screen?" → Full-screen analysis

🖱️ "What's this I'm pointing at?" → Cursor area capture + AI

🔘 "What does this button do?" → UI element explanation

🔤 "Read this text" → Extracts and reads text via OCR

🪟 "What application is this?" → Identifies active window

🔄 Intelligent Fallback System
If vision-based AI fails, Jarvis doesn’t stop:

ChatGPT Vision – Primary: Visual analysis with GPT-4o

DeepSeek AI + OCR – Secondary: Text-based analysis

Google Search – Fallback: Contextual web search

🛠️ Requirements
Python 3.8+

OpenAI API key (for GPT-4o Vision)

Microphone and speaker

Required Python packages (see requirements.txt)
