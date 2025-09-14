# Techtonic AI Web Chat

Techtonic AI is a web-based chatbot application powered by the TinyLlama-1.1B-Chat model. It provides users with an AI assistant capable of generating responses in English. The web interface allows real-time conversation with the AI via a simple and responsive chat interface.

## Features

- Web-based chat interface built with Flask and HTML/CSS/JavaScript
- Real-time conversation with the TinyLlama AI model
- Automatic message handling with typing indicators
- Maintains a short conversation history
- Health check endpoint to monitor model status
- Clear conversation history functionality

## Technologies Used

- Python 3.10+
- Flask
- PyTorch
- Hugging Face Transformers
- TinyLlama-1.1B-Chat Model
- HTML, CSS, JavaScript (Frontend)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/techtonic-ai.git
cd techtonic-ai

2. Create a virtual environment and activate it:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the application:

python app.py

5. Open your browser and navigate to:

http://localhost:5000

Usage

Project Structure:

techtonic-ai/
│
├─ app.py                  # Flask web server and endpoints
├─ chatbot_gptneo.py       # TinyLlama model loading and response generation
├─ templates/
│   └─ index.html          # Web interface
├─ static/
│   └─ style.css           # CSS styling
├─ requirements.txt        # Python dependencies
└─ README.md               # Project documentation

API Endpoints:

POST /ask : Send a user message and receive AI response

POST /clear : Clear conversation history

GET /health : Check the health/status of the model and server

Notes:.

This application is intended for development purposes. For production deployment, a proper WSGI server should be used.

Ensure that your machine has sufficient resources to run TinyLlama efficiently. GPU usage is recommended for faster response generation.