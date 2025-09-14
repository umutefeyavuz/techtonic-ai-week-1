import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request, jsonify
from datetime import datetime

warnings.filterwarnings("ignore")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = (
    "You are Techtonic AI Assistant, a helpful and friendly AI created to assist users. "
    "Always answer in English. "
    "Be concise, accurate, and polite. "
    "If you don't know something, say so honestly."
)

def clean_response(response: str) -> str:
    # Remove known separators and redundant text
    for sep in ["<|assistant|>", "<|user|>", "<|system|>", "assistant:", "Assistant:", "ASSISTANT:"]:
        if sep in response:
            response = response.split(sep)[-1]
    response = response.strip()
    # Remove repeated sentences
    sentences = []
    for sent in response.split('.'):
        sent = sent.strip()
        if sent and sent not in sentences:
            sentences.append(sent)
    final = '. '.join(sentences).strip()
    if final and not final.endswith('.'):
        final += '.'
    # Fallback for too short/empty responses
    if len(final) < 5:
        return "Sorry, I couldn't generate a suitable answer. Please rephrase your question."
    return final

def generate_response(user_input: str) -> str:
    try:
        system_msg = SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input}
        ]
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_input}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = clean_response(response)
        # Hardcoded responses for greetings/identity
        low = user_input.lower()
        if any(q in low for q in ['who are you', 'what are you']):
            return "I am Techtonic AI Assistant, here to help you. Feel free to ask me anything."
        if any(q in low for q in ['hello', 'hi', 'hey']):
            return "Hello! How can I assist you today?"
        return response
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return "Sorry, something went wrong. Please try again."

# Flask app setup
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
conversation_history = []
MAX_HISTORY = 10

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Please enter a message.", "error": False})
        if len(user_message) > 1000:
            return jsonify({"reply": "Your message is too long. Please keep it under 1000 characters.", "error": False})
        timestamp = datetime.now().strftime("%H:%M")
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_message, "timestamp": timestamp})
        bot_reply = generate_response(user_message)
        conversation_history.append({"role": "assistant", "content": bot_reply, "timestamp": timestamp})
        # Keep only recent history
        if len(conversation_history) > MAX_HISTORY * 2:
            conversation_history[:2] = []
        print(f"[{timestamp}] User: {user_message}")
        print(f"[{timestamp}] Bot: {bot_reply[:100]}...")
        return jsonify({"reply": bot_reply, "error": False, "timestamp": timestamp})
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"reply": "Sorry, an error occurred. Please try again.", "error": True})

@app.route("/health")
def health():
    try:
        test_response = generate_response("test")
        status = "healthy" if test_response else "degraded"
    except:
        status = "unhealthy"
    return jsonify({
        "status": status,
        "model": "TinyLlama-1.1B-Chat",
        "version": "1.0.0",
        "conversation_count": len(conversation_history)
    })

@app.route("/clear", methods=["POST"])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({"message": "Conversation history cleared.", "success": True})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Page not found.", "status": 404}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error.", "status": 500}), 500

if __name__ == "__main__":
    print("="*50)
    print("TECHTONIC AI WEB SERVER")
    print("="*50)
    print("http://localhost:5000")
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )