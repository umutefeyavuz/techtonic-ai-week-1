import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import time
import gc
import psutil
from collections import OrderedDict
from functools import wraps

warnings.filterwarnings("ignore")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Starting optimized TinyLlama server...")

# ============= OPTIMIZED MODEL CLASS =============
class TinyLlamaServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = OrderedDict()
        self.max_cache = 40
        
        # Predefined responses for instant reply
        self.instant_replies = {
            'hi': "Hi! How can I help?",
            'hello': "Hello! What can I do for you?",
            'hey': "Hey there! How may I assist?",
            'thanks': "You're welcome!",
            'bye': "Goodbye! Have a great day!",
            'test': "Test successful!",
            'help': "I'm here to help! What do you need?",
            'who are you': "I'm your AI assistant.",
            'what are you': "I'm an AI here to help.",
            'how are you': "I'm doing well, ready to help!",
            'merhaba': "Merhaba! Size nasıl yardımcı olabilirim?",
            'selam': "Selam! Ne konuda yardım istiyorsunuz?",
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load model with minimal overhead"""
        print(f"Device: {self.device}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model with optimizations
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # CPU optimizations
        if self.device == "cpu":
            threads = min(2, psutil.cpu_count() // 4)
            torch.set_num_threads(threads)
            torch.backends.mkldnn.enabled = True
            print(f"CPU threads: {threads}")
        
        self.model.eval()
        
        # Warmup
        print("Warming up...")
        self._generate("test")
        print("Ready!")
    
    def _generate(self, text: str) -> str:
        """Core generation logic"""
        # Quick validation
        if not text or len(text.strip()) < 2:
            return "Please enter a message."
        
        text = text.strip()[:200]
        
        # Instant replies
        text_lower = text.lower()
        for key, reply in self.instant_replies.items():
            if key in text_lower:
                return reply
        
        # Cache check
        cache_key = hash(text_lower[:100])
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        # Resource check
        cpu = psutil.cpu_percent(interval=0.05)
        mem = psutil.virtual_memory().percent
        
        # Dynamic parameters
        if cpu > 70 or mem > 85:
            max_tokens, temperature, sample = 15, 0.5, False
        else:
            max_tokens, temperature, sample = 25, 0.7, True
        
        # Generate
        try:
            prompt = f"User: {text}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=150)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if sample else None,
                    do_sample=sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
            
            # Decode
            new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean
            if not response:
                response = "Could you please rephrase that?"
            
            # Truncate
            for delim in ['. ', '! ', '? ']:
                if delim in response:
                    response = response.split(delim)[0] + delim[0]
                    break
            
            if len(response) > 100:
                response = response[:97] + "..."
            
            # Cache
            self.cache[cache_key] = response
            if len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return "Service temporarily unavailable."
    
    def generate(self, text: str) -> tuple:
        """Generate with timing"""
        start = time.time()
        response = self._generate(text)
        duration = time.time() - start
        return response, duration
    
    def get_stats(self):
        """Get current stats"""
        return {
            'device': self.device,
            'cache_size': len(self.cache),
            'cpu': f"{psutil.cpu_percent(interval=0.1):.1f}%",
            'memory': f"{psutil.virtual_memory().percent:.1f}%",
        }

# ============= FLASK APP =============
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600

# Initialize model
ai_model = TinyLlamaServer()

# Simple in-memory history (limited)
history = []
MAX_HISTORY = 5

def cleanup_history():
    """Keep history small"""
    global history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"reply": "Please enter a message.", "error": False})
        
        if len(message) > 500:
            return jsonify({"reply": "Please keep messages under 500 characters.", "error": False})
        
        # Generate response
        reply, duration = ai_model.generate(message)
        timestamp = datetime.now().strftime("%H:%M")
        
        # Update history
        history.append({
            "user": message[:100],
            "bot": reply[:100],
            "time": timestamp
        })
        cleanup_history()
        
        return jsonify({
            "reply": reply,
            "error": False,
            "timestamp": timestamp,
            "response_time": f"{duration:.2f}s"
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"reply": "Service temporarily unavailable.", "error": True})

@app.route("/health")
def health():
    try:
        # Quick health check
        test_reply, test_time = ai_model.generate("test")
        stats = ai_model.get_stats()
        
        status = "healthy"
        if test_time > 2.0:
            status = "slow"
        elif float(stats['cpu'].rstrip('%')) > 70:
            status = "degraded"
        
        return jsonify({
            "status": status,
            "model": "TinyLlama-Optimized",
            **stats
        })
    except:
        return jsonify({"status": "error"})

@app.route("/clear", methods=["POST"])
def clear():
    global history
    history = []
    gc.collect()
    return jsonify({"message": "History cleared.", "success": True})

@app.route("/stats")
def stats():
    return jsonify(ai_model.get_stats())

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

# ============= PERIODIC CLEANUP =============
def periodic_cleanup():
    """Background cleanup thread"""
    import threading
    def cleanup():
        while True:
            time.sleep(30)  # Every 30 seconds
            try:
                mem = psutil.virtual_memory().percent
                if mem > 80:
                    gc.collect()
                    if ai_model.device == "cuda":
                        torch.cuda.empty_cache()
                    print(f"Memory cleanup at {mem:.1f}%")
            except:
                pass
    
    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()

# ============= MAIN =============
if __name__ == "__main__":
    print("=" * 50)
    print("TECHTONIC AI - OPTIMIZED SERVER")
    print("=" * 50)
    print(f"Device: {ai_model.device}")
    print(f"Cache: {ai_model.max_cache} entries")
    print("Server: http://localhost:5000")
    print("=" * 50)
    
    # Start cleanup thread
    periodic_cleanup()
    
    # Run server
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )