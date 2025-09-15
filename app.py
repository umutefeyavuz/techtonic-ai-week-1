import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import time
import threading
from functools import lru_cache
import gc
import psutil

warnings.filterwarnings("ignore")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading optimized TinyLlama model...")

# Global variables
model = None
tokenizer = None
device = None
response_cache = {}
MAX_CACHE_SIZE = 80  # Reduced for better memory management

def load_model_optimized():
    """Ultra-optimized model loading based on test results"""
    global model, tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # System info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Model loading with aggressive optimizations
    load_kwargs = {
        'low_cpu_mem_usage': True,
        'torch_dtype': torch.bfloat16 if device == "cpu" else torch.float16,
    }
    
    if device == "cuda":
        load_kwargs.update({
            'device_map': "auto",
            'torch_dtype': torch.float16,
        })
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    
    # CPU optimizations based on test results
    if device == "cpu":
        # Optimal thread count to keep CPU usage under 60%
        optimal_threads = min(2, cpu_count // 4) 
        torch.set_num_threads(optimal_threads)
        print(f"PyTorch threads: {optimal_threads} (optimized for ~55% CPU usage)")
        
        # Enable CPU optimizations
        torch.backends.mkldnn.enabled = True
        torch.set_flush_denormal(True)
        
        # Try Intel Extension if available
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.bfloat16)
            print("Intel Extension for PyTorch activated")
        except ImportError:
            pass
    
    model.eval()
    model.config.use_cache = True
    
    # Reduce context window for memory savings
    if hasattr(model.config, 'max_position_embeddings'):
        model.config.max_position_embeddings = min(1024, model.config.max_position_embeddings)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    
    # Warmup
    print("Warming up model...")
    start = time.time()
    _ = generate_ultra_fast_response("test")
    print(f"Warmup completed in {time.time() - start:.2f}s")

class MemoryManager:
    """Manages memory to keep usage under 85%"""
    def __init__(self, threshold=75):
        self.threshold = threshold
        self.monitoring = False
        
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        
    def _monitor(self):
        while self.monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.threshold:
                    print(f"Memory cleanup at {memory_percent:.1f}%")
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    # Clear oldest cache entries
                    if len(response_cache) > 50:
                        keys = list(response_cache.keys())[:25]
                        for k in keys:
                            del response_cache[k]
                time.sleep(3)
            except:
                continue

memory_manager = MemoryManager(threshold=85)

def generate_ultra_fast_response(user_input: str) -> str:
    """Ultra-optimized generation based on test results"""
    global model, tokenizer, device, response_cache
    
    try:
        # Input validation
        if not user_input or len(user_input.strip()) < 2:
            return "Please provide a valid question."
        
        user_input = user_input.strip()[:200]  # Limit for speed
        
        # Cache check
        cache_key = hash(user_input.lower()[:100])
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Ultra-fast predefined responses
        quick_map = {
            'hi': "Hi! How can I help?",
            'hello': "Hello! What can I do for you?",
            'hey': "Hey there! How may I assist?",
            'who are you': "I'm your AI assistant.",
            'what are you': "I'm an AI here to help.",
            'how are you': "I'm doing well, ready to help!",
            'help': "I'm here to help! Ask me anything.",
            'thanks': "You're welcome!",
            'thank you': "My pleasure!",
            'bye': "Goodbye! Have a great day!",
            'goodbye': "See you later!",
            'test': "Test successful! I'm working.",
            'merhaba': "Merhaba! Size nasıl yardımcı olabilirim?",
            'selam': "Selam! Ne konuda yardım istiyorsunuz?",
        }
        
        user_lower = user_input.lower()
        for key, response in quick_map.items():
            if key in user_lower:
                response_cache[cache_key] = response
                return response
        
        # Check system resources
        current_cpu = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Adaptive parameters based on system load
        if current_cpu > 70 or memory_percent > 85:
            # Emergency mode - minimal processing
            max_tokens = 15
            prompt = f"Q: {user_input[:30]}\nA:"
            do_sample = False
            top_k = 5
        elif len(user_input) < 20:
            # Short input - quick response
            max_tokens = 20
            prompt = f"Q: {user_input}\nA:"
            do_sample = False
            top_k = 10
        else:
            # Normal mode
            max_tokens = 30
            prompt = f"User: {user_input}\nAssistant:"
            do_sample = True
            top_k = 15
        
        # Tokenize with minimal overhead
        inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=min(100, 256 - max_tokens),  # Daha kısa context
        truncation=True,
        padding=False
        )
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation parameters optimized for speed
        gen_kwargs = {
            'max_new_tokens': max_tokens,
            'min_new_tokens': 3,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'early_stopping': True,
            'use_cache': True,
            'output_scores': False,
            'return_dict_in_generate': False,
        }
        
        if do_sample:
            gen_kwargs.update({
                'do_sample': True,
                'temperature': 0.7,
                'top_k': top_k,
                'top_p': 0.9,
                'repetition_penalty': 1.05,
            })
        else:
            gen_kwargs['do_sample'] = False
            gen_kwargs['num_beams'] = 1
        
        # CPU-specific optimizations
        if device == "cpu":
            gen_kwargs['max_new_tokens'] = min(20, max_tokens)
            gen_kwargs['do_sample'] = False  # Greedy for CPU
        
        # Generate with timing
        start_time = time.time()
        with torch.no_grad():
            # Memory cleanup if needed
            if memory_percent > 80:
                gc.collect()
            
            outputs = model.generate(**inputs, **gen_kwargs)
        
        gen_time = time.time() - start_time
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean response
        if not response or len(response) < 3:
            response = "Could you please rephrase that?"
        
        # Take only first sentence for speed
        for delimiter in ['. ', '! ', '? ', '\n']:
            if delimiter in response:
                response = response.split(delimiter)[0] + delimiter[0]
                break
        
        # Limit response length
        if len(response) > 100:
            response = response[:97] + "..."
        
        # Cache management
        if len(response_cache) < MAX_CACHE_SIZE:
            response_cache[cache_key] = response
        elif len(response_cache) > MAX_CACHE_SIZE * 1.5:
            # Cleanup old entries
            keys = list(response_cache.keys())
            for key in keys[:-50]:
                del response_cache[key]
        
        print(f"[Gen {gen_time:.2f}s | CPU {current_cpu:.0f}% | Mem {memory_percent:.0f}%] {user_input[:30]}...")
        
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        return "I need a moment to process that. Please try again."

# Initialize model
load_model_optimized()
memory_manager.start()

# Flask app with optimizations
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600

conversation_history = []
MAX_HISTORY = 5  # Very limited for memory

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
        
        if len(user_message) > 500:
            return jsonify({
                "reply": "Please keep your message shorter.", 
                "error": False
            })
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Generate response
        start_total = time.time()
        bot_reply = generate_ultra_fast_response(user_message)
        total_time = time.time() - start_total
        
        # Minimal history update
        conversation_history.append({
            "user": user_message[:100], 
            "bot": bot_reply[:100], 
            "time": timestamp
        })
        
        # Keep history very small
        if len(conversation_history) > MAX_HISTORY:
            conversation_history.pop(0)
        
        return jsonify({
            "reply": bot_reply, 
            "error": False, 
            "timestamp": timestamp,
            "response_time": f"{total_time:.2f}s"
        })
        
    except Exception as e:
        print(f"Flask error: {e}")
        return jsonify({
            "reply": "Service temporarily unavailable.", 
            "error": True
        })

@app.route("/health")
def health():
    try:
        start = time.time()
        test_response = generate_ultra_fast_response("test")
        response_time = time.time() - start
        
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        status = "healthy"
        if response_time > 2.0:
            status = "slow"
        if cpu_usage > 70 or memory_usage > 90:
            status = "degraded"
        
        return jsonify({
            "status": status,
            "model": "TinyLlama-1.1B-Ultra-Optimized",
            "device": device,
            "response_time": f"{response_time:.3f}s",
            "cache_size": len(response_cache),
            "cpu_usage": f"{cpu_usage:.1f}%",
            "memory_usage": f"{memory_usage:.1f}%",
            "torch_threads": torch.get_num_threads() if device == "cpu" else "N/A"
        })
    except:
        return jsonify({"status": "error"})

@app.route("/clear", methods=["POST"])
def clear_history():
    global conversation_history, response_cache
    conversation_history = []
    # Keep some cache for common queries
    if len(response_cache) > 20:
        response_cache = dict(list(response_cache.items())[-20:])
    gc.collect()
    return jsonify({"message": "History cleared.", "success": True})

@app.route("/stats")
def stats():
    """Performance statistics endpoint"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    
    stats_data = {
        "cache_size": len(response_cache),
        "history_size": len(conversation_history),
        "device": device,
        "model_dtype": str(model.dtype),
        "torch_threads": torch.get_num_threads() if device == "cpu" else "N/A",
        "cpu_usage": f"{cpu_percent:.1f}%",
        "memory_usage": f"{memory_info.percent:.1f}%",
        "memory_available_gb": f"{memory_info.available / (1024**3):.1f}",
    }
    
    if device == "cuda":
        stats_data["gpu_memory_gb"] = f"{torch.cuda.memory_allocated() / 1e9:.2f}"
    
    return jsonify(stats_data)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("TECHTONIC AI - ULTRA OPTIMIZED SERVER")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Target CPU usage: <60%")
    print(f"Target Memory usage: <85%")
    print(f"Cache: {MAX_CACHE_SIZE} entries")
    print("Server: http://localhost:5000")
    print("=" * 50)
    
    app.run(
        debug=False,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )
    
