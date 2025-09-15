from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import threading
from queue import Queue
import time
import gc
import os
import psutil
from typing import Optional, Dict
warnings.filterwarnings("ignore")

print("Loading optimized TinyLlama model with enhanced performance...")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# System resources check
cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)
print(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")

# Enhanced device and optimization detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_kwargs = {
    'low_cpu_mem_usage': True,
    'torch_dtype': torch.bfloat16 if device == "cpu" else torch.float16,
}

if device == "cuda":
    model_kwargs.update({
        'device_map': "auto",
        'torch_dtype': torch.float16,
    })

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

if device == "cpu":
    # Optimal thread count based on test results (CPU peaks at 55-60%)
    # More conservative thread usage to reduce CPU spikes
    optimal_threads = max(2, min(4, cpu_count // 2))  # Reduced from 8
    torch.set_num_threads(optimal_threads)
    print(f"PyTorch threads set to: {optimal_threads} (was causing CPU spikes)")
    
    # CPU backend optimizations
    torch.backends.mkldnn.enabled = True
    torch.set_flush_denormal(True)
    
    # Intel Extension - Fixed import error with proper error handling
    intel_optimized = False
    try:
        # Only try import if we detect Intel CPU
        cpu_info = str(psutil.cpu_freq()).lower()
        if 'intel' in cpu_info or os.environ.get('INTEL_EXTENSION', 'false').lower() == 'true':
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.bfloat16)
            intel_optimized = True
            print("Intel Extension for PyTorch activated")
    except (ImportError, Exception) as e:
        print(f"Intel Extension not available (this is normal): {type(e).__name__}")
        print("Continuing with standard PyTorch optimizations")
    
    # Additional CPU optimizations
    torch.backends.quantized.engine = 'fbgemm'  # Faster quantized operations
    if hasattr(torch.backends, 'opt_einsum'):
        torch.backends.opt_einsum.enabled = True

# Model optimizations based on test results
model.eval()
model.config.use_cache = True

# Memory optimization - Reduce memory footprint (was 85-90%)
if hasattr(model.config, 'max_position_embeddings'):
    # Reduce context window for memory savings
    model.config.max_position_embeddings = min(1024, model.config.max_position_embeddings)

# Enhanced tokenizer setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Memory-conscious caching with cleanup
class OptimizedCache:
    def __init__(self, max_size: int = 50):  # Reduced from 100
        self.cache: Dict[int, str] = {}
        self.access_times: Dict[int, float] = {}
        self.max_size = max_size
        self.cleanup_threshold = max_size * 1.5
    
    def get(self, key: int) -> Optional[str]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: int, value: str):
        if len(self.cache) >= self.cleanup_threshold:
            self._cleanup()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _cleanup(self):
        # Remove least recently used items
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) - self.max_size
        
        for key, _ in sorted_items[:items_to_remove]:
            del self.cache[key]
            del self.access_times[key]
        
        # Force garbage collection after cleanup
        gc.collect()
    
    def size(self) -> int:
        return len(self.cache)

# Initialize optimized cache
response_cache = OptimizedCache(max_size=50)

# System prompt - minimal for performance
SYSTEM_PROMPT = "You are a helpful assistant. Be brief and accurate."

def get_cache_key(text: str) -> int:
    """Optimized cache key generation"""
    return hash(text.lower().strip()[:100])  # Only hash first 100 chars

def clean_response(response: str, user_input: str) -> str:
    """Memory-optimized response cleaning"""
    if not response or len(response) < 3:
        return "I need more context to provide a helpful answer."
    
    # Remove generation artifacts - optimized
    separators = ["<|assistant|>", "<|user|>", "<|system|>", "\nUser:", "\nAssistant:"]
    
    for sep in separators:
        idx = response.find(sep)
        if idx != -1:
            response = response[idx + len(sep):].strip()
            break
    
    response = response.strip()
    
    # Truncate for memory efficiency and speed
    if len(response) > 150:
        # Find natural break point
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[0]) > 20:
            response = sentences[0].strip() + '.'
        else:
            response = response[:150].rsplit(' ', 1)[0] + '...'
    
    return response

def generate_response_optimized(user_input: str) -> str:
    """Ultra-optimized response generation with memory management"""
    try:
        # Input validation and preprocessing
        if not user_input or len(user_input.strip()) < 2:
            return "Please provide a valid question."
        
        user_input = user_input.strip()[:400]  # Reduced from 500
        
        # Cache check
        cache_key = get_cache_key(user_input)
        cached_response = response_cache.get(cache_key)
        if cached_response:
            print("Response from cache (memory optimized)")
            return cached_response
        
        # Ultra-fast predefined responses - expanded set
        quick_responses = {
            'hello': "Hello! How can I help you?",
            'hi': "Hi there! What can I do for you?",
            'hey': "Hey! How may I assist?",
            'who are you': "I'm an AI assistant here to help you.",
            'what are you': "I'm an AI designed to help with questions and tasks.",
            'how are you': "I'm doing well and ready to help!",
            'thanks': "You're welcome!",
            'thank you': "My pleasure!",
            'bye': "Goodbye! Have a great day!",
            'goodbye': "See you later!",
            'help': "I'm here to help! What do you need?",
            'test': "Test successful! I'm working properly.",
            'merhaba': "Merhaba! Size nasıl yardımcı olabilirim?",
            'selam': "Selam! Ne konuda yardım istiyorsunuz?",
            'nasılsın': "İyiyim, teşekkürler! Size nasıl yardımcı olabilirim?",
        }
        
        user_lower = user_input.lower()
        for key, response in quick_responses.items():
            if key in user_lower:
                response_cache.put(cache_key, response)
                return response
        
        # Dynamic generation parameters based on input length and system load
        current_cpu = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Adaptive parameters based on system state
        if current_cpu > 70 or memory_percent > 85:
            # High load mode - maximum efficiency
            max_tokens = 20
            use_sampling = False
            top_k = 5
            prompt_template = f"Q: {user_input[:50]}\nA:"
        elif len(user_input) < 20:
            # Short input mode
            max_tokens = 25
            use_sampling = True
            top_k = 10
            prompt_template = f"User: {user_input}\nAssistant:"
        else:
            # Normal mode
            max_tokens = 35
            use_sampling = True
            top_k = 15
            prompt_template = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
        
        # Tokenize with memory optimization
        inputs = tokenizer(
            prompt_template,
            return_tensors="pt",
            padding=False,  # No padding for single input
            truncation=True,
            max_length=min(200, 512 - max_tokens)  # Leave room for generation
        )
        
        # Move to device
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Optimized generation parameters - based on test results
        generation_kwargs = {
            'max_new_tokens': max_tokens,
            'min_new_tokens': 3,
            'do_sample': use_sampling,
            'temperature': 0.7 if use_sampling else None,
            'top_p': 0.85 if use_sampling else None,
            'top_k': top_k,
            'repetition_penalty': 1.05,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'early_stopping': True,
            'use_cache': True,
            'output_scores': False,  # Disable score computation for speed
            'return_dict_in_generate': False,  # Simpler output format
        }
        
        # CPU-specific optimizations from test results
        if device == "cpu":
            generation_kwargs.update({
                'num_beams': 1,  # Greedy decoding for speed
                'do_sample': False,  # Deterministic for CPU
                'max_new_tokens': min(20, max_tokens),
            })
        
        # Generate with memory management
        start_time = time.time()
        with torch.no_grad():
            # Force garbage collection before generation if memory is high
            if memory_percent > 80:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            outputs = model.generate(**inputs, **generation_kwargs)
        
        generation_time = time.time() - start_time
        
        # Extract and decode response
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean and optimize response
        response = clean_response(response, user_input)
        
        # Cache the response
        response_cache.put(cache_key, response)
        
        # Performance logging (optional)
        print(f"Generated in {generation_time:.3f}s | Cache size: {response_cache.size()} | CPU: {current_cpu:.1f}%")
        
        return response
        
    except Exception as e:
        print(f"Generation error: {str(e)[:50]}...")
        # Fallback response
        return "I encountered an issue. Please try rephrasing your question."

# Memory monitoring and cleanup thread
class MemoryMonitor:
    def __init__(self):
        self.monitoring = False
        self.cleanup_threshold = 85  # Memory percentage
        
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        
    def _monitor(self):
        while self.monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.cleanup_threshold:
                    print(f"Memory cleanup triggered at {memory_percent:.1f}%")
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    
                time.sleep(5)  # Check every 5 seconds
            except:
                continue

# Initialize memory monitor
memory_monitor = MemoryMonitor()
memory_monitor.start()

# Enhanced async response generator
class AsyncResponseGenerator:
    def __init__(self):
        self.queue = Queue(maxsize=5)  # Limit queue size
        self.worker_thread = None
        self.is_running = False
    
    def start_worker(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        while self.is_running:
            try:
                task = self.queue.get(timeout=2)
                if task is None:
                    break
                    
                user_input, callback = task
                response = generate_response_optimized(user_input)
                callback(response)
                self.queue.task_done()
                
            except Exception as e:
                print(f"Async worker error: {e}")
                continue
    
    def generate_async(self, user_input: str, callback):
        try:
            self.queue.put((user_input, callback), timeout=1)
        except:
            callback("System is busy. Please try again.")
    
    def stop(self):
        self.is_running = False
        self.queue.put(None)

# Global async generator
async_generator = AsyncResponseGenerator()

# Model warmup with memory tracking
print("Warming up model...")
warmup_start = time.time()
initial_memory = psutil.virtual_memory().percent

_ = generate_response_optimized("Hello")

warmup_time = time.time() - warmup_start
final_memory = psutil.virtual_memory().percent

print(f"Warmup completed in {warmup_time:.2f}s")
print(f"Memory usage: {initial_memory:.1f}% -> {final_memory:.1f}%")

# Performance summary
print("\n" + "="*50)
print("OPTIMIZED MODEL READY")
print("="*50)
print(f"Device: {device}")
print(f"Model dtype: {model.dtype}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"Cache capacity: {response_cache.max_size} entries")
print(f"Memory monitoring: Active")
if device == "cpu":
    print(f"Intel optimization: {'✓' if 'intel_optimized' in locals() and intel_optimized else '✗'}")
print(f"Quick responses: {len(response_cache.cache.keys() if hasattr(response_cache, 'cache') else 0)} cached")
print("="*50)