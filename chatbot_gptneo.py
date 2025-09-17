from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import time
import gc
import psutil
from typing import Dict, Optional
from collections import OrderedDict

warnings.filterwarnings("ignore")

print("Loading optimized TinyLlama model...")

class OptimizedTinyLlama:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.cache = OrderedDict()  # LRU cache
        self.max_cache_size = 50
        
        # Quick responses for immediate return
        self.quick_responses = {
            'hi': "Hi! How can I help?",
            'hello': "Hello! What can I do for you?",
            'hey': "Hey there! How may I assist?",
            'thanks': "You're welcome!",
            'bye': "Goodbye! Have a great day!",
            'test': "Test successful! I'm working.",
            'help': "I'm here to help! What do you need?",
        }
        
        self._load_model()
    
    def _load_model(self):
        """Optimized model loading"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading parameters
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # CPU optimizations
        if self.device == "cpu":
            # Conservative thread count for stable CPU usage
            cpu_cores = psutil.cpu_count()
            optimal_threads = min(2, max(1, cpu_cores // 4))
            torch.set_num_threads(optimal_threads)
            torch.backends.mkldnn.enabled = True
            print(f"CPU threads: {optimal_threads}")
        
        self.model.eval()
        
        # Warmup
        print("Warming up...")
        _ = self.generate("test")
        print("Model ready!")
    
    def _clean_cache(self):
        """LRU cache cleanup"""
        if len(self.cache) > self.max_cache_size:
            # Remove oldest 25% of items
            remove_count = self.max_cache_size // 4
            for _ in range(remove_count):
                self.cache.popitem(last=False)
            gc.collect()
    
    def generate(self, text: str) -> str:
        """Optimized text generation"""
        # Input validation
        if not text or len(text.strip()) < 2:
            return "Please provide a valid question."
        
        text = text.strip()[:200]  # Limit input length
        
        # Check quick responses first
        text_lower = text.lower()
        for key, response in self.quick_responses.items():
            if key in text_lower:
                return response
        
        # Check cache
        cache_key = hash(text_lower[:100])
        if cache_key in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        # Check system resources
        cpu_usage = psutil.cpu_percent(interval=0.05)
        mem_usage = psutil.virtual_memory().percent
        
        # Adaptive generation parameters
        if cpu_usage > 70 or mem_usage > 85:
            # Emergency mode
            max_tokens = 15
            temperature = 0.5
            do_sample = False
        else:
            # Normal mode
            max_tokens = 25
            temperature = 0.7
            do_sample = True
        
        # Simple prompt
        prompt = f"User: {text}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=150,
            padding=False
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=0.9 if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean response
            if not response or len(response) < 3:
                response = "Could you please rephrase that?"
            
            # Truncate at first sentence
            for delim in ['. ', '! ', '? ']:
                if delim in response:
                    response = response.split(delim)[0] + delim[0]
                    break
            
            # Final length limit
            if len(response) > 100:
                response = response[:97] + "..."
            
            # Cache the response
            self.cache[cache_key] = response
            self._clean_cache()
            
            return response
            
        except Exception as e:
            print(f"Error: {e}")
            return "I need a moment to process that. Please try again."
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'device': self.device,
            'cache_size': len(self.cache),
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'threads': torch.get_num_threads() if self.device == "cpu" else "N/A"
        }

# Global model instance
model_instance = None

def get_model():
    """Get or create model instance"""
    global model_instance
    if model_instance is None:
        model_instance = OptimizedTinyLlama()
    return model_instance

def generate_response_optimized(user_input: str) -> str:
    """Main generation function for compatibility"""
    model = get_model()
    return model.generate(user_input)

# Initialize on import
if __name__ != "__main__":
    model_instance = OptimizedTinyLlama()
    print("\n" + "="*50)
    print("MODEL READY")
    print("="*50)
    stats = model_instance.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*50)