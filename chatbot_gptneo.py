from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

print("TinyLlama modeli yükleniyor...")

# Model ve tokenizer'ı yükle
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Tokenizer pad token ayarı
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Sistem mesajı - Türkçe ve İngilizce destekli
SYSTEM_PROMPT = """You are Techtonic AI Assistant, a helpful and friendly AI created to assist users. 
You can communicate in both Turkish and English. 
When users write in Turkish, respond in Turkish. When they write in English, respond in English.
Always be helpful, concise, and accurate in your responses.
If you don't know something, say so honestly."""

def clean_response(response: str, user_input: str) -> str:
    """Yanıtı temizle ve sadece asistan kısmını al"""
    # Olası ayırıcıları kontrol et
    separators = ["<|assistant|>", "<|user|>", "<|system|>", "assistant:", "Assistant:", "ASSISTANT:"]
    
    for sep in separators:
        if sep in response:
            parts = response.split(sep)
            # Son asistan yanıtını al
            if len(parts) > 1:
                response = parts[-1]
                break
    
    # Kullanıcı mesajını yanıttan çıkar
    if user_input in response:
        parts = response.split(user_input)
        if len(parts) > 1:
            response = parts[-1]
    
    # Gereksiz boşlukları temizle
    response = response.strip()
    
    # Çok kısa yanıtları kontrol et
    if len(response) < 5:
        return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
    
    # Tekrar eden cümleleri kaldır
    sentences = response.split('.')
    unique_sentences = []
    for sent in sentences:
        if sent.strip() and sent.strip() not in unique_sentences:
            unique_sentences.append(sent.strip())
    
    if unique_sentences:
        response = '. '.join(unique_sentences)
        if not response.endswith('.'):
            response += '.'
    
    return response

def generate_response(user_input: str) -> str:
    """Kullanıcı mesajına AI yanıtı üretir"""
    try:
        # Dil tespiti (basit)
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        is_turkish = any(char in user_input for char in turkish_chars) or \
                    any(word in user_input.lower() for word in ['merhaba', 'nasıl', 'nedir', 'kimsin', 'selam'])
        
        # Sistem mesajını dile göre ayarla
        if is_turkish:
            system_msg = """Sen Techtonic AI Asistanı'sın. Yardımsever ve arkadaş canlısı bir yapay zeka asistanısın.
            Kullanıcılara yardımcı olmak için burada bulunuyorsun. Her zaman kibarsın ve doğru bilgi vermeye çalışıyorsun.
            Bilmediğin bir şey varsa dürüstçe söylüyorsun."""
        else:
            system_msg = SYSTEM_PROMPT
        
        # Chat formatını oluştur
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input}
        ]
        
        # Chat template uygula
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Manuel format
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_input}\n<|assistant|>\n"
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # GPU'ya taşı (varsa)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Yanıt üret - daha iyi parametrelerle
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # Daha uzun yanıtlar için
                min_new_tokens=20,   # Minimum yanıt uzunluğu
                do_sample=True,
                temperature=0.8,     # Biraz daha yaratıcı
                top_p=0.95,         # Nucleus sampling
                top_k=50,           # Top-k sampling
                repetition_penalty=1.2,  # Tekrarları azalt
                no_repeat_ngram_size=3,  # 3-gram tekrarlarını engelle
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Sadece yeni üretilen kısmı al
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Yanıtı temizle
        response = clean_response(response, user_input)
        
        # Özel sorular için önceden tanımlanmış yanıtlar
        if any(q in user_input.lower() for q in ['who are you', 'kim sin', 'kimsin', 'what are you']):
            if is_turkish:
                return "Ben Techtonic AI Asistanı'yım. Size yardımcı olmak için buradayım. Sorularınızı yanıtlayabilir, bilgi verebilir ve çeşitli konularda destek olabilirim."
            else:
                return "I am Techtonic AI Assistant, an AI chatbot designed to help and assist you. I can answer questions, provide information, and help with various tasks."
        
        if any(q in user_input.lower() for q in ['hello', 'hi', 'merhaba', 'selam']):
            if is_turkish:
                return "Merhaba! Size nasıl yardımcı olabilirim?"
            else:
                return "Hello! How can I help you today?"
        
        return response if response else "I apologize, but I couldn't generate a response. Please try again."
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return "I encountered an error while processing your request. Please try again."

# Model test
print("Model başarıyla yüklendi!")
print("\nTest yanıtı üretiliyor...")
test_response = generate_response("Hello")
print(f"Test yanıtı: {test_response[:100]}...")
print("\nModel hazır!\n")