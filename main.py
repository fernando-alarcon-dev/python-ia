from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from textblob import TextBlob
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from language_tool_python import LanguageTool
from transformers import MarianMTModel, MarianTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List
from uuid import uuid4
import torch

app = FastAPI()

# Cargar modelo y tokenizador de GPT-2
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2").to("mps")

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model = model.to('mps')  

# Verificar si MPS está disponible y mover el modelo
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

chat_histories: Dict[str, List[str]] = {}


# Configurar CORS (para Laravel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Inicializar el modelo de resumen (¡esto faltaba!)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tool = LanguageTool('es')  # 'es' para español, 'en-US' para inglés

# Modelos pre-entrenados para traducción (inglés -> español y español -> inglés)
TRANSLATION_MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en"
}
# Almacena historiales de conversación {session_id: history}
chat_histories: Dict[str, List[str]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # ID de conversación (opcional)
    max_history: int = 5     # Máximo de mensajes a recordar

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100  # Longitud máxima del texto generado
    num_return_sequences: int = 1  # Número de textos a generar

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "en"  # Código de idioma origen (ej: "en", "es")
    target_lang: str = "es"  # Código de idioma destino

class GrammarRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 130  # Longitud máxima del resumen
    min_length: int = 30   # Longitud mínima

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-sentiment")
async def analyze_sentiment(request: TextRequest):
    blob = TextBlob(request.text)
    polarity = blob.sentiment.polarity
    # Clasificar en positivo/neutro/negativo
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {
        "text": request.text,
        "polarity": polarity,
        "sentiment": sentiment
    }

@app.post("/summarize-text")
async def summarize_text(request: SummarizeRequest):
    try:
        summary = summarizer(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=False
        )[0]['summary_text']
        return {
            "original_text": request.text,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-grammar")
async def check_grammar(request: GrammarRequest):
    try:
        matches = tool.check(request.text)
        corrections = []
        for match in matches:
            corrections.append({
                "error": match.message,
                "suggestions": match.replacements,
                "context": match.context,
                "offset": match.offset
            })
        return {
            "text": request.text,
            "errors": corrections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate-text")
async def translate_text(request: TranslationRequest):
    try:
        model_name = TRANSLATION_MODELS.get(f"{request.source_lang}-{request.target_lang}")
        if not model_name:
            raise HTTPException(status_code=400, detail="Combinación de idiomas no soportada")
        
        # Cargar modelo y tokenizador
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Tokenizar y traducir
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "languages": f"{request.source_lang}→{request.target_lang}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-text")
async def generate_text(request: TextGenerationRequest):
    try:
        # Tokenizar el prompt
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        
        # Generar texto
        outputs = model.generate(
            inputs,
            max_length=request.max_length,
            num_return_sequences=request.num_return_sequences,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7
        )
        
        # Decodificar y formatear
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        return {
            "prompt": request.prompt,
            "generated_texts": generated_texts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Obtener session_id y historial
        session_id = request.session_id or str(uuid4())
        history = chat_histories.get(session_id, [])
        
        # 2. Construir el prompt (aquí SÍ se define la variable)
        prompt = "\n".join(history + [f"Usuario: {request.message}", "Bot:"])
        
        # 3. Tokenizar y generar respuesta
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 100,
            temperature=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 4. Decodificar y actualizar historial
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = response.split("Bot:")[-1].strip()
        
        new_history = history + [f"Usuario: {request.message}", f"Bot: {bot_response}"]
        chat_histories[session_id] = new_history[-request.max_history*2:]
        
        return {
            "session_id": session_id,
            "response": bot_response,
            "history": new_history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))