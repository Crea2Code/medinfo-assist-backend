from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, requests
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Initialiser le client Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configuration FastAPI
app = FastAPI(title="MedInfo Assist - API IA médicale", version="3.0")

# CORS (connexion avec le frontend Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pour test, tu pourras restreindre à ton domaine plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schéma d'entrée
class TextRequest(BaseModel):
    text: str

# Endpoint principal d'analyse médicale
@app.post("/analyze")
async def analyze_text(request: TextRequest):
    text = request.text.strip()
    if not text:
        return {"error": "Le texte est vide."}

    try:
        # Prompt IA
        prompt = (
            f"Explique simplement ce texte médical pour qu'un patient non spécialiste puisse le comprendre :\n{text}"
        )

        # Appel à l’API Groq avec le nouveau modèle
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ modèle à jour
            messages=[
                {"role": "system", "content": "Tu es un assistant médical qui vulgarise les textes médicaux."},
                {"role": "user", "content": prompt},
            ],
        )

        simplified = response.choices[0].message.content.strip()

        return {
            "original_text": text,
            "simplified_text": simplified,
            "model_used": "llama-3.1-8b-instant (Groq)"
        }

    except Exception as e:
        return {"error": f"Erreur Groq : {str(e)}"}
    
# 🔹 Modèle de données pour l’entrée utilisateur
class HealthAdviceRequest(BaseModel):
    question: str    
    
# 🔹 Endpoint pour demander un conseil santé
@app.post("/advice")
async def get_health_advice(request: HealthAdviceRequest):
    question = request.question
    prompt = (
        f"Donne un conseil de prévention santé clair et bienveillant pour la question suivante : "
        f"{question}. Utilise un ton simple et empathique, accessible à tous."
    )

    try:
        # Exemple avec Groq ou tout modèle compatible OpenAI
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        MODEL_ID = "llama-3.1-8b-instant"

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        data = response.json()
        text = data["choices"][0]["message"]["content"]

        return {
            "question": question,
            "advice": text,
            "model_used": MODEL_ID,
        }

    except Exception as e:
        return {"error": f"Erreur Groq : {e}"}

# Route de test
@app.get("/")
async def root():
    return {"message": "Bienvenue sur MedInfo Assist API (Groq ✅ nouvelle version)"}
