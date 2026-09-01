import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

def generate_answer(query: str, context: str, chat_history: list = None, language: str = "English") -> str:
    """
    Generate an answer using Gemini based ONLY on the provided document context.
    """
    if not client:
        return "⚠️ Gemini API key is missing. Please add it to your .env file."
        
    if not context.strip():
        return "I couldn't find relevant information in the uploaded documents."

    # Format previous conversation history (last 4 turns)
    history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

    # Simple, clear prompt
    prompt = f"""
You are an AI research assistant. Answer the user's question ONLY using the provided Document Context. 
If the answer is not in the context, say "I couldn't find this in the uploaded documents."

Language to respond in: {language}

Document Context:
{context}

Conversation History:
{history_text}

User Question:
{query}

Answer:
"""
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

def generate_summary(documents_text: str, filename: str) -> str:
    """
    Generate a short summary of the document.
    """
    if not client:
        return "⚠️ Gemini API key is missing."
        
    prompt = f"""
Summarize the following document in a few bullet points. Include the main topic and key points.
Filename: {filename}

Document Text:
{documents_text[:4000]}
"""
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip() if response.text else "Summary unavailable."
    except Exception:
        return "Summary unavailable."

def extract_keywords(text: str) -> list:
    """
    Extract up to 10 important keywords from the text.
    """
    if not client:
        return []
        
    prompt = f"Extract the 10 most important keywords from this text as a comma-separated list:\n\n{text[:2500]}"
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        if response.text:
            # Split by comma and clean up whitespace
            keywords = [k.strip() for k in response.text.split(",") if k.strip()]
            return keywords[:10]
        return []
    except Exception:
        return []

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using Gemini.
    """
    if not client:
        return "⚠️ Gemini API key is missing."
        
    prompt = "Transcribe the following audio exactly as spoken."
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                prompt
            ]
        )
        return response.text.strip() if response.text else "Could not transcribe audio."
    except Exception as e:
        return f"Audio Transcription Error: {str(e)}"