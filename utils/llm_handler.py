"""
LLM handler using Google Gemini API for intelligent QA.
"""

import os
from typing import List, Dict, Optional

import google.generativeai as genai
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────────────────────
# Configure Gemini
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY not found in .env file"
    )

genai.configure(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────
# Gemini Model
# ─────────────────────────────────────────────────────────────
model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)
# ─────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are DocMind, an advanced AI-powered document analyst
and research assistant.

Your job is to answer questions ONLY using the provided
document context.

Rules:
- Never fabricate information
- If answer is unavailable, say:
  "I couldn't find this in the uploaded documents."
- Use concise professional language
- Use markdown formatting
- Use bullet points when useful
- Mention source/page references naturally
- Focus on factual accuracy
"""


# ─────────────────────────────────────────────────────────────
# Generate Answer
# ─────────────────────────────────────────────────────────────
def generate_answer(
    query: str,
    context: str,
    chat_history: Optional[List[Dict]] = None,
    language: str = "English",
) -> str:
    """
    Generate contextual answer using Gemini.
    """

    # Handle empty retrieval
    if not context.strip():
        return "I couldn't find relevant information in the uploaded documents."

    # Build conversation history
    history_text = ""

    if chat_history:

        for turn in chat_history[-4:]:

            role = turn.get("role", "")

            content = turn.get("content", "")

            history_text += f"{role}: {content}\n"

    # Final prompt
    prompt = f"""
{SYSTEM_PROMPT}

Respond in: {language}

Conversation History:
{history_text}

Document Context:
{context}

Question:
{query}

Answer:
"""

    try:

        response = model.generate_content(
            prompt
        )

        # Handle empty response
        if not response:
            return "⚠️ No response generated."

        # Handle blocked/empty text
        if not hasattr(response, "text"):
            return "⚠️ Response blocked or unavailable."

        answer = response.text.strip()

        if not answer:
            return "⚠️ Empty response generated."

        return answer

    except Exception as e:

        return f"⚠️ Gemini Error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# Generate Summary
# ─────────────────────────────────────────────────────────────
def generate_summary(
    documents_text: str,
    filename: str,
) -> str:
    """
    Generate concise AI summary for document.
    """

    prompt = f"""
Analyze and summarize this document.

Filename:
{filename}

Provide:
1. Main topic
2. Key points
3. Important conclusions
4. Suggested questions

Document:
{documents_text[:4000]}
"""

    try:

        response = model.generate_content(
            prompt
        )

        if hasattr(response, "text"):
            return response.text.strip()

        return "Summary unavailable."

    except Exception:

        return "Summary unavailable."


# ─────────────────────────────────────────────────────────────
# Extract Keywords
# ─────────────────────────────────────────────────────────────
def extract_keywords(
    text: str
) -> List[str]:
    """
    Extract important keywords and phrases.
    """

    prompt = f"""
Extract the 10 most important keywords
and key phrases from this text.

Return ONLY a comma-separated list.

Text:
{text[:2500]}
"""

    try:

        response = model.generate_content(
            prompt
        )

        if not hasattr(response, "text"):
            return []

        raw = response.text.strip()

        keywords = [
            k.strip()
            for k in raw.split(",")
            if k.strip()
        ]

        # Remove duplicates
        keywords = list(dict.fromkeys(keywords))

        return keywords[:10]

    except Exception:

        return []


# ─────────────────────────────────────────────────────────────
# Transcribe Audio
# ─────────────────────────────────────────────────────────────
def transcribe_audio(
    audio_bytes: bytes,
) -> str:
    """
    Transcribe audio bytes to text using Gemini.
    """
    prompt = "Transcribe the following audio accurately. Return ONLY the transcribed text without any formatting, quotes, or markdown."
    
    try:
        response = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_bytes},
            prompt
        ])
        
        if hasattr(response, "text"):
            return response.text.strip()
            
        return "⚠️ Could not transcribe audio."
        
    except Exception as e:
        return f"⚠️ Audio Transcription Error: {str(e)}"