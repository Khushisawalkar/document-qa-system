import os
from typing import Optional, List, Tuple, Generator
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key from .env file if present
load_dotenv()

# Ordered preference of valid Gemini models
SUPPORTED_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"]

# Client & model cache for performance
_CLIENT_CACHE: dict = {}
_LAST_WORKING_MODEL: Optional[str] = None


def get_client(api_key: Optional[str] = None) -> Optional[genai.Client]:
    """
    Get or initialize a cached Gemini API client dynamically.
    Reuses existing client instances to avoid redundant connection setup.
    """
    key = api_key
    if not key:
        # Check Streamlit session or secrets if available
        try:
            import streamlit as st
            key = st.session_state.get("gemini_api_key")
            if not key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not key:
        key = os.getenv("GEMINI_API_KEY")

    if not key or not key.strip():
        return None

    clean_key = key.strip()
    if clean_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[clean_key]

    try:
        client = genai.Client(api_key=clean_key)
        _CLIENT_CACHE[clean_key] = client
        return client
    except Exception:
        return None


def _get_ordered_models() -> List[str]:
    """Return models with the last successfully used model prioritized first."""
    global _LAST_WORKING_MODEL
    if _LAST_WORKING_MODEL and _LAST_WORKING_MODEL in SUPPORTED_MODELS:
        return [_LAST_WORKING_MODEL] + [m for m in SUPPORTED_MODELS if m != _LAST_WORKING_MODEL]
    return SUPPORTED_MODELS


def _generate_with_fallback(client: genai.Client, contents, **kwargs):
    """
    Generate content with automatic fallback across supported Gemini models.
    Remembers the working model to prevent repeated fallback delays.
    """
    global _LAST_WORKING_MODEL
    last_error = None
    
    for model_name in _get_ordered_models():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                **kwargs
            )
            _LAST_WORKING_MODEL = model_name
            return response
        except Exception as e:
            last_error = e
            continue
            
    raise last_error


def _stream_with_fallback(client: genai.Client, contents, **kwargs):
    """
    Stream content with automatic fallback across supported Gemini models.
    """
    global _LAST_WORKING_MODEL
    last_error = None

    for model_name in _get_ordered_models():
        try:
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                **kwargs
            )
            _LAST_WORKING_MODEL = model_name
            return stream
        except Exception as e:
            last_error = e
            continue

    raise last_error


def build_grounded_rag_prompt(
    query: str,
    context: str,
    chat_history: list = None,
    language: str = "English"
) -> str:
    """Build a strictly grounded prompt with inline citation requirements."""
    history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_text += f"{role}: {turn.get('content', '')}\n"

    return f"""You are DocMind, a rigorous and accurate AI Document QA Assistant.
Your task is to answer the user's question with precision, strictly grounded ONLY in the provided Document Context.

CORE INSTRUCTIONS:
1. Strict Grounding: Rely strictly on the clear facts stated in the Document Context. Do NOT extrapolate or introduce external facts not supported by the sources.
2. Inline Citations: Cite your sources inline using bracketed tags corresponding to the source index, such as [1], [2], immediately after the relevant sentence or claim (e.g., "...according to the report [1].").
3. Insufficient Information: If the context does not contain sufficient details to answer the question, state:
   "I couldn't find this information in the uploaded documents."
   Do NOT attempt to guess or invent an answer.
4. Structure: Provide a clear, well-structured response using bullet points or concise paragraphs where helpful.
5. Language: Respond in {language}.

Document Context:
{context}

Conversation History:
{history_text}

User Question:
{query}

Answer:"""


def generate_answer(
    query: str, 
    context: str, 
    chat_history: list = None, 
    language: str = "English",
    api_key: Optional[str] = None
) -> str:
    """
    Generate a complete answer using Gemini based ONLY on the provided document context.
    """
    client = get_client(api_key)
    if not client:
        return "⚠️ Gemini API key is missing. Please add it to your .env file or enter it in the sidebar."
        
    if not context or not context.strip():
        return "I couldn't find relevant information in the uploaded documents."

    prompt = build_grounded_rag_prompt(query, context, chat_history, language)

    try:
        response = _generate_with_fallback(client, contents=prompt)
        return response.text.strip() if (response and response.text) else "No response generated."
    except Exception as e:
        return f"Gemini API Error: {str(e)}"


def generate_answer_stream(
    query: str,
    context: str,
    chat_history: list = None,
    language: str = "English",
    api_key: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Real-time streaming generation using Gemini API with inline citation grounding.
    Yields chunks as they are generated by the model.
    """
    client = get_client(api_key)
    if not client:
        yield "⚠️ Gemini API key is missing. Please add it to your .env file or enter it in the sidebar."
        return

    if not context or not context.strip():
        yield "I couldn't find relevant information in the uploaded documents."
        return

    prompt = build_grounded_rag_prompt(query, context, chat_history, language)

    try:
        stream = _stream_with_fallback(client, contents=prompt)
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Gemini API Error: {str(e)}"


def generate_summary_and_keywords(
    documents_text: str, 
    filename: str, 
    api_key: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Single combined Gemini call to extract both summary and keywords in parallel,
    reducing document processing API overhead and latency by 50%.
    """
    client = get_client(api_key)
    if not client:
        return ("⚠️ Gemini API key is missing. Please add it to your .env file or enter it in the sidebar.", [])

    if not documents_text or not documents_text.strip():
        return ("No text available to summarize.", [])

    prompt = f"""Analyze the following document and output two sections exactly as formatted below:

---SUMMARY---
A concise 3-4 bullet point summary covering the main topic and essential findings.

---KEYWORDS---
A single line with up to 10 most important keywords separated by commas.

Filename: {filename}
Document Text:
{documents_text[:4000]}"""

    try:
        response = _generate_with_fallback(client, contents=prompt)
        text = response.text.strip() if (response and response.text) else ""
        
        summary = "Summary unavailable."
        keywords = []
        
        if "---SUMMARY---" in text and "---KEYWORDS---" in text:
            parts = text.split("---KEYWORDS---")
            summary_part = parts[0].replace("---SUMMARY---", "").strip()
            keywords_part = parts[1].strip() if len(parts) > 1 else ""
            
            if summary_part:
                summary = summary_part
            if keywords_part:
                keywords = [k.strip() for k in keywords_part.split(",") if k.strip()][:10]
        elif text:
            summary = text
            
        return (summary, keywords)
    except Exception as e:
        return (f"Summary unavailable ({str(e)}).", [])


def generate_summary(documents_text: str, filename: str, api_key: Optional[str] = None) -> str:
    """Generate a short summary of the document (fallback/standalone)."""
    summary, _ = generate_summary_and_keywords(documents_text, filename, api_key=api_key)
    return summary


def extract_keywords(text: str, api_key: Optional[str] = None) -> List[str]:
    """Extract up to 10 important keywords from the text (fallback/standalone)."""
    _, keywords = generate_summary_and_keywords(text, "document", api_key=api_key)
    return keywords


def transcribe_audio(
    audio_bytes: bytes, 
    mime_type: str = "audio/wav", 
    api_key: Optional[str] = None
) -> str:
    """
    Transcribe audio bytes to text using Gemini.
    """
    client = get_client(api_key)
    if not client:
        return "⚠️ Gemini API key is missing. Please add it to your .env file or enter it in the sidebar."

    if not audio_bytes:
        return "No audio data received."
        
    prompt = "Transcribe the following audio exactly as spoken."
    valid_mime = mime_type if mime_type and "/" in mime_type else "audio/wav"

    try:
        contents = [
            types.Part.from_bytes(data=audio_bytes, mime_type=valid_mime),
            prompt
        ]
        response = _generate_with_fallback(client, contents=contents)
        return response.text.strip() if (response and response.text) else "Could not transcribe audio."
    except Exception as e:
        return f"Audio Transcription Error: {str(e)}"