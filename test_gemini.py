import os
import sys
from google import genai
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key or not api_key.strip():
    print("[WARNING] GEMINI_API_KEY is not set.")
    print("Please create a .env file with GEMINI_API_KEY=your_key or set the environment variable.")
    sys.exit(1)

try:
    client = genai.Client(api_key=api_key.strip())
    print("\nAVAILABLE MODELS:\n")
    found_any = False
    for model in client.models.list():
        print(f" - {model.name}")
        found_any = True
    if not found_any:
        print("No models returned.")
    else:
        print("\n[SUCCESS] Successfully connected to Gemini API.")
except Exception as e:
    print(f"\n[ERROR] Error connecting to Gemini API: {str(e)}")
    sys.exit(1)