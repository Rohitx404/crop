import json
import os

import streamlit as st
from google import genai
from google.genai import types

def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def _models():
    """Optional GEMINI_MODEL secret first, then fallbacks tried in order."""
    custom = _secret("GEMINI_MODEL")
    return ([custom] if custom else []) + ["gemini-flash-latest", "gemini-2.5-flash", "gemini-3.5-flash"]


def _api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def gemini_available():
    return bool(_api_key())


def get_advice(image, label, confidence, language="Hindi"):
    """image: PIL image. label: e.g. 'Tomato - Late Blight'. Returns a dict, or {'error': ...}."""
    prompt = f"""You are an agriculture expert helping Indian farmers.
A CNN model looked at this leaf photo and predicted: {label} (confidence {confidence:.0f}%).

1. Check whether the photo really shows a plant leaf, and whether the CNN prediction looks correct.
2. Explain the condition in simple words. If the plant is healthy, give care tips instead of treatment.
3. Give practical organic and chemical treatment options with timing, and prevention steps.

Write all text values in {language}. Return JSON with exactly these keys:
is_leaf (boolean), agrees_with_cnn (boolean), explanation (string), severity (string),
organic_treatment (string), chemical_treatment (string), prevention (string), warning (string)."""
    last_error = "unknown error"
    try:
        client = genai.Client(api_key=_api_key())
    except Exception as e:
        return {"error": str(e)}
    for model in _models():
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = str(e)
    return {"error": last_error}
