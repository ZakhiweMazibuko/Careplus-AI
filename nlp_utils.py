# Install rapidfuzz: pip install rapidfuzz
from rapidfuzz import process

# Example synonyms dictionary
SYNONYMS = {
    "high temperature": "fever",
    "tummy pain": "abdominal pain",
    "stomach ache": "abdominal pain",
    "throat pain": "sore throat",
    "head pain": "headache",
    "nausea": "feeling sick"
}

def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace."""
    return " ".join(text.lower().strip().split())

def replace_synonyms(text: str) -> str:
    """Replace known synonyms."""
    for key, val in SYNONYMS.items():
        if key in text:
            text = text.replace(key, val)
    return text

def preprocess_input(text: str) -> str:
    """Full preprocessing pipeline for user input."""
    text = normalize_text(text)
    text = replace_synonyms(text)
    return text
