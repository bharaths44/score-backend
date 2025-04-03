import spacy
import string
from typing import Dict, List

# Load spaCy models
nlp = spacy.load("en_core_web_trf")

# Expanded topic keywords with more domains
topic_keywords = {
    "computer science": [
        "algorithm",
        "programming",
        "database",
        "software",
        "machine learning",
        "data structure",
        "network",
        "security",
        "artificial intelligence",
        "operating system",
        "web development",
        "cloud computing",
        "python",
        "java",
        "javascript",
        "coding",
        "compiler",
        "api",
        "framework",
        "encryption",
        "cybersecurity",
        "data mining",
        "big data",
    ],
    "mathematics": [
        "calculus",
        "algebra",
        "geometry",
        "statistics",
        "probability",
        "equation",
        "theorem",
        "mathematical",
        "numerical",
        "linear algebra",
        "differential",
        "integral",
        "matrix",
        "vector",
        "polynomial",
        "arithmetic",
        "trigonometry",
        "logarithm",
        "function",
        "graph theory",
        "set theory",
        "optimization",
        "discrete mathematics",
    ],
    "physics": [
        "mechanics",
        "quantum",
        "thermodynamics",
        "electricity",
        "magnetism",
        "relativity",
        "wave",
        "particle",
        "force",
        "energy",
        "momentum",
        "gravity",
        "electromagnetic",
        "nuclear",
        "optics",
        "fluid dynamics",
        "quantum mechanics",
        "string theory",
        "dark matter",
        "plasma",
        "acceleration",
        "velocity",
        "mass",
        "charge",
    ],
    "engineering": [
        "design",
        "circuit",
        "mechanical",
        "electrical",
        "construction",
        "system",
        "material",
        "process",
        "control",
        "robotics",
        "automation",
        "manufacturing",
        "CAD",
        "civil engineering",
        "electronics",
        "semiconductor",
        "VLSI",
        "microprocessor",
        "embedded systems",
        "signal processing",
        "power systems",
        "thermodynamics",
    ],
    "biology": [
        "cell",
        "genetics",
        "organism",
        "evolution",
        "molecular",
        "ecosystem",
        "physiology",
        "biochemistry",
        "anatomy",
        "DNA",
        "protein",
        "enzyme",
        "chromosome",
        "mutation",
        "natural selection",
        "biodiversity",
        "microbiology",
        "botany",
        "zoology",
        "ecology",
        "immunology",
        "neuroscience",
        "biotechnology",
    ],
    "chemistry": [
        "chemical",
        "reaction",
        "molecule",
        "atom",
        "bond",
        "organic",
        "inorganic",
        "polymer",
        "acid",
        "base",
        "catalyst",
        "solution",
        "compound",
        "element",
        "periodic table",
        "electrochemistry",
        "thermochemistry",
        "biochemistry",
        "analytical",
        "physical chemistry",
        "quantum chemistry",
    ],
    "medicine": [
        "diagnosis",
        "treatment",
        "pathology",
        "anatomy",
        "physiology",
        "surgery",
        "pharmaceutical",
        "clinical",
        "patient",
        "disease",
        "therapy",
        "medicine",
        "drug",
        "vaccine",
        "immunology",
        "cardiology",
        "neurology",
        "oncology",
        "pediatrics",
        "psychiatry",
    ],
}


def is_valid_keyword(text: str) -> bool:
    """
    Check if a keyword is valid (not just punctuation or single characters).
    """
    # Remove whitespace
    text = text.strip()

    # Check minimum length
    if len(text) <= 1:
        return False

    # Check if it's not just punctuation
    if all(char in string.punctuation for char in text):
        return False

    # Check if it's not just numbers
    if text.isdigit():
        return False

    return True


def extract_keywords_spacy(text: str, top_n: int = 10) -> List[Dict]:
    """
    Extract keywords from text using spaCy with improved filtering.
    """
    doc = nlp(text)
    keywords = set()

    # Extract noun chunks and named entities
    for chunk in doc.noun_chunks:
        if is_valid_keyword(chunk.text):
            keywords.add(chunk.text.lower())

    for ent in doc.ents:
        if is_valid_keyword(ent.text):
            keywords.add(ent.text.lower())

    # Calculate frequency and confidence
    keyword_freq = {word: text.lower().count(word) for word in keywords}
    if not keyword_freq:
        return []

    max_freq = max(keyword_freq.values())
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)

    return [{"name": k, "confidence": v / max_freq} for k, v in sorted_keywords[:top_n]]
