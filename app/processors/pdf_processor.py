import logging
import re

import fitz  # PyMuPDF
import spacy


logger = logging.getLogger(__name__)

# Load spaCy model

nlp = spacy.load("en_core_web_sm")
logger.info("SpaCy model loaded successfully for text preprocessing")


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            text += doc[page_num].get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""


def preprocess_text(text: str, options: dict = None) -> str:
    """
    Preprocess text by:
    - Removing unwanted characters
    - Removing stop words
    - Lemmatizing words
    - Removing named entities (people, places, organizations)

    Args:
        text: Input text to preprocess
        options: Dictionary of preprocessing options:
            - remove_stopwords: Whether to remove stop words (default: True)
            - lemmatize: Whether to lemmatize words (default: True)
            - remove_entities: Whether to remove named entities (default: True)
            - entity_types: List of entity types to remove (default: ["PERSON", "GPE", "LOC", "ORG"])
            - min_word_length: Minimum word length to keep (default: 2)

    Returns:
        Preprocessed text
    """
    if not text:
        return ""

    # Default options
    default_options = {
        "remove_stopwords": True,
        "lemmatize": True,
        "remove_entities": False,
        "entity_types": ["PERSON", "GPE", "LOC", "ORG"],
        "min_word_length": 2,
    }

    # Use provided options or defaults
    options = options or default_options

    try:
        # Initial basic cleaning
        text = text.lower()
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        text = re.sub(r"[^\w\s]", " ", text)  # Replace punctuation with space

        # Process with spaCy
        doc = nlp(text)

        # Identify entities to remove
        entities_to_remove = []
        if options.get("remove_entities", True):
            entity_types = options.get("entity_types", ["PERSON", "GPE", "LOC", "ORG"])
            for ent in doc.ents:
                if ent.label_ in entity_types:
                    entities_to_remove.append((ent.start, ent.end))

        # Create filtered tokens
        result_tokens = []
        for i, token in enumerate(doc):
            # Skip if token is part of an entity to be removed
            if any(start <= token.i < end for start, end in entities_to_remove):
                continue

            # Apply filters
            if (
                (not options.get("remove_stopwords", True) or not token.is_stop)
                and not token.is_punct
                and not token.is_space
                and len(token.text) >= options.get("min_word_length", 2)
            ):
                # Lemmatize if requested
                if options.get("lemmatize", True) and token.lemma_ != "-PRON-":
                    result_tokens.append(token.lemma_)
                else:
                    result_tokens.append(token.text)

        # Join tokens back into text
        result = " ".join(result_tokens)

        logger.info(
            f"Text preprocessed: removed {len(doc) - len(result_tokens)} tokens"
        )
        return result

    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text  # Return original text if processing fails


def extract_and_preprocess(file_path: str, preprocess_options: dict = None) -> str:
    """
    Extract text from PDF and preprocess it.

    Args:
        file_path: Path to the PDF file
        preprocess_options: Options for preprocessing

    Returns:
        Preprocessed text
    """
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        return ""

    return preprocess_text(raw_text, preprocess_options)
