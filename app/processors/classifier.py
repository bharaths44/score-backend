import logging

from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict

from ..processors.keywords import KEYWORDS

logger = logging.getLogger(__name__)


# Pre-defined keywords for tagging


# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute category embeddings
CATEGORY_EMBEDDINGS = {
    cat: embedding_model.encode([cat] + keywords)[0]
    for cat, keywords in KEYWORDS.items()
}

SUBCATEGORY_EMBEDDINGS = {
    cat: {kw: embedding_model.encode(kw) for kw in keywords}
    for cat, keywords in KEYWORDS.items()
}


def classify_document(text: str, file_path: str) -> Dict[str, Any]:
    """Classify document into predefined categories with subtopics."""
    try:
        # Text preprocessing
        clean_text = " ".join(text.split()[:1000])

        # Generate document embedding
        doc_embedding = embedding_model.encode(clean_text)

        # Calculate category similarities
        category_scores = {}
        for cat, cat_emb in CATEGORY_EMBEDDINGS.items():
            similarity = cosine_similarity([doc_embedding], [cat_emb])[0][0]
            keyword_count = sum(
                1 for kw in KEYWORDS[cat] if kw.lower() in clean_text.lower()
            )
            category_scores[cat] = similarity + 0.2 * keyword_count  # Weighted score

        # Normalize scores
        total = sum(category_scores.values())
        if total == 0:
            return {"categories": [], "topics": []}
        normalized_scores = {k: v / total for k, v in category_scores.items()}

        # Detect subtopics for top category
        main_category = max(normalized_scores, key=normalized_scores.get)

        # Prepare results with hierarchical structure and collect all topics
        results = []
        all_topics = []

        for cat, score in normalized_scores.items():
            if score > 0.60:
                subtopics = (
                    detect_subtopics(clean_text, cat) if cat == main_category else []
                )
                filtered_subtopics = [
                    st["subtopic"] for st in subtopics if st["score"] > 0.60
                ]

                entry = {
                    "category": cat,
                    "score": round(score, 2),
                    "subtopics": subtopics,
                }
                results.append(entry)

                # Add to topics list
                all_topics.append(cat)
                all_topics.extend(filtered_subtopics)

        # Sort by descending score
        results.sort(key=lambda x: x["score"], reverse=True)

        # If no categories have score > 0.5, return just the main category
        if not results:
            main_cat = max(normalized_scores, key=normalized_scores.get)
            main_score = normalized_scores[main_cat]
            subtopics = detect_subtopics(clean_text, main_cat)
            filtered_subtopics = [
                st["subtopic"] for st in subtopics if st["score"] > 0.60
            ]

            results.append(
                {
                    "category": main_cat,
                    "score": round(main_score, 2),
                    "subtopics": subtopics,
                }
            )

            all_topics = [main_cat] + filtered_subtopics
            logger.info(
                f"No categories with score > 0.5, returning only main category: {main_cat}"
            )

        logger.info(f"Document {file_path} classified with topics: {all_topics}")

        return {"categories": results, "topics": all_topics}

    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        return {"categories": [], "topics": []}


def detect_subtopics(text: str, category: str) -> List[Dict[str, float]]:
    """Detect specific subtopics within a category."""
    try:
        # Get relevant subcategory embeddings
        subcats = SUBCATEGORY_EMBEDDINGS[category]

        # Calculate keyword presence and similarity
        scores = defaultdict(float)
        for kw, emb in subcats.items():
            # Keyword presence score
            presence = 1.0 if kw.lower() in text.lower() else 0.0
            # Semantic similarity score
            similarity = cosine_similarity([embedding_model.encode(text)], [emb])[0][0]
            scores[kw] = (0.6 * presence) + (0.4 * similarity)

        # Return top 5 subtopics
        return sorted(
            [{"subtopic": k, "score": round(v, 2)} for k, v in scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )[:5]

    except Exception as e:
        logger.warning(f"Subtopic detection failed: {str(e)}")
        return []
