import os
import re
import json
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import spacy
from pathlib import Path
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_trf")
except:
    logging.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class DynamicTopicClassifier:
    def __init__(self, topics_file="topics_database.json"):
        """Initialize the dynamic topic classifier"""
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.academic_subjects = []  # Track academic subjects

        # For TF-IDF approach
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 3), min_df=2
        )

        # For topic modeling
        self.nmf_model = NMF(n_components=10, random_state=42)
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)

        # Topic database
        self.topics_file = topics_file
        self.topics_db = self.load_topics_database()

        logging.info(
            f"Initialized classifier with {len(self.topics_db['topics'])} existing topics"
        )

    def load_topics_database(self):
        """Load the topics database from file or create a new one"""
        if os.path.exists(self.topics_file):
            with open(self.topics_file, "r") as f:
                return json.load(f)
        else:
            # Initialize with empty structure
            topics_db = {"topics": {}, "hierarchies": {}, "documents": []}
            return topics_db

    def save_topics_database(self):
        """Save the topics database to file"""
        with open(self.topics_file, "w") as f:
            json.dump(self.topics_db, f, indent=2)

    def initialize_academic_subjects(self):
        """Initialize core academic subjects taxonomy"""
        academic_subjects = {
            "Computer Science": [
                "algorithm",
                "programming",
                "artificial intelligence",
                "data structures",
                "parsing",
                "stack",
                "compiler",
                "machine learning",
                "computer vision",
                "networking",
                "syntax",
                "grammar",
                "reduce",
                "shift",
                "precedence",
                "automata",
                "turing machine",
                "context-free",
                "lexical",
                "semantic",
                "pipeline",
                "optimization",
                "runtime",
            ],
            "Electrical Engineering": [
                "circuit",
                "voltage",
                "signal processing",
                "electronics",
                "power systems",
                "transistor",
                "semiconductor",
                "control systems",
            ],
            "Mechanical Engineering": [
                "thermodynamics",
                "fluid mechanics",
                "robotics",
                "manufacturing",
                "materials science",
                "cad",
            ],
        }

        # Add subjects to classifier
        for subject, keywords in academic_subjects.items():
            if subject not in self.topics_db["topics"]:
                self.add_topic(subject, keywords=keywords)
                self.academic_subjects.append(subject)

        logging.info("Initialized academic subject taxonomy")

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r"[^\w\s]", " ", text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]
        return " ".join(tokens)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file"""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return text

    def extract_keywords(self, text, n=10):
        """Extract keywords from text using spaCy"""
        try:
            # Limit text length to avoid memory issues
            doc = nlp(text[:20000])

            # Extract noun phrases as potential keywords
            keywords = []
            for chunk in doc.noun_chunks:
                if 1 <= len(chunk.text.split()) <= 3:  # Phrases with 1-3 words
                    clean_text = re.sub(r"[^\w\s]", "", chunk.text.lower())
                    if clean_text and len(clean_text) > 2:
                        keywords.append(clean_text)
            keywords = [
                " ".join(
                    [word for word in phrase.split() if word not in self.stop_words]
                )
                for phrase in keywords
            ]
            # Extract important entities
            for ent in doc.ents:
                if ent.label_ in [
                    "ORG",
                    "PRODUCT",
                    "WORK_OF_ART",
                    "EVENT",
                    "LAW",
                    "LANGUAGE",
                ]:
                    clean_text = re.sub(r"[^\w\s]", "", ent.text.lower())
                    if clean_text and len(clean_text) > 2:
                        keywords.append(clean_text)

            # Count frequencies and return top n
            keyword_freq = {}
            for kw in keywords:
                if kw not in self.stop_words and len(kw) > 2:
                    keyword_freq[kw] = keyword_freq.get(kw, 0) + 1

            sorted_keywords = sorted(
                keyword_freq.items(), key=lambda x: x[1], reverse=True
            )
            return [kw for kw, _ in sorted_keywords[:n]]

        except Exception as e:
            logging.error(f"Error extracting keywords: {str(e)}")
            # Simple fallback - extract words by frequency
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word = re.sub(r"[^\w]", "", word)
                if word and word not in self.stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1

            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:n]]

    def generate_topics(self, text):
        """Generate topics using NMF and LDA"""

        try:
            # Create temporary vectorizer adjusted for single documents
            temp_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 3),
                min_df=1,  # Adjusted for single documents
                max_df=1.0,
            )

            # Vectorize the text
            vectorized_text = temp_vectorizer.fit_transform([text])

            # NMF topic modeling
            nmf_model = NMF(n_components=5, random_state=42)  # Reduced components
            nmf_topics = nmf_model.fit_transform(vectorized_text)
            nmf_components = nmf_model.components_

            # LDA topic modeling
            lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
            lda_topics = lda_model.fit_transform(vectorized_text)
            lda_components = lda_model.components_

            # Extract top words for each topic
            feature_names = temp_vectorizer.get_feature_names_out()

            nmf_topic_words = []
            for topic in nmf_components:
                top_features_ind = topic.argsort()[:-6:-1]  # Top 5 words
                nmf_topic_words.append([feature_names[i] for i in top_features_ind])

            lda_topic_words = []
            for topic in lda_components:
                top_features_ind = topic.argsort()[:-6:-1]
                lda_topic_words.append([feature_names[i] for i in top_features_ind])

            return nmf_topic_words, lda_topic_words

        except Exception as e:
            logging.error(f"Error generating topics: {str(e)}")
            return [], []

    def find_similar_topics(self, keywords):
        """Find similar topics in the database based on keyword overlap"""
        similar_topics = []

        # Check keywords against existing topics
        for topic, data in self.topics_db["topics"].items():
            # Check for keyword overlap
            topic_keywords = data.get("keywords", [])
            overlap = set(keywords).intersection(set(topic_keywords))

            if overlap:
                similarity = len(overlap) / max(len(keywords), len(topic_keywords))
                similar_topics.append((topic, similarity))

        # Sort by similarity score
        return sorted(similar_topics, key=lambda x: x[1], reverse=True)

    def classify_document(self, file_path):
        """Classify a document and return potential topics"""
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return {"error": "File not found"}

        logging.info(f"Processing document: {file_path}")

        # Extract text from PDF
        text = self.extract_text_from_pdf(file_path)
        if not text:
            logging.error(f"Could not extract text from: {file_path}")
            return {"error": "Could not extract text from document"}

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Extract keywords
        keywords = self.extract_keywords(text)
        logging.info(f"Extracted {len(keywords)} keywords")

        # Generate potential topics
        nmf_topics, lda_topics = self.generate_topics(processed_text)

        # Find similar existing topics
        similar_topics = self.find_similar_topics(keywords)

        # Separate academic subjects and dynamic topics
        academic_suggestions = []
        dynamic_suggestions = []
        for topic, score in similar_topics:
            if score > 0.1:  # Minimum similarity threshold
                if topic in self.academic_subjects:
                    academic_suggestions.append(
                        {
                            "name": topic,
                            "similarity": score,
                            "keywords": self.topics_db["topics"][topic]["keywords"][:5],
                        }
                    )
                else:
                    dynamic_suggestions.append(
                        {
                            "name": topic,
                            "similarity": score,
                            "keywords": self.topics_db["topics"][topic]["keywords"][:5],
                        }
                    )

        # Generate potential new topics from NMF and LDA
        potential_topics = []

        # From NMF
        for i, topic_words in enumerate(nmf_topics):
            if topic_words:  # Skip empty topics
                potential_topics.append(
                    {
                        "name": f"Topic {i + 1}",
                        "keywords": topic_words[:5],
                        "source": "NMF",
                    }
                )

        # From LDA
        for i, topic_words in enumerate(lda_topics):
            if topic_words:  # Skip empty topics
                potential_topics.append(
                    {
                        "name": f"Topic {i + 1}",
                        "keywords": topic_words[:5],
                        "source": "LDA",
                    }
                )

        return {
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "keywords": keywords,
            "suggested_topics": dynamic_suggestions,
            "academic_subjects": academic_suggestions,
            "potential_topics": potential_topics,
            "text_preview": text[:500] + "..." if len(text) > 500 else text,
        }

    def add_topic(self, topic_name, keywords=None, parent_topic=None):
        """Add a new topic to the database"""
        if topic_name in self.topics_db["topics"]:
            logging.warning(f"Topic already exists: {topic_name}")
            return False

        logging.info(f"Adding new topic: {topic_name}")
        self.topics_db["topics"][topic_name] = {
            "keywords": keywords or [],
            "documents": [],
            "created_at": datetime.now().isoformat(),
        }

        # Add hierarchy relationship if parent topic is provided
        if parent_topic:
            if parent_topic not in self.topics_db["hierarchies"]:
                self.topics_db["hierarchies"][parent_topic] = []

            if topic_name not in self.topics_db["hierarchies"][parent_topic]:
                self.topics_db["hierarchies"][parent_topic].append(topic_name)

        self.save_topics_database()
        return True

    def add_document_to_topic(self, doc_id, topic_name, file_path, keywords):
        """Add a document to a topic"""
        # Ensure topic exists
        if topic_name not in self.topics_db["topics"]:
            logging.warning(f"Topic does not exist: {topic_name}")
            return False

        # Create document entry if it doesn't exist
        doc_exists = False
        for doc in self.topics_db["documents"]:
            if doc["id"] == doc_id:
                doc_exists = True
                if topic_name not in doc["topics"]:
                    doc["topics"].append(topic_name)
                break

        if not doc_exists:
            self.topics_db["documents"].append(
                {
                    "id": doc_id,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "topics": [topic_name],
                    "keywords": keywords,
                    "added_at": datetime.now().isoformat(),
                }
            )

        # Add document to topic
        if doc_id not in self.topics_db["topics"][topic_name]["documents"]:
            self.topics_db["topics"][topic_name]["documents"].append(doc_id)

            # Update topic keywords based on document keywords
            self.topics_db["topics"][topic_name]["keywords"] = list(
                set(self.topics_db["topics"][topic_name]["keywords"]) | set(keywords)
            )

        self.save_topics_database()
        logging.info(f"Added document {doc_id} to topic: {topic_name}")
        return True

    def process_pdf(self, pdf_path, auto_classify=False, auto_create_topics=False):
        """Process a PDF document and classify"""
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        result = self.classify_document(pdf_path)
        if "error" in result:
            return result

        actions_taken = []

        # Auto-assign to existing topics (academic or dynamic)
        if auto_classify:
            for suggestion in result["suggested_topics"] + result["academic_subjects"]:
                if suggestion["similarity"] > 0.2:
                    self.add_document_to_topic(
                        doc_id, suggestion["name"], pdf_path, result["keywords"]
                    )
                    actions_taken.append(
                        f"Auto-assigned to topic: {suggestion['name']}"
                    )

        # Auto-create new topics
        if auto_create_topics and result["potential_topics"]:
            for potential_topic in result["potential_topics"]:
                if potential_topic["source"] == "NMF":
                    topic_name = "_".join(potential_topic["keywords"][:2])
                    topic_name = re.sub(r"\W+", "_", topic_name)

                    if topic_name not in self.topics_db["topics"]:
                        self.add_topic(topic_name, potential_topic["keywords"])
                        self.add_document_to_topic(
                            doc_id, topic_name, pdf_path, result["keywords"]
                        )
                        actions_taken.append(
                            f"Auto-created and assigned to new topic: {topic_name}"
                        )
                    break

        result["doc_id"] = doc_id
        result["actions_taken"] = actions_taken
        return result

    def get_topic_hierarchy(self):
        """Get the full topic hierarchy"""
        hierarchy = {}

        # Get all root topics (those without parents)
        root_topics = set(self.topics_db["topics"].keys())
        for parent, children in self.topics_db["hierarchies"].items():
            for child in children:
                if child in root_topics:
                    root_topics.remove(child)

        # Build hierarchy starting from root topics
        def build_hierarchy(topic):
            result = {"name": topic, "children": []}
            if topic in self.topics_db["hierarchies"]:
                for child in self.topics_db["hierarchies"][topic]:
                    result["children"].append(build_hierarchy(child))
            return result

        # Create hierarchy for each root topic
        for topic in root_topics:
            hierarchy[topic] = build_hierarchy(topic)

        return hierarchy


def process_directory(
    directory_path, classifier=None, auto_classify=False, auto_create_topics=False
):
    """Process all PDF files in a directory"""
    if classifier is None:
        classifier = DynamicTopicClassifier()

    results = []
    pdf_files = list(Path(directory_path).glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files in {directory_path}")

    for pdf_path in pdf_files:
        result = classifier.process_pdf(
            str(pdf_path),
            auto_classify=auto_classify,
            auto_create_topics=auto_create_topics,
        )
        results.append(result)

    return results


def print_classification_summary(results):
    """Print a summary of classification results"""
    print("\n===== Classification Summary =====")

    for result in results:
        if "error" in result:
            print(
                f"Error processing {result.get('file', 'unknown file')}: {result['error']}"
            )
            continue

        print(f"\nFile: {result['file']}")
        print(f"Top Keywords: {', '.join(result['keywords'][:5])}")

        if result["academic_subjects"]:
            print("Academic Subject Suggestions:")
            for subject in result["academic_subjects"][:3]:
                print(
                    f"  - {subject['name']} (confidence: {subject['similarity']:.2f})"
                )

        if result["suggested_topics"]:
            print("Dynamic Topic Suggestions:")
            for topic in result["suggested_topics"][:3]:
                print(f"  - {topic['name']} (similarity: {topic['similarity']:.2f})")

        if result["actions_taken"]:
            print("Actions Taken:")
            for action in result["actions_taken"]:
                print(f"  - {action}")


def main():
    # Create and initialize classifier
    classifier = DynamicTopicClassifier()
    classifier.initialize_academic_subjects()

    # Configure argument parsing
    import argparse

    parser = argparse.ArgumentParser(description="Classify a PDF document")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to process")
    args = parser.parse_args()

    # Process the single PDF
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: File not found - {pdf_path}")
        return

    print(f"\n{'=' * 40}")
    print(f"Processing document: {pdf_path}")
    result = classifier.process_pdf(
        pdf_path, auto_classify=True, auto_create_topics=False
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    # Print results
    print(f"\nDocument: {result['file']}")
    print(f"\nTop Keywords: {', '.join(result['keywords'][:5])}")

    if result["academic_subjects"]:
        print("\nAcademic Subject Suggestions:")
        for subject in result["academic_subjects"][:3]:
            print(f"- {subject['name']} (confidence: {subject['similarity']:.2f})")

    if result["suggested_topics"]:
        print("\nDynamic Topic Suggestions:")
        for topic in result["suggested_topics"][:3]:
            print(f"- {topic['name']} (similarity: {topic['similarity']:.2f})")

    # Print current hierarchy
    def print_hierarchy(hierarchy_item, level=0):
        print(f"{'  ' * level}└─ {hierarchy_item['name']}")
        for child in hierarchy_item.get("children", []):
            print_hierarchy(child, level + 1)

    hierarchy = classifier.get_topic_hierarchy()
    print("\nCurrent Topic Hierarchy:")
    for topic, structure in hierarchy.items():
        print_hierarchy(structure)


if __name__ == "__main__":
    main()
