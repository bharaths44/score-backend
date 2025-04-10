import os
import logging
import tempfile
import requests
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from supabase import create_client, Client


from .models.model import ProcessingResult
from .processors.pdf_processor import extract_and_preprocess
from .processors.classifier import classify_document


# Define a base directory for PDF files
PDF_DIR = Path(
    os.environ.get("PDF_DIR", os.path.join(os.path.dirname(__file__), "..", "pdfs"))
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API", description="API for PDF processing with NLP"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


@app.get("/")
def read_root():
    return {"status": "active", "service": "Document Processing API"}


@app.post("/process_pdf")
async def process_pdf(
    storage_url: str = Query(..., description="Supabase storage URL to the PDF file"),
):
    """
    Process a PDF file from Supabase storage.

    - Accepts a Supabase storage URL
    - Downloads the PDF temporarily
    - Extracts text from PDF
    - Classifies the document
    - Returns analysis results
    """
    try:
        # Initialize result
        result = ProcessingResult(file_path=storage_url)

        # Validate URL format
        parsed_url = urlparse(storage_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid storage URL: {storage_url}. Please provide a valid URL.",
            )

        # Download file from URL to temporary location
        logger.info(f"Downloading PDF from storage URL: {storage_url}")

        try:
            # Download file
            response = requests.get(storage_url, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                temp_file.write(response.content)
                logger.info(f"Downloaded PDF to temporary file: {temp_path}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            raise HTTPException(
                status_code=502, detail=f"Failed to download PDF from storage: {str(e)}"
            )

        # Process the PDF file
        try:
            text = extract_and_preprocess(temp_path)
            result.text = text[:5000]  # Limit text size in the response
            logger.info(f"Extracted {len(text)} characters of text")

            if text:
                classification_result = classify_document(
                    text, os.path.basename(parsed_url.path)
                )
                result.classification = classification_result
                topics = classification_result.get("topics", [])

            # Update Supabase StudyMaterial table with appended keywords
            try:
                # First get existing keywords
                existing = (
                    supabase.table("study_materials")
                    .select("keywords")
                    .eq("pdf_url", storage_url)
                    .execute()
                )

                current_keywords = []
                if existing.data:
                    current_keywords = existing.data[0].get("keywords", []) or []

                # Combine existing and new topics, removing duplicates
                combined_keywords = list(set(current_keywords + topics))

                update_data = {
                    "keywords": combined_keywords,
                    "text": result.text,
                }
                # Perform atomic update
                update_result = (
                    supabase.table("study_materials")
                    .update(update_data)
                    .eq("pdf_url", storage_url)
                    .execute()
                )

                if len(update_result.data) == 0:
                    logger.warning(
                        f"No StudyMaterial found with pdf_url: {storage_url}"
                    )
                else:
                    logger.info(
                        f"Updated keywords for {storage_url}. New keywords: {combined_keywords}"
                    )

            except Exception as supabase_error:
                logger.error(f"Supabase update failed: {str(supabase_error)}")
                raise HTTPException(
                    status_code=500,
                    detail="Document processed but failed to update database",
                )
            # Clean up temporary file
            os.unlink(temp_path)

            return result.classification["topics"]

        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/search")
async def search_pdfs(
    query: str,
    search_type: str = "all",  # "all", "any", or "exact"
):
    """
    Search for PDFs based on text content using PostgreSQL full-text search

    Args:
        query: Search term
        search_type: Type of search - "all" (match all words), "any" (match any word), or "exact"

    Returns:
        List of matching documents
    """
    logger.info(f"Searching for: '{query}', search_type={search_type}")

    if not query or query.strip() == "":
        return {"data": [], "message": "Empty search query"}

    try:
        # Format the query based on search type
        formatted_query = format_search_query(query, search_type)

        # Use Postgres full-text search
        results = (
            supabase.table("study_materials")
            .select("id, title, pdf_url, text, keywords")
            .text_search("text", formatted_query)
            .execute()
        )

        logger.info(f"Text search found {len(results.data)} results")

        # If no results in text field, try keywords
        if len(results.data) == 0:
            logger.info("No results in text field, trying keywords")

            # Create a combined_text field that includes keywords for searching
            all_materials = (
                supabase.table("study_materials")
                .select("id, title, pdf_url, text, keywords")
                .execute()
            )

            # Filter manually based on keywords
            filtered_results = []
            for doc in all_materials.data:
                keywords = doc.get("keywords", [])
                if keywords and any(
                    term.lower() in " ".join(keywords).lower()
                    for term in query.lower().split()
                ):
                    filtered_results.append(doc)

            results = {"data": filtered_results}
            logger.info(f"Keyword search found {len(filtered_results)} results")

        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {"data": [], "error": str(e)}


def format_search_query(query: str, search_type: str = "all") -> str:
    """
    Format search query for PostgreSQL full-text search

    Args:
        query: Raw search query
        search_type: Type of search - "all", "any", or "exact"

    Returns:
        Formatted query string
    """
    # Remove special characters and split into terms
    terms = [term for term in query.split() if term.strip()]

    if not terms:
        return ""

    if search_type == "exact":
        # For exact phrase matching, wrap in quotes
        return f"'{query}'"

    # Format each term
    formatted_terms = [f"'{term}'" for term in terms]

    if search_type == "all":
        # Use & to require all terms
        return " & ".join(formatted_terms)
    else:  # "any"
        # Use | to match any term
        return " | ".join(formatted_terms)


if __name__ == "__main__":
    import uvicorn

    # Ensure the PDF directory exists
    os.makedirs(PDF_DIR, exist_ok=True)
    logger.info(f"PDF directory set to: {PDF_DIR}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
