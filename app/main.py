import os
import logging

from fastapi import FastAPI, HTTPException, Query
from pathlib import Path

from .models.model import ProcessingResult
from .processors.pdf_processor import extract_and_preprocess
from .processors.classifier import classify_document


# Define a base directory for PDF files
PDF_DIR = Path(
    os.environ.get("PDF_DIR", os.path.join(os.path.dirname(__file__), "..", "pdfs"))
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API", description="API for PDF processing with NLP"
)


@app.get("/")
def read_root():
    return {"status": "active", "service": "Document Processing API"}


@app.get("/process_pdf", response_model=ProcessingResult)
async def process_pdf(
    file_path: str = Query(..., description="Path to the local PDF file"),
):
    """
    Process a local PDF file.

    - Extracts text from PDF
    - Classifies the document
    - Extracts relevant tags and entities
    """
    try:
        # Initialize result
        result = ProcessingResult(file_path=file_path)

        # Get the full path to the local PDF file
        full_path = os.path.join(PDF_DIR, file_path)

        # Check if file exists
        if not os.path.isfile(full_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF file not found: {file_path}. Make sure the file exists in the configured directory.",
            )

        logger.info(f"Processing local PDF: {full_path}")

        text = ""

        text = extract_and_preprocess(full_path)
        result.text = text[:5000]  # Limit text size in the response
        logger.info(f"Extracted {len(text)} characters of text")
        if text:
            result.classification = classify_document(text, file_path)
        return result

    except HTTPException:
        raise  # Re-raise HTTPExceptions as they are
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Ensure the PDF directory exists
    os.makedirs(PDF_DIR, exist_ok=True)
    logger.info(f"PDF directory set to: {PDF_DIR}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
