import os
import logging
import tempfile
from supabase import create_client, Client

logger = logging.getLogger(__name__)


# Initialize Supabase client
def get_supabase_client() -> Client:
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )

        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")
        return supabase
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise


def download_file_from_storage(bucket_name: str, file_path: str) -> str:
    """
    Download a file from Supabase storage to a temporary file.
    Returns the path to the temporary file.
    """
    supabase = get_supabase_client()

    try:
        logger.info(f"Downloading file from {bucket_name}/{file_path}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name

            # Get file from Supabase storage
            response = supabase.storage.from_(bucket_name).download(file_path)
            temp_file.write(response)
            logger.info(f"File downloaded to {temp_path}")

            return temp_path
    except Exception as e:
        logger.error(f"Error downloading file from Supabase: {e}")
        raise
