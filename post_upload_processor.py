from firebase_config import get_storage_bucket
from io import BytesIO
import pandas as pd
import json
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get Firebase Storage bucket
bucket = get_storage_bucket()

def process_uploaded_file(file_content: bytes, file_name: str, email: str) -> dict:
    """
    Main function to process an uploaded file. Handles both metadata extraction and preview generation.
    
    Args:
        file_content (bytes): Raw file content
        file_name (str): Name of the uploaded file
        email (str): User's email for storage organization
        
    Returns:
        dict: Processing results including status and any errors
    """
    try:
        # Determine file type
        is_excel = file_name.endswith(('.xls', '.xlsx'))
        file_type = 'Excel' if is_excel else 'CSV'
        
        # Process metadata
        metadata = extract_and_save_metadata(file_content, file_name, email, file_type)
        
        # Generate previews
        preview_status = generate_and_save_previews(file_content, file_name, email, file_type)
        
        return {
            "success": True,
            "metadata_status": metadata.get("success", False),
            "preview_status": preview_status.get("success", False),
            "errors": metadata.get("errors", []) + preview_status.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}")
        return {
            "success": False,
            "errors": [str(e)]
        }

def extract_and_save_metadata(file_content: bytes, file_name: str, email: str, file_type: str) -> dict:
    """
    Extracts metadata from file and saves it to Firebase Storage.
    """
    try:
        # Extract metadata based on file type
        metadata = extract_metadata(file_content, file_type)
        
        # Save metadata to Firebase
        metadata_blob = bucket.blob(f"{email}/metadata/{file_name}.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata),
            content_type='application/json'
        )
        
        logger.info(f"Metadata saved successfully for {file_name}")
        return {"success": True, "metadata": metadata}
        
    except Exception as e:
        logger.error(f"Metadata extraction failed for {file_name}: {str(e)}")
        return {"success": False, "errors": [str(e)]}

def generate_and_save_previews(file_content: bytes, file_name: str, email: str, file_type: str) -> dict:
    """
    Generates and saves preview files to Firebase Storage.
    """
    errors = []
    try:
        file_buffer = BytesIO(file_content)
        
        if file_type == 'Excel':
            excel_file = pd.ExcelFile(file_buffer)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=50)
                    if not df.empty:
                        preview_buffer = BytesIO()
                        df.to_csv(preview_buffer, index=False)
                        preview_buffer.seek(0)
                        
                        preview_path = f"{email}/previews/{file_name}/{sheet_name}_preview.csv"
                        preview_blob = bucket.blob(preview_path)
                        preview_blob.upload_from_file(
                            preview_buffer,
                            content_type='text/csv'
                        )
                        
                except Exception as e:
                    errors.append(f"Error processing sheet {sheet_name}: {str(e)}")
                    
        else:  # CSV
            try:
                df = pd.read_csv(file_buffer, nrows=50)
                if not df.empty:
                    preview_buffer = BytesIO()
                    df.to_csv(preview_buffer, index=False)
                    preview_buffer.seek(0)
                    
                    preview_path = f"{email}/previews/{file_name}/preview.csv"
                    preview_blob = bucket.blob(preview_path)
                    preview_blob.upload_from_file(
                        preview_buffer,
                        content_type='text/csv'
                    )
                    
            except Exception as e:
                errors.append(f"Error processing CSV: {str(e)}")
                
        return {
            "success": len(errors) == 0,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Preview generation failed for {file_name}: {str(e)}")
        return {
            "success": False,
            "errors": [str(e)]
        }

def get_file_metadata(file_name: str, email: str) -> dict:
    """
    Retrieves previously saved metadata for a file.
    
    Args:
        file_name (str): Name of the file
        email (str): User's email
        
    Returns:
        dict: File metadata or None if not found
    """
    try:
        blob = bucket.blob(f"{email}/metadata/{file_name}.json")
        if blob.exists():
            return json.loads(blob.download_as_string())
        return None
    except Exception as e:
        logger.error(f"Error retrieving metadata for {file_name}: {str(e)}")
        return None 