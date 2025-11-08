from flask import Blueprint, request, jsonify
from firebase_config import get_storage_bucket
from io import BytesIO
import pandas as pd
import os
import json
import traceback
from werkzeug.utils import secure_filename
from post_upload_processor import process_uploaded_file
from datetime import datetime, timezone
from models import db
from models import (
    User, 
    UserProcess, 
    DataFrame, 
    DataFrameOperation, 
    OperationType,
)

file_operations = Blueprint('file_operations', __name__,url_prefix='/api/operations/')

def load_file(file_name: str, email: str, sheet_name: str = None) -> pd.DataFrame:
    """Load a file from Firebase Storage based on its extension and sheet name."""
    try:
        bucket = get_storage_bucket()
        
        # First try uploaded_files
        file_path = f'{email}/uploaded_files/{file_name}'
        blob = bucket.blob(file_path)
        
        # If not in uploaded_files, try processed_files
        if not blob.exists():
            file_path = f'{email}/processed_files/{file_name}'
            blob = bucket.blob(file_path)
            if not blob.exists():
                raise FileNotFoundError(f"File '{file_name}' not found in storage")

        file_content = blob.download_as_bytes()
        
        # Load based on file extension
        if file_name.lower().endswith('.csv'):
            return pd.read_csv(BytesIO(file_content))
        elif file_name.lower().endswith(('.xls', '.xlsx')):
            if sheet_name:
                return pd.read_excel(BytesIO(file_content), sheet_name=sheet_name)
            else:
                return pd.read_excel(BytesIO(file_content))
        else:
            raise ValueError("Unsupported file type. Only Excel and CSV files are supported.")
            
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def save_file(df: pd.DataFrame, file_name: str, email: str, sheet_name: str = None, replace_existing: bool = False):
    """Save a DataFrame to Firebase Storage."""
    try:
        bucket = get_storage_bucket()
        output_buffer = BytesIO()
        
        if file_name.lower().endswith('.csv'):
            df.to_csv(output_buffer, index=False)
        elif file_name.lower().endswith(('.xls', '.xlsx')):
            with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name=sheet_name or 'Sheet1', index=False)
        
        output_buffer.seek(0)
        
        # Determine file path based on replace_existing flag
        if replace_existing:
            file_path = f"{email}/processed_files/{file_name}"
            # Delete existing file if it exists
            existing_blob = bucket.blob(file_path)
            if existing_blob.exists():
                existing_blob.delete()
        else:
            base_name, ext = os.path.splitext(file_name)
            new_file_name = f"{base_name}_modified{ext}"
            file_path = f"{email}/processed_files/{new_file_name}"
            
        # Upload new file
        blob = bucket.blob(file_path)
        content_type = 'text/csv' if file_name.lower().endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        blob.upload_from_file(output_buffer, content_type=content_type)
        
        # Save preview
        preview_df = df.head(50)
        preview_buffer = BytesIO()
        preview_df.to_csv(preview_buffer, index=False)
        preview_buffer.seek(0)
        
        preview_path = f"{email}/previews/{file_name}/{sheet_name or 'preview'}_preview.csv"
        preview_blob = bucket.blob(preview_path)
        preview_blob.upload_from_file(preview_buffer, content_type='text/csv')
        
        return os.path.basename(file_path)
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")

def move_column(df: pd.DataFrame, column: str, positions: int) -> pd.DataFrame:
    """Move a column by the specified number of positions."""
    cols = df.columns.tolist()
    current_pos = cols.index(column)
    new_pos = max(0, min(len(cols) - 1, current_pos + positions))
    
    cols.pop(current_pos)
    cols.insert(new_pos, column)
    
    return df[cols]

@file_operations.route('/columns/', methods=['GET'])
def get_columns():
    try:
        # Assuming you have the DataFrame stored somewhere
        df = pd.read_csv('your_file.csv')  # Adjust based on your storage method
        return jsonify({
            'columns': df.columns.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_value(value):
    """Convert string value to appropriate data type."""
    if not isinstance(value, str):
        return value
        
    value = value.strip()
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() in ['null', 'nan', '']:
        return None
    
    # Try parsing as date first
    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        pass
    
    # Try numeric conversion
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def apply_replace_operation(df, columns, old_value, new_value):
    """Apply replace operation to specified columns."""
    old_value = convert_value(old_value)
    new_value = convert_value(new_value)
    
    if 'All' in columns:
        columns = df.columns.tolist()
    
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if isinstance(old_value, (pd.Timestamp, str)):
                try:
                    old_dt = pd.to_datetime(old_value)
                    new_dt = pd.to_datetime(new_value)
                    df[col] = df[col].replace(old_dt, new_dt)
                except (ValueError, TypeError):
                    pass
            elif old_value is None:
                df[col] = df[col].fillna(new_value)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if isinstance(old_value, (int, float)):
                df[col] = df[col].replace(old_value, new_value)
            elif old_value is None:
                df[col] = df[col].fillna(new_value)
        elif pd.api.types.is_bool_dtype(df[col]):
            if isinstance(old_value, bool):
                df[col] = df[col].replace(old_value, new_value)
            elif old_value is None:
                df[col] = df[col].fillna(new_value)
        else:
            if old_value is None:
                df[col] = df[col].fillna(new_value)
            else:
                df[col] = df[col].astype(str).replace(str(old_value), str(new_value))
    return df

def apply_rename_operation(df, columns_config):
    """Apply rename operation to columns."""
    rename_map = {
        config['column']: config['newName']
        for config in columns_config
        if config['column'] and config['newName']
    }
    return df.rename(columns=rename_map)

def get_column_type(series):
    """Determine the type of a pandas Series, prioritizing date detection and returning 'date'."""
    try:
        # If it's not a Series, try to convert it
        if not isinstance(series, pd.Series):
            try:
                series = pd.Series(series)
            except Exception:
                return 'string'  # Default to string if conversion fails

        non_null = series.dropna()
        if len(non_null) == 0:
            return 'string'
        
        # Check for boolean
        unique_values = set(non_null.astype(str).str.lower())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
        if unique_values.issubset(boolean_values):
            return 'boolean'
        
        # Check for date patterns BEFORE numeric (handles YYYYMMDD too)
        try:
            str_series = non_null.astype(str)
            import re
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}$',
                r'^\d{2}/\d{2}/\d{4}$',
                r'^\d{4}/\d{2}/\d{2}$',
                r'^\d{2}-\d{2}-\d{4}$',
                r'^\d{4}\d{2}\d{2}$',
                r'^\d{2}\.\d{2}\.\d{4}$',
                r'^\d{4}\.\d{2}\.\d{2}$',
            ]
            date_matches = sum(1 for val in str_series if any(re.match(p, val) for p in date_patterns))
            if date_matches > len(str_series) * 0.7:
                pd.to_datetime(non_null, errors='raise')
                return 'date'
        except (ValueError, TypeError):
            pass
        
        # Check for numeric
        try:
            numeric_series = pd.to_numeric(non_null)
            if all(numeric_series.apply(lambda x: float(x).is_integer())):
                return 'integer'
            else:
                return 'float'
        except (ValueError, TypeError):
            # Fallback date parsing
            try:
                pd.to_datetime(non_null, errors='raise')
                return 'date'
            except (ValueError, TypeError):
                return 'string'
        
        # Default to string for everything else
        return 'string'
    except Exception:
        return 'string'  # Default to string for any other errors

def save_processed_file(email, df, output_file_name, sheet_name):
    """Save processed file, metadata, and preview to Firebase."""
    bucket = get_storage_bucket()
    output_buffer = BytesIO()
    
    if output_file_name.lower().endswith('.csv'):
        df.to_csv(output_buffer, index=False)
        content_type = 'text/csv'
    else:
        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name or 'Sheet1', index=False)
        content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    output_buffer.seek(0)
    processed_blob = bucket.blob(f"{email}/processed_files/{output_file_name}")
    processed_blob.upload_from_file(output_buffer, content_type=content_type)

    # Generate and save metadata
    metadata = {
        "sheets": {
            sheet_name or "Sheet1": {
                "columns": df.columns.tolist(),
                "columnTypes": {col: get_column_type(df[col]) for col in df.columns}
            }
        }
    }

    metadata_blob = bucket.blob(f"{email}/metadata/{output_file_name}.json")
    metadata_blob.upload_from_string(json.dumps(metadata), content_type='application/json')

    # Save preview
    preview_buffer = BytesIO()
    df.head(50).to_csv(preview_buffer, index=False)
    preview_buffer.seek(0)
    preview_blob = bucket.blob(f"{email}/previews/{output_file_name}/{sheet_name or 'Sheet1'}_preview.csv")
    preview_blob.upload_from_file(preview_buffer, content_type='text/csv')

    return processed_blob.generate_signed_url(expiration=3600, version='v4')

def process_file_operations(email, file_name, operations, sheet_name=None, new_file_name=''):
    """Core logic for processing file operations."""
    try:
        if isinstance(operations, str):
            operations = [{
                'operation': operations,
                'data': {}
            }]
        elif not isinstance(operations, list):
            operations = []

        # Load the DataFrame
        df = load_file(file_name, email, sheet_name)
        
        # Apply each operation in sequence
        for operation_config in operations:
            operation = operation_config.get('operation')
            operation_data = operation_config.get('data', {})
            
            if operation == 'replace':
                df = apply_replace_operation(
                    df,
                    operation_data.get('selectedColumns', []),
                    operation_data.get('oldValue'),
                    operation_data.get('newValue')
                )
            elif operation == 'rename':
                df = apply_rename_operation(df, operation_data.get('columns', []))
            elif operation == 'reorder':
                for config in operation_data.get('columns', []):
                    if config['column']:
                        df = move_column(df, config['column'], config['positions'])

        # Determine output filename
        if not new_file_name:
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"modified_{base_name}.xlsx"
        else:
            if not os.path.splitext(new_file_name)[1]:
                output_file_name = f"{new_file_name}.xlsx"
            else:
                output_file_name = new_file_name

        # Save processed file and get download URL
        download_url = save_processed_file(email, df, output_file_name, sheet_name)

        return {
            "success": True,
            "message": "Operations applied successfully.",
            "downloadUrl": download_url,
            "fileName": output_file_name
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@file_operations.route('/operations/', methods=['POST'])
def handle_operations():
    """Endpoint for handling file operations."""
    try:
        data = request.json
        email = request.headers.get('X-User-Email')
        
        if not email:
            return jsonify({"error": "User email is required"}), 400
            
        file_name = data.get('file_name')
        if not file_name:
            return jsonify({"error": "File name is required"}), 400

        result = process_file_operations(
            email=email,
            file_name=file_name,
            operations=data.get('operation'),
            sheet_name=data.get('sheet_name'),
            new_file_name=data.get('newFileName', '').strip()
        )

        if not result.get('success'):
            return jsonify({"error": result.get('error')}), 400

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@file_operations.route('/preview/', methods=['POST'])
def preview_operations():
    try:
        data = request.json
        operations = data.get('operation')  # Changed to match payload
        if isinstance(operations, str):
            # If a single operation is sent as string, convert to list with operation object
            operations = [{
                'operation': operations,
                'data': data.get('data', {})
            }]
        elif not isinstance(operations, list):
            operations = []

        file_name = data.get('file_name')  # Changed to match payload
        sheet_name = data.get('sheet_name')  # Changed to match payload
        email = request.headers.get('X-User-Email')
        
        if not email:
            return jsonify({"error": "User email is required"}), 400
            
        if not file_name:
            return jsonify({"error": "File name is required"}), 400
        
        # Load the DataFrame
        df = load_file(file_name, email, sheet_name)
        preview_df = df.copy()
        
        # Apply each operation in sequence
        for operation_config in operations:
            operation = operation_config.get('operation')
            operation_data = operation_config.get('data', {})
            
            if operation == 'replace':
                columns = operation_data.get('selectedColumns', [])
                old_value = operation_data.get('oldValue')
                new_value = operation_data.get('newValue')
                
                if 'All' in columns:
                    columns = preview_df.columns.tolist()
                
                # Convert values based on data type
                def convert_value(value):
                    if not isinstance(value, str):
                        return value
                        
                    value = value.strip()
                    if value.lower() == 'true':
                        return True
                    elif value.lower() == 'false':
                        return False
                    elif value.lower() in ['null', 'nan', '']:
                        return None
                    
                    # Try parsing as date first
                    try:
                        return pd.to_datetime(value)
                    except (ValueError, TypeError):
                        pass
                    
                    # Try numeric conversion
                    try:
                        if '.' in value:
                            return float(value)
                        return int(value)
                    except ValueError:
                        return value

                old_value = convert_value(old_value)
                new_value = convert_value(new_value)
                
                for col in columns:
                    # Handle different data types appropriately
                    if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                        if isinstance(old_value, (pd.Timestamp, str)):
                            try:
                                old_dt = pd.to_datetime(old_value)
                                new_dt = pd.to_datetime(new_value)
                                preview_df[col] = preview_df[col].replace(old_dt, new_dt)
                            except (ValueError, TypeError):
                                pass
                        elif old_value is None:
                            preview_df[col] = preview_df[col].fillna(new_value)
                    elif pd.api.types.is_numeric_dtype(preview_df[col]):
                        if isinstance(old_value, (int, float)):
                            preview_df[col] = preview_df[col].replace(old_value, new_value)
                        elif old_value is None:
                            preview_df[col] = preview_df[col].fillna(new_value)
                    elif pd.api.types.is_bool_dtype(preview_df[col]):
                        if isinstance(old_value, bool):
                            preview_df[col] = preview_df[col].replace(old_value, new_value)
                        elif old_value is None:
                            preview_df[col] = preview_df[col].fillna(new_value)
                    else:
                        # For string/object columns, convert to string for comparison
                        if old_value is None:
                            preview_df[col] = preview_df[col].fillna(new_value)
                        else:
                            preview_df[col] = preview_df[col].astype(str).replace(str(old_value), str(new_value))

            elif operation == 'rename':
                rename_map = {
                    config['column']: config['newName']
                    for config in operation_data.get('columns', [])
                    if config['column'] and config['newName']
                }
                preview_df = preview_df.rename(columns=rename_map)
                
            elif operation == 'reorder':
                for config in operation_data.get('columns', []):
                    if config['column']:
                        preview_df = move_column(preview_df, config['column'], config['positions'])
        
        preview_data = {
            'columns': preview_df.columns.tolist(),
            'rows': preview_df.head(100).to_dict('records')
        }
        
        return jsonify({
            'success': True,
            'data': preview_data
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@file_operations.route('/process/apply/', methods=['POST'])
def process_file_operations():
    """Endpoint for handling DataFrame operations within a process."""
    try:
        data = request.json
        email = request.headers.get('X-User-Email')
        
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        process_id = data.get('processId')
        table_name = data.get('tableName')
        output_table_name = data.get('outputTableName', '').strip()
        operations = data.get('operations', [])

        if not all([process_id, table_name, output_table_name]):
            return jsonify({
                "error": "Missing required parameters: processId, tableName, or outputTableName."
            }), 400

        # Get process and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get source DataFrame record
        source_df = DataFrame.query.filter_by(
            process_id=process_id,
            name=table_name
        ).first()
        if not source_df:
            return jsonify({"error": f"Table '{table_name}' not found in process"}), 404

        # Check if output table name already exists
        existing_df = DataFrame.query.filter_by(
            process_id=process_id,
            name=output_table_name
        ).first()
        if existing_df:
            if existing_df.is_temporary == False:
                return jsonify({"error": f"Table with name {output_table_name} already exists."}), 409

        # Generate message based on operations
        message_parts = []
        column_mapping = {}  # Keep track of renamed columns for reorder operations
        
        for operation in operations:
            op_type = operation.get('type')
            
            if op_type == 'rename_columns':
                mapping = operation.get('mapping', {})
                renamed_columns = []
                for old_name, new_name in mapping.items():
                    renamed_columns.append(old_name)
                    column_mapping[old_name] = new_name
                if renamed_columns:
                    message_parts.append(f"Rename columns {', '.join(renamed_columns)}")
            
            elif op_type == 'reorder_columns':
                order = operation.get('order', [])
                if order:
                    message_parts.append(f"Reorder columns {', '.join(order)}")
            
            elif op_type == 'replace_values':
                replacements = operation.get('replacements', [])
                replace_columns = set()
                for rep in replacements:
                    col_name = column_mapping.get(rep['column'], rep['column'])
                    replace_columns.add(col_name)
                if replace_columns:
                    message_parts.append(f"Replace values in columns {', '.join(replace_columns)}")

        message = " and ".join(message_parts)

        # Create DataFrameOperation record with IN_PROGRESS status
        df_operation = DataFrameOperation(
            process_id=process_id,
            dataframe_id=source_df.id,
            operation_type=OperationType.REPLACE_RENAME_REORDER.value,
            payload=data,
            message=message
        )
        df_operation.user_id = user.id
        
        # Save initial operation record
        db.session.add(df_operation)
        db.session.commit()

        try:
            result = process_dataframe_operations(
                email=email,
                process_id=process_id,
                source_df=source_df,
                operations=operations,
                output_table_name=output_table_name,
                existing_df=existing_df
            )

            if not result.get('success'):
                df_operation.set_error(result.get('error'))
                db.session.commit()
                return jsonify({"error": result.get('error')}), 400

            # Update operation status to success
            df_operation.set_success()
            db.session.commit()

            # Add operation details to the response
            result['operationId'] = df_operation.id
            result['operationStatus'] = df_operation.status
            result['message'] = message  # Include the message in the response

            return jsonify(result)

        except Exception as e:
            df_operation.set_error(str(e))
            db.session.commit()
            traceback.print_exc()
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def process_dataframe_operations(email, process_id, source_df, operations, output_table_name, existing_df=None):
    """Process operations on DataFrame data."""
    # Initialize variables for cleanup
    df_blob = None
    metadata_blob = None
    
    try:
        bucket = get_storage_bucket()

        # Read source DataFrame
        content = bucket.blob(source_df.storage_path).download_as_string()
        df = pd.read_csv(BytesIO(content))

        # Separate rename operations from other operations
        rename_operations = []
        other_operations = []
        
        for operation in operations:
            if operation.get('type') == 'rename_columns':
                rename_operations.append(operation)
            else:
                other_operations.append(operation)

        # Process non-rename operations first
        for operation in other_operations:
            op_type = operation.get('type')
            
            if op_type == 'delete_columns':
                columns = operation.get('columns', [])
                df = df.drop(columns=columns, errors='ignore')
                
            elif op_type == 'reorder_columns':
                new_order = operation.get('order', [])
                all_columns = df.columns.tolist()
                
                # Validate columns
                invalid_columns = [col for col in new_order if col not in all_columns]
                if invalid_columns:
                    raise ValueError(f"Invalid columns in reorder operation: {invalid_columns}")
                
                # Keep unspecified columns in their original order
                remaining_columns = [col for col in all_columns if col not in new_order]
                final_order = new_order + remaining_columns
                df = df[final_order]

            elif op_type == 'replace_values':
                replacements = operation.get('replacements', [])
                for replacement in replacements:
                    column = replacement.get('column')
                    if column not in df.columns:
                        raise ValueError(f"Column '{column}' not found for replacement operation")
                    
                    old_value = replacement.get('oldValue')
                    new_value = replacement.get('newValue')
                    match_case = replacement.get('matchCase', True)
                    
                    if match_case or pd.api.types.is_numeric_dtype(df[column]):
                        # Direct replacement for numeric or case-sensitive
                        try:
                            df[column] = df[column].replace(float(old_value), float(new_value))
                        except ValueError:
                            df[column] = df[column].replace(old_value, new_value)
                    else:
                        # Case-insensitive replacement for string values
                        mask = df[column].astype(str).str.lower() == str(old_value).lower()
                        df.loc[mask, column] = new_value

        # Process rename operations last
        for operation in rename_operations:
            rename_map = operation.get('mapping', {})
            df = df.rename(columns=rename_map)

        # Generate storage paths
        storage_path = f"{email}/process/{process_id}/dataframes/{output_table_name}.csv"

        try:
            # Save DataFrame as CSV
            df_buffer = BytesIO()
            df.to_csv(df_buffer, index=False)
            df_buffer.seek(0)

            # Upload to Firebase
            df_blob = bucket.blob(storage_path)
            df_blob.upload_from_file(df_buffer, content_type='text/csv')

            # Create or update DataFrame record
            if existing_df:
                existing_df.row_count = len(df)
                existing_df.column_count = len(df.columns)
                existing_df.updated_at = datetime.now(timezone.utc)
                existing_df.storage_path = storage_path
                dataframe_record = existing_df
            else:
                dataframe_record = DataFrame.create_from_pandas(
                    df=df,
                    process_id=process_id,
                    name=output_table_name,
                    email=email,
                    storage_path=storage_path,
                    user_id=source_df.user_id,
                    is_temporary=True,
                )
                db.session.add(dataframe_record)

            # Save metadata
            metadata = {
                "type": "processed_table",
                "description": f"Processed table from {source_df.name}",
                "processId": process_id,
                "createdAt": (existing_df.created_at if existing_df else datetime.now(timezone.utc)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                "updatedAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                "rowCount": len(df),
                "columnCount": len(df.columns),
                "columns": df.columns.tolist(),
                "columnTypes": {col: get_column_type(df[col]) for col in df.columns},
                "operation": {
                    "type": "file_operations",
                    "operations": operations,
                    "sourceTable": {
                        "id": source_df.id,
                        "name": source_df.name
                    }
                }
            }

            metadata_path = f"{email}/process/{process_id}/metadata/{output_table_name}.json"
            metadata_blob = bucket.blob(metadata_path)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )

            # Commit the database changes
            db.session.commit()

            return {
                "success": True,
                "message": f"Table processed successfully as '{output_table_name}'",
                "id": dataframe_record.id,
                "name": output_table_name,
                "rowCount": len(df),
                "columnCount": len(df.columns),
                "metadata": metadata,
                "isUpdate": existing_df is not None
            }

        except Exception as e:
            db.session.rollback()
            # Clean up uploaded files if they exist
            if df_blob and df_blob.exists():
                df_blob.delete()
            if metadata_blob and metadata_blob.exists():
                metadata_blob.delete()
            raise e

    except Exception as e:
        # Clean up any uploaded files in case of outer exception
        if df_blob and df_blob.exists():
            df_blob.delete()
        if metadata_blob and metadata_blob.exists():
            metadata_blob.delete()
        raise Exception(f"Error processing table: {str(e)}") 