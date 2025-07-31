from flask import Blueprint, request, jsonify
from firebase_config import get_storage_bucket
from io import BytesIO
import pandas as pd
import os
import json 
import traceback
from models import User, db
from models import UserProcess
from datetime import datetime, timezone
from models import DataFrame
from models import DataFrameOperation
from models import DataFrameBatchOperation

# Blueprint
edit_file_bp = Blueprint('edit_file', __name__, url_prefix='/api/edit-file/')

def handle_duplicate_columns(df):
    """Handle duplicate column names in DataFrame"""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}.{i+1}" if i > 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

def apply_data_type_conversion(df, column_types, datetime_formats):
    """Apply data type conversions for the selected columns"""
    for column, target_type in column_types.items():
        if column in df.columns:
            try:
                if target_type == "integer":
                    df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
                elif target_type == "float":
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                elif target_type in ["str", "string"]:  # Accept both "str" and "string"
                    df[column] = df[column].astype(str)
                elif target_type == "datetime":
                    df[column] = parse_and_format_dates(df[column], datetime_formats.get(column))
                    
            except Exception as e:
                raise ValueError(f"Error converting column '{column}' to '{target_type}': {str(e)}")

def infer_date_format(series):
    """
    Infer the most likely date format from a series of dates.
    Returns both python and display formats.
    """
    common_formats = [
        ('%Y-%m-%d', 'YYYY-MM-DD'),
        ('%d/%m/%Y', 'DD/MM/YYYY'),
        ('%m/%d/%Y', 'MM/DD/YYYY'),
        ('%Y/%m/%d', 'YYYY/MM/DD'),
        ('%d-%m-%Y', 'DD-MM-YYYY'),
        ('%m-%d-%Y', 'MM-DD-YYYY'),
        ('%Y%m%d', 'YYYYMMDD'),
        ('%d.%m.%Y', 'DD.MM.YYYY'),
        ('%Y.%m.%d', 'YYYY.MM.DD')
    ]
    
    # Get non-null sample values
    sample_dates = series.dropna().head(10).astype(str)
    if sample_dates.empty:
        return '%Y-%m-%d', 'YYYY-MM-DD'  # Default format if no data
    
    for py_format, _ in common_formats:
        try:
            # Try to parse all sample dates with this format
            all_parsed = all(pd.to_datetime(date, format=py_format, errors='raise') 
                           for date in sample_dates)
            if all_parsed:
                return py_format, next(display for py, display in common_formats if py == py_format)
        except:
            continue
    
    return '%Y-%m-%d', 'YYYY-MM-DD'  # Default format if no match found

def parse_and_format_dates(series, format=None):
    """
    Parse and format date values with smart format detection and standardization.
    
    Args:
        series: pandas Series containing date values
        format: optional Python date format string
    
    Returns:
        tuple: (formatted_series, inferred_format, display_format)
    """
    def parse_partial_date(val):
        if pd.isna(val):
            return pd.NaT
        
        val = str(val).strip()
        
        # Handle year-only values
        if val.isdigit() and len(val) == 4:
            return f"{val}-01-01"
        
        # Handle year-month values
        if len(val.split('-')) == 2:
            year, month = val.split('-')
            if year.isdigit() and month.isdigit():
                return f"{val}-01"
        
        return val

    # Clean and standardize date values
    series = series.apply(parse_partial_date)
    
    # Infer format if not provided
    if not format:
        inferred_format, display_format = infer_date_format(series)
        format = inferred_format
    else:
        # Map common display formats to Python formats
        format_mapping = {
            'YYYY-MM-DD': '%Y-%m-%d',
            'DD/MM/YYYY': '%d/%m/%Y',
            'MM/DD/YYYY': '%m/%d/%Y',
            'YYYY/MM/DD': '%Y/%m/%d',
            'DD-MM-YYYY': '%d-%m-%Y',
            'MM-DD-YYYY': '%m-%d-%Y',
            'YYYYMMDD': '%Y%m%d',
            'DD.MM.YYYY': '%d.%m.%Y',
            'YYYY.MM.DD': '%Y.%m.%d'
        }
        format = format_mapping.get(format, format)
        display_format = next((display for py, display in format_mapping.items() if py == format), format)
    
    try:
        # Try with specified/inferred format
        parsed_series = pd.to_datetime(series, format=format, errors='raise')
    except ValueError:
        try:
            # Fallback to automatic parsing
            parsed_series = pd.to_datetime(series, infer_datetime_format=True, errors='raise')
        except ValueError as e:
            # Get examples of problematic values
            sample_bad_values = series[~series.isin(pd.to_datetime(series, errors='coerce').dropna())].head(5)
            error_msg = f"Could not parse dates. Sample invalid values: {', '.join(map(str, sample_bad_values))}"
            raise ValueError(error_msg)
    
    # Standardize output format
    formatted_series = parsed_series.dt.strftime('%Y-%m-%d')
    
    return formatted_series, format, display_format

def process_excel_file(df, sheet_data):
    """Process Excel file sheet data"""
    df = handle_duplicate_columns(df)

    columns = sheet_data.get('columns', [])
    column_types = sheet_data.get('columnTypes', {})
    datetime_formats = sheet_data.get('datetimeFormats', {})

    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        return None, None, None

    filtered_df = df.reindex(columns=valid_columns)
    apply_data_type_conversion(filtered_df, column_types, datetime_formats)

    # Generate preview
    preview_buffer = BytesIO()
    filtered_df.head(50).to_csv(preview_buffer, index=False)
    preview_buffer.seek(0)

    # Generate metadata
    metadata = {
        "columns": valid_columns,
        "columnTypes": column_types,
    }

    return filtered_df, preview_buffer, metadata

def edit_file(email, file_name, new_file_name, replace_existing, selected_sheets):
    """Core logic for editing files"""
    try:
        bucket = get_storage_bucket()
        
        # Try both locations for the input file
        file_paths = [
            f'{email}/uploaded_files/{file_name}',
            f'{email}/processed_files/{file_name}'
        ]
        
        blob = None
        for path in file_paths:
            temp_blob = bucket.blob(path)
            if temp_blob.exists():
                blob = temp_blob
                break
                
        if not blob:
            return {
                "success": False,
                "error": "The specified file does not exist"
            }

        file_content = blob.download_as_bytes()
        file_extension = os.path.splitext(file_name)[-1].lower()

        new_metadata = {}
        preview_buffers = {}
        output_buffer = BytesIO()

        if file_extension in ['.xls', '.xlsx']:
            with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                xls = pd.ExcelFile(BytesIO(file_content))
                for sheet_name, sheet_data in selected_sheets.items():
                    if sheet_name in xls.sheet_names:
                        df = xls.parse(sheet_name)
                        filtered_df, preview_buffer, metadata = process_excel_file(df, sheet_data)
                        
                        if filtered_df is not None:
                            filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            preview_buffers[sheet_name] = preview_buffer
                            new_metadata[sheet_name] = metadata

        elif file_extension == '.csv':
            df = pd.read_csv(BytesIO(file_content))
            filtered_df, preview_buffer, metadata = process_excel_file(df, selected_sheets)
            
            if filtered_df is not None:
                filtered_df.to_csv(output_buffer, index=False)
                preview_buffers['CSV'] = preview_buffer
                new_metadata['CSV'] = metadata
            else:
                return {
                    "success": False,
                    "error": "No valid columns found in CSV file"
                }

        # Determine output filename
        if not new_file_name:
            base_name, ext = os.path.splitext(file_name)
            output_file_name = f"{base_name}_Edited{file_extension}"
        elif not os.path.splitext(new_file_name)[1]:
            output_file_name = f"{new_file_name}{file_extension}"
        else:
            output_file_name = new_file_name

        # Save the processed file
        output_buffer.seek(0)
        if replace_existing:
            new_file_path = f"{email}/processed_files/{file_name}"
            blob.delete()
        else:
            new_file_path = f"{email}/processed_files/{output_file_name}"

        new_blob = bucket.blob(new_file_path)
        new_blob.upload_from_file(output_buffer, content_type=blob.content_type)

        # Save metadata
        metadata_blob_path = f"{email}/metadata/{output_file_name}.json"
        metadata_blob = bucket.blob(metadata_blob_path)
        metadata_blob.upload_from_string(
            json.dumps({"sheets": new_metadata}),
            content_type='application/json'
        )

        # Save previews
        for sheet_name, preview_buffer in preview_buffers.items():
            preview_blob_path = f"{email}/previews/{output_file_name}/{sheet_name}_preview.csv"
            preview_blob = bucket.blob(preview_blob_path)
            preview_blob.upload_from_file(preview_buffer, content_type='text/csv')

        return {
            "success": True,
            "message": "File processed successfully",
            "newFileName": os.path.basename(new_file_path),
            "downloadUrl": new_blob.generate_signed_url(expiration=3600, version='v4')
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@edit_file_bp.route('/process/', methods=['POST'])
def process_file():
    """Endpoint for processing files"""
    try:
        data = request.json
        email = request.headers.get('X-User-Email')
        
        if not email:
            return jsonify({"error": "Email is required in headers"}), 400

        file_name = data.get('fileName')
        new_file_name = data.get('newFileName', '').strip()
        replace_existing = data.get('replaceExisting', False)
        selected_sheets = data.get('selectedSheets', {})

        if not file_name or not selected_sheets:
            return jsonify({"error": "File name and selected sheets/columns are required"}), 400

        result = edit_file(
            email=email,
            file_name=file_name,
            new_file_name=new_file_name,
            replace_existing=replace_existing,
            selected_sheets=selected_sheets
        )

        if not result.get('success'):
            return jsonify({"error": result.get('error')}), 400

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def process_columns_and_types(email, process_id, table_name, column_selections=None, column_types=None, datetime_formats=None):
    """
    Process selected columns with specified data types and datetime formats.
    Updates existing DataFrame if one exists with the same name in the process.
    """
    try: 
        # Get user and process first
        user = User.query.filter_by(email=email).first()
        if not user:
            raise ValueError("User not found")
            
        process = UserProcess.query.filter_by(id=process_id).first()
        if not process:
            raise ValueError("Process not found")

        # Check if DataFrame already exists in this process
        existing_df = DataFrame.query.filter_by(
            process_id=process_id,
            name=table_name
        ).first()

        bucket = get_storage_bucket()
        
        # Define paths with .csv extension
        source_path = f"{email}/process/{process_id}/dataframes/{table_name}.csv"  # Changed from .parquet
        new_path = f"{email}/process/{process_id}/dataframes/{table_name}.csv"  # Changed from .parquet
        
        # Read source CSV file
        source_blob = bucket.blob(source_path)
        if not source_blob.exists():
            # Try to find the file in uploaded_files directory as fallback
            alt_source_path = f"{email}/uploaded_files/{table_name}"
            source_blob = bucket.blob(alt_source_path)
            if not source_blob.exists():
                raise FileNotFoundError(f"Source table {table_name} not found")
            
        df_buffer = BytesIO(source_blob.download_as_bytes())
        df = pd.read_csv(df_buffer)  # Changed from read_parquet
        
        # If no column selections provided, use all columns
        if column_selections is None:
            selected_columns = list(df.columns)
        else:
            selected_columns = [col for col, selected in column_selections.items() if selected]
            
        if not selected_columns:
            raise ValueError("No columns selected for processing")
            
        
        
        # Infer data types if not provided
        inferred_types = {}
        for column in df.columns:
            # If type is explicitly provided, use it
            if column_types and column in column_types:
                inferred_types[column] = column_types[column]
                continue
                
            try:
                # Try numeric types first
                numeric_series = pd.to_numeric(df[column], errors='raise')
                if all(numeric_series.apply(lambda x: x.is_integer())):
                    inferred_types[column] = 'integer'
                else:
                    inferred_types[column] = 'float'
                continue
            except:
                pass
            
            # Only try datetime if format is specified
            if datetime_formats and column in datetime_formats:
                try:
                    pd.to_datetime(df[column], errors='raise')
                    inferred_types[column] = 'datetime'
                    continue
                except:
                    pass
            
            # Default to string type
            inferred_types[column] = 'str'  # Changed from 'string' to 'str' for consistency
        
        # Use provided types or fall back to inferred types
        final_types = column_types or inferred_types
        
        # Process each column
        inferred_formats = {}
        for column in df.columns:
            target_type = final_types.get(column, inferred_types[column])
            
            try:
                if target_type == 'datetime':
                    format_str = datetime_formats.get(column) if datetime_formats else None
                    df[column], py_format, display_format = parse_and_format_dates(df[column], format_str)
                    inferred_formats[column] = {
                        'python_format': py_format,
                        'display_format': display_format
                    }
                elif target_type == 'integer':
                    df[column] = pd.to_numeric(df[column], errors='raise').astype('Int64')
                elif target_type == 'float':
                    df[column] = pd.to_numeric(df[column], errors='raise')
                elif target_type == 'str':
                    df[column] = df[column].astype(str)
                    
            except Exception as e:
                raise ValueError(f"Error converting column '{column}' to {target_type}: {str(e)}")
        
        # Filter DataFrame to selected columns
        df = df[selected_columns]
        # Save processed DataFrame as CSV
        output_buffer = BytesIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        
        # Define storage path
        storage_path = f"{email}/process/{process_id}/dataframes/{table_name}.csv"
        
        # Upload to Firebase
        new_blob = bucket.blob(storage_path)
        new_blob.upload_from_file(output_buffer, content_type='text/csv')
        
        # Generate metadata
        metadata = {
            "tableName": table_name,
            "description": f"Processed from {table_name} with type conversions",
            "sourceTableName": table_name,
            "processId": process_id,
            "createdAt": (existing_df.created_at if existing_df else datetime.now(timezone.utc)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "updatedAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "rowCount": len(df),
            "columnCount": len(df.columns)
        }

        # If existing metadata exists, preserve it and update only necessary fields
        if existing_df and existing_df.data_metadata:
            existing_metadata = existing_df.data_metadata
            # Update only specific fields while preserving others
            existing_metadata.update({
                "tableName": metadata["tableName"],
                "description": metadata["description"],
                "sourceTableName": metadata["sourceTableName"],
                "processId": metadata["processId"],
                "updatedAt": metadata["updatedAt"],
                "rowCount": metadata["rowCount"],
                "columnCount": metadata["columnCount"]
            })
            metadata = existing_metadata

        # Add or update column information
        if "columns" not in metadata:
            metadata["columns"] = []
        
        # Update column information
        for col in df.columns:
            # Check if column info already exists
            existing_col = next((c for c in metadata["columns"] if c["name"] == col), None)
            if existing_col:
                existing_col["type"] = final_types[col]
            else:
                metadata["columns"].append({
                    "name": col,
                    "type": final_types[col]
                })

        try:
            if existing_df:
                # Update existing record
                existing_df.row_count = len(df)
                existing_df.column_count = len(df.columns)
                existing_df.updated_at = datetime.now(timezone.utc)
                existing_df.storage_path = storage_path
                existing_df.data_metadata = metadata
                dataframe_record = existing_df
            else:
                # Create new DataFrame record
                dataframe_record = DataFrame.create_from_pandas(
                    df=df,
                    process_id=process_id,
                    name=table_name,
                    email=email,
                    storage_path=storage_path,
                    user_id=user.id
                )
                db.session.add(dataframe_record)

            db.session.commit()

        except Exception as db_error:
            db.session.rollback()
            # Clean up the uploaded file if database operation fails
            new_blob.delete()
            raise Exception(f"Failed to save DataFrame record: {str(db_error)}")
        
        # Save metadata after successful database commit
        metadata_path = f"{email}/process/{process_id}/metadata/{table_name}.json"
        metadata_blob = bucket.blob(metadata_path)
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type='application/json'
        )
        
        return {
            "success": True,
            "message": "Data processed and saved successfully",
            "id": dataframe_record.id,
            "metadata": metadata,
            "rowCount": len(df),
            "columnCount": len(df.columns),
            "isUpdate": existing_df is not None
        }
        
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@edit_file_bp.route('/edit/', methods=['POST'])
def process_columns():
    """
    Endpoint for processing multiple tables with type conversions
    """
    try:
        data = request.get_json()
        email = request.headers.get('X-User-Email')
        
        if not email:
            return jsonify({"error": "Email is required in headers"}), 400
            
        process_id = data.get('processId')
        tables = data.get('tables', [])
        
        if not process_id:
            return jsonify({"error": "Process ID is required"}), 400
        
        if not tables:
            return jsonify({"error": "At least one table configuration is required"}), 400
            
        # Verify process exists and user has access
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Create batch operation record
        batch_operation = DataFrameBatchOperation(
            process_id=process_id,
            payload=data,
            dataframe_ids=[],
            operation_ids=[],  # This will remain empty as we're not creating individual operations
            total_dataframes=len(tables)
        )
        
        db.session.add(batch_operation)
        db.session.commit()

        results = []
        error_messages = []

        # Process each table
        for table_config in tables:
            table_name = table_config.get('tableName')
            if not table_name:
                error_messages.append("Table name is required for each table configuration")
                continue

            # Get the DataFrame
            dataframe = DataFrame.query.filter_by(
                process_id=process_id,
                name=table_name,
                is_active=True
            ).first()

            if not dataframe:
                error_messages.append(f"DataFrame '{table_name}' not found")
                continue

            batch_operation.dataframe_ids.append(dataframe.id)

            try:
                result = process_columns_and_types(
                    email=email,
                    process_id=process_id,
                    table_name=table_name,
                    column_selections=table_config.get('columnSelections'),
                    column_types=table_config.get('columnTypes'),
                    datetime_formats=table_config.get('datetimeFormats')
                )
                
                if result.get('success'):
                    batch_operation.increment_success_count()
                else:
                    error_msg = f"Error processing {table_name}: {result.get('error')}"
                    error_messages.append(error_msg)

                result.update({
                    "tableName": table_name
                })
                results.append(result)

            except Exception as e:
                error_msg = f"Error processing {table_name}: {str(e)}"
                error_messages.append(error_msg)
                results.append({
                    "success": False,
                    "tableName": table_name,
                    "error": str(e)
                })

        db.session.commit()
        
        response = {
            "success": batch_operation.successful_dataframes > 0,
            "message": (
                "All tables processed successfully" if batch_operation.successful_dataframes == len(tables)
                else "Some tables failed to process" if batch_operation.successful_dataframes > 0
                else "All tables failed to process"
            ),
            "operationId": batch_operation.id,
            "results": results,
            "successCount": batch_operation.successful_dataframes,
            "totalCount": batch_operation.total_dataframes,
            "errors": error_messages if error_messages else None
        }
            
        return jsonify(response), 200 if batch_operation.successful_dataframes > 0 else 400
        
    except Exception as e:
        if 'batch_operation' in locals():
            batch_operation.set_error(str(e))
            try:
                db.session.commit()
            except:
                db.session.rollback()
            return jsonify({
                "success": False,
                "error": str(e),
                "batchOperationId": batch_operation.id
            }), 500
        
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def get_process_table_data(email, process_name, table_name, page=1, per_page=100):
    try:
        if not all([email, process_name, table_name]):
            raise ValueError("Email, process name, and table name are required")

        # Define paths (changed to CSV)
        df_path = f"{email}/process/{process_name}/dataframes/{table_name}.csv"
        metadata_path = f"{email}/process/{process_name}/metadata/{table_name}.json"

        # Get metadata
        metadata_blob = bucket.blob(metadata_path)
        if not metadata_blob.exists():
            raise FileNotFoundError(f"Metadata for table {table_name} not found")
        
        metadata = json.loads(metadata_blob.download_as_string())

        # Get DataFrame
        df_blob = bucket.blob(df_path)
        if not df_blob.exists():
            raise FileNotFoundError(f"Table {table_name} not found")

        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        # Read CSV file instead of parquet
        df_buffer = BytesIO(df_blob.download_as_bytes())
        df = pd.read_csv(df_buffer)
        
        # Get total rows for pagination
        total_rows = len(df)
        
        # Slice the DataFrame for pagination
        df = df.iloc[start_idx:end_idx]

        # Prepare response
        response = {
            "metadata": metadata,
            "data": {
                "columns": list(df.columns),
                "rows": df.fillna('').to_dict('records'),
                "pagination": {
                    "total_rows": total_rows,
                    "current_page": page,
                    "per_page": per_page,
                    "total_pages": (total_rows + per_page - 1) // per_page
                }
            }
        }

        return response

    except ValueError as e:
        raise ValueError(str(e))
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except Exception as e:
        raise Exception(f"Error fetching table data: {str(e)}")

