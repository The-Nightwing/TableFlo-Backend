from flask import Blueprint, request, jsonify, send_file
import pandas as pd
from io import BytesIO
import os
from firebase_config import get_storage_bucket
import recordlinkage
import numpy as np
import time
import traceback
import json
from datetime import datetime, timezone
from file_upload import get_process_table_data  # Add this line
from models import User, UserProcess, DataFrame, DataFrameOperation, OperationType, MergeSubType, db

merge_files_bp = Blueprint('merge_files', __name__, url_prefix='/api/merge-files')

def load_file_from_firebase(bucket, email, file_name):
    """Load a file from Firebase Storage and return as DataFrame or ExcelFile."""
    blob = bucket.blob(f"{email}/uploaded_files/{file_name}")
    if not blob.exists():
        blob = bucket.blob(f"{email}/processed_files/{file_name}")
        if not blob.exists():
            raise FileNotFoundError(f"File '{file_name}' not found.")
            
    file_content = blob.download_as_bytes()
    extension = os.path.splitext(file_name)[-1].lower()

    if extension in ['.xls', '.xlsx']:
        return pd.ExcelFile(BytesIO(file_content))
    elif extension == '.csv':
        return pd.read_csv(BytesIO(file_content))
    else:
        raise ValueError(f"Unsupported file extension: '{extension}'.")

def parse_file(file):
    """Parse ExcelFile into DataFrame; if CSV, return as is."""
    return file.parse() if isinstance(file, pd.ExcelFile) else file

def perform_horizontal_merge(df1, df2, key_pairs, method, show_count_summary):
    """Perform horizontal merge using key pairs and method."""
    try:
        merged_df = df1

        for key_pair in key_pairs:
            left_key = key_pair.get('left')
            right_key = key_pair.get('right')

            # Validate key pair
            if not left_key or not right_key:
                raise ValueError("Both 'left' and 'right' keys are required in each key pair.")
            
            # Check if columns exist and print column info
            print(f"\nMerging on columns:")
            print(f"Left table columns: {list(df1.columns)}")
            print(f"Right table columns: {list(df2.columns)}")
            
            if left_key not in df1.columns:
                raise ValueError(f"Key '{left_key}' not found in first table columns: {list(df1.columns)}")
            if right_key not in df2.columns:
                raise ValueError(f"Key '{right_key}' not found in second table columns: {list(df2.columns)}")

            # Print data types and sample values before conversion
            print(f"\nBefore conversion:")
            print(f"Left key '{left_key}' dtype: {df1[left_key].dtype}")
            print(f"Left key sample values: {df1[left_key].head()}")
            print(f"Right key '{right_key}' dtype: {df2[right_key].dtype}")
            print(f"Right key sample values: {df2[right_key].head()}")

            # Handle data type conversion based on the data
            try:
                # Try to convert to datetime first if the column contains date-like values
                try:
                    if any(isinstance(x, (pd.Timestamp, datetime)) for x in df1[left_key].dropna()):
                        df1[left_key] = pd.to_datetime(df1[left_key])
                        df2[right_key] = pd.to_datetime(df2[right_key])
                        print("Converting columns to datetime")
                    else:
                        # Try numeric conversion first
                        try:
                            df1[left_key] = pd.to_numeric(df1[left_key], errors='raise')
                            df2[right_key] = pd.to_numeric(df2[right_key], errors='raise')
                            print("Converting columns to numeric")
                        except:
                            # If numeric conversion fails, convert to string
                            df1[left_key] = df1[left_key].fillna('').astype(str).str.strip()
                            df2[right_key] = df2[right_key].fillna('').astype(str).str.strip()
                            print("Converting columns to string")
                except:
                    # If datetime conversion fails, try string conversion
                    df1[left_key] = df1[left_key].fillna('').astype(str).str.strip()
                    df2[right_key] = df2[right_key].fillna('').astype(str).str.strip()
                    print("Converting columns to string (after datetime attempt failed)")

                print(f"\nAfter conversion:")
                print(f"Left key '{left_key}' dtype: {df1[left_key].dtype}")
                print(f"Left key unique values: {df1[left_key].unique()}")
                print(f"Right key '{right_key}' dtype: {df2[right_key].dtype}")
                print(f"Right key unique values: {df2[right_key].unique()}")
                
            except Exception as e:
                raise ValueError(f"Error converting columns to compatible types: {str(e)}")

            # Perform merge
            try:
                merged_df = pd.merge(
                    merged_df,
                    df2,
                    left_on=left_key,
                    right_on=right_key,
                    how=method,
                    indicator=show_count_summary
                )

                # Get the final column order: original df1 + new (non-key) columns from df2
                left_columns = list(df1.columns)
                # Exclude right keys and also avoid duplicates
                right_new_columns = [col for col in df2.columns if col not in right_keys and col not in df1.columns]
                final_column_order = left_columns + right_new_columns

                # Filter merged_df to keep only those columns, if they exist in the result
                merged_df = merged_df[[col for col in final_column_order if col in merged_df.columns]]

                print(f"\nMerge successful. Result shape: {merged_df.shape}")
            except Exception as e:
                # Enhanced error reporting
                error_msg = (
                    f"Error during merge operation with keys '{left_key}' and '{right_key}':\n"
                    f"Error details: {str(e)}\n"
                    f"Left key unique values: {sorted(df1[left_key].unique())}\n"
                    f"Right key unique values: {sorted(df2[right_key].unique())}\n"
                    f"Left key non-matching values: {sorted(set(df1[left_key].unique()) - set(df2[right_key].unique()))}\n"
                    f"Right key non-matching values: {sorted(set(df2[right_key].unique()) - set(df1[left_key].unique()))}\n"
                    f"Common values: {sorted(set(df1[left_key].unique()) & set(df2[right_key].unique()))}"
                )
                raise ValueError(error_msg)

        # Process count summary if requested
        count_summary = None
        if show_count_summary and '_merge' in merged_df.columns:
            count_summary = merged_df['_merge'].value_counts().rename_axis('Match Type').reset_index(name='Count')
            merged_df = merged_df.drop(columns=['_merge'])

        return merged_df, count_summary

    except Exception as e:
        raise ValueError(f"Merge operation failed: {str(e)}")

def perform_vertical_merge(df1, df2):
    """Perform vertical merge by concatenating two dataframes."""
    df1_columns = {col.strip().lower() for col in df1.columns}
    df2_columns = {col.strip().lower() for col in df2.columns}
    missing_in_df1 = df2_columns - df1_columns
    missing_in_df2 = df1_columns - df2_columns

    if missing_in_df1 or missing_in_df2:
        raise ValueError("Column mismatch: file1 and file2 must have the same columns for vertical merge.")
    return pd.concat([df1, df2], ignore_index=True), None

def format_merged_data(merged_df):
    """Format datetime columns and years in the merged dataframe."""
    for col in merged_df.columns:
        if merged_df[col].dtype == 'datetime64[ns]':
            merged_df[col] = merged_df[col].dt.strftime('%Y-%m-%d')
        elif 'year' in col.lower() and merged_df[col].dtype in ['int64', 'float64']:
            merged_df[col] = merged_df[col].astype(int)
    return merged_df

def save_merged_file(bucket, email, merged_df, count_summary, output_file_name):
    """Save merged file, metadata, and previews to Firebase."""
    output_buffer = BytesIO()
    MAX_ROWS_PER_SHEET = 1000000
    total_rows = len(merged_df)
    num_sheets = (total_rows - 1) // MAX_ROWS_PER_SHEET + 1
    metadata = {"sheets": {}}

    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
        if num_sheets == 1:
            merged_df.to_excel(writer, index=False, sheet_name="Merged Data")
            metadata["sheets"]["Merged Data"] = {
                "columns": merged_df.columns.tolist(),
                "columnTypes": {col: str(merged_df[col].dtype) for col in merged_df.columns}
            }
        else:
            for i in range(num_sheets):
                sheet_name = f"Merged Data {i+1}"
                start_idx = i * MAX_ROWS_PER_SHEET
                end_idx = min((i + 1) * MAX_ROWS_PER_SHEET, total_rows)
                chunk_df = merged_df.iloc[start_idx:end_idx]
                
                chunk_df.to_excel(writer, index=False, sheet_name=sheet_name)
                metadata["sheets"][sheet_name] = {
                    "columns": chunk_df.columns.tolist(),
                    "columnTypes": {col: str(chunk_df[col].dtype) for col in chunk_df.columns}
                }

        if count_summary is not None:
            count_summary.to_excel(writer, index=False, sheet_name="Count Summary")
            metadata["sheets"]["Count Summary"] = {
                "columns": count_summary.columns.tolist(),
                "columnTypes": {col: str(count_summary[col].dtype) for col in count_summary.columns}
            }

    # Save merged file
    output_buffer.seek(0)
    blob = bucket.blob(f"{email}/processed_files/{output_file_name}")
    blob.upload_from_file(
        output_buffer,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Save metadata
    metadata_blob = bucket.blob(f"{email}/metadata/{output_file_name}.json")
    metadata_blob.upload_from_string(json.dumps(metadata), content_type='application/json')

    # Save previews
    for sheet_name in metadata["sheets"].keys():
        if sheet_name == "Count Summary":
            preview_df = count_summary
        else:
            preview_df = merged_df.iloc[:50]

        preview_buffer = BytesIO()
        preview_df.to_csv(preview_buffer, index=False)
        preview_buffer.seek(0)
        
        preview_blob = bucket.blob(f"{email}/previews/{output_file_name}/{sheet_name}_preview.csv")
        preview_blob.upload_from_file(preview_buffer, content_type='text/csv')

    return blob.generate_signed_url(expiration=3600, version='v4')

def standardize_column(df, column):
    """Standardize column data types for merging."""
    try:
        # Handle different data types appropriately
        if df[column].dtype == 'datetime64[ns]':
            # For Year column, extract just the year
            if column.lower() == 'year':
                return df[column].dt.year.astype(str)
            # For other datetime columns, use YYYY-MM-DD format
            return df[column].dt.strftime('%Y-%m-%d')
        elif df[column].dtype == 'float64':
            # Keep floats as is, but handle NaN
            return df[column].fillna(0)
        elif df[column].dtype == 'int64':
            # Keep integers as is
            return df[column]
        else:
            # Convert to string and strip whitespace for other types
            return df[column].fillna('').astype(str).str.strip()
    except Exception as e:
        print(f"Error processing column {column}: {str(e)}")
        # Return original column if processing fails
        return df[column]

def process_merge_tables(email, process_id, table1, table2, merge_type, merge_method='inner',
                        key_pairs=None, show_count_summary=False, output_table_name='', existing_df=None):
    """
    Process table merge operation.
    merge_type can be 'horizontal' or 'vertical'
    merge_method can be 'inner', 'outer', 'left', 'right'
    """
    try:
        bucket = get_storage_bucket()

        # Read both DataFrames
        def read_dataframe(df_record):
            csv_blob = bucket.blob(df_record.storage_path)
            if not csv_blob.exists():
                raise FileNotFoundError(f"Data not found for table: {df_record.name}")
            content = csv_blob.download_as_string()
            return pd.read_csv(BytesIO(content))

        df1 = read_dataframe(table1)
        df2 = read_dataframe(table2)

        # Perform merge based on type
        if merge_type == 'horizontal':
            if not key_pairs:
                return {"success": False, "error": "Key pairs are required for horizontal merge"}

            # Validate merge method
            valid_methods = ['inner', 'outer', 'left', 'right']
            if merge_method not in valid_methods:
                return {"success": False, "error": f"Invalid merge method. Must be one of: {', '.join(valid_methods)}"}

            try:
                # Extract left and right keys from key pairs
                left_keys = []
                right_keys = []
                for pair in key_pairs:
                    if 'left' not in pair or 'right' not in pair:
                        return {"success": False, "error": "Each key pair must have 'left' and 'right' keys"}
                    
                    left_key = pair['left']
                    right_key = pair['right']
                    
                    # Validate columns exist
                    if left_key not in df1.columns:
                        return {"success": False, "error": f"Column '{left_key}' not found in first table"}
                    if right_key not in df2.columns:
                        return {"success": False, "error": f"Column '{right_key}' not found in second table"}
                    
                    left_keys.append(left_key)
                    right_keys.append(right_key)

                # Perform merge with specified method
                merged_df = pd.merge(
                    df1, df2,
                    left_on=left_keys,
                    right_on=right_keys,
                    how=merge_method,
                    indicator=show_count_summary
                )

                # Get the final column order: original df1 + new (non-key) columns from df2
                left_columns = list(df1.columns)
                # Exclude right keys and also avoid duplicates
                right_new_columns = [col for col in df2.columns if col not in right_keys and col not in df1.columns]
                final_column_order = left_columns + right_new_columns

                # Filter merged_df to keep only those columns, if they exist in the result
                merged_df = merged_df[[col for col in final_column_order if col in merged_df.columns]]

                # Get merge statistics if requested
                merge_stats = None
                if show_count_summary and '_merge' in merged_df.columns:
                    merge_stats = merged_df['_merge'].value_counts().to_dict()
                    merged_df = merged_df.drop(columns=['_merge'])

                merge_info = {
                    "type": "horizontal",
                    "method": merge_method,
                    "keyPairs": key_pairs,
                    "mergeStats": merge_stats
                }

            except Exception as e:
                return {"success": False, "error": f"Merge operation failed: {str(e)}"}

        elif merge_type == 'vertical':
            # Check if columns match or need alignment
            common_columns = [col for col in df1.columns if col in df2.columns]
            if not common_columns:
                return {"success": False, "error": "No common columns found for vertical merge"}

            # Align columns if needed
            df1_aligned = df1[common_columns]
            df2_aligned = df2[common_columns]

            # Perform vertical merge (concatenation)
            merged_df = pd.concat([df1_aligned, df2_aligned], axis=0, ignore_index=True)
            merged_df = merged_df[common_columns]
            merge_info = {
                "type": "vertical",
                "commonColumns": list(common_columns),
                "droppedColumns": {
                    "table1": list(set(df1.columns) - set(common_columns)),
                    "table2": list(set(df2.columns) - set(common_columns))
                }
            }

        else:
            return {"success": False, "error": f"Invalid merge type: {merge_type}"}

        # Generate storage paths
        base_path = f"{email}/process/{process_id}"
        storage_path = f"{base_path}/dataframes/{output_table_name}.csv"
        metadata_path = f"{base_path}/metadata/{output_table_name}.json"

        # Create metadata
        metadata = {
            "tableName": output_table_name,
            "description": f"Merged table from {table1.name} and {table2.name}",
            "processId": process_id,
            "createdAt": (existing_df.created_at if existing_df else datetime.now(timezone.utc)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "updatedAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "rowCount": len(merged_df),
            "columnCount": len(merged_df.columns),
            "columns": [],
            "operation": {
                "type": "merge",
                "mergeType": merge_type,
                "sources": [
                    {
                        "id": table1.id,
                        "name": table1.name
                    },
                    {
                        "id": table2.id,
                        "name": table2.name
                    }
                ],
                **merge_info  # Include type-specific merge information
            }
        }

        # Add column information
        for column in merged_df.columns:
            column_info = {
                "name": column,
                "type": detect_column_type(merged_df[column])
            }
            metadata["columns"].append(column_info)

        # Save merged DataFrame and metadata
        try:
            # Save DataFrame as CSV
            df_buffer = BytesIO()
            merged_df.to_csv(df_buffer, index=False)
            df_buffer.seek(0)

            # Upload to Firebase
            df_blob = bucket.blob(storage_path)
            df_blob.upload_from_file(df_buffer, content_type='text/csv')

            # Save metadata
            metadata_blob = bucket.blob(metadata_path)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )

            # Update or create DataFrame record
            if existing_df:
                # Update existing DataFrame record
                existing_df.row_count = len(merged_df)
                existing_df.column_count = len(merged_df.columns)
                existing_df.updated_at = datetime.now(timezone.utc)
                df_record = existing_df
                db.session.add(existing_df)  # Add existing record to session
            else:
                # Create new DataFrame record
                df_record = DataFrame.create_from_pandas(
                    df=merged_df,
                    process_id=process_id,
                    name=output_table_name,
                    email=email,
                    storage_path=storage_path,
                    user_id=table1.user_id
                )
                db.session.add(df_record)  # Add new record to session

            # Commit the changes
            db.session.commit()

            return {
                "success": True,
                "message": f"Tables merged successfully into '{output_table_name}'",
                "id": df_record.id,
                "name": output_table_name,
                "rowCount": len(merged_df),
                "columnCount": len(merged_df.columns),
                "metadata": metadata,
                "isUpdate": existing_df is not None
            }

        except Exception as e:
            db.session.rollback()
            # Clean up uploaded files if they exist
            if df_blob.exists():
                df_blob.delete()
            if metadata_blob.exists():
                metadata_blob.delete()
            raise e

    except Exception as e:
        raise Exception(f"Error processing merge: {str(e)}")

@merge_files_bp.route('/merge/', methods=['POST'])
def merge_tables():
    """Endpoint for merging tables within a process."""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        process_id = data.get('processId')
        table1_name = data.get('table1Name')
        table2_name = data.get('table2Name')
        merge_type = data.get('mergeType')
        output_table_name = data.get('outputTableName', '').strip()
        
        if not all([process_id, table1_name, table2_name, merge_type, output_table_name]):
            return jsonify({
                "error": "Missing required parameters: processId, table1Name, table2Name, mergeType, or outputTableName."
            }), 400

        # Get process and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get DataFrame records by name
        df1 = DataFrame.query.filter_by(process_id=process_id, name=table1_name).first()
        df2 = DataFrame.query.filter_by(process_id=process_id, name=table2_name).first()

        if not df1:
            return jsonify({"error": f"Table '{table1_name}' not found in process"}), 404
        if not df2:
            return jsonify({"error": f"Table '{table2_name}' not found in process"}), 404

        # Check if output table name already exists
        existing_df = DataFrame.query.filter_by(
            process_id=process_id,
            name=output_table_name
        ).first()

        # Generate message based on merge type and method
        merge_method = data.get('mergeMethod', 'inner')
        message = f"Merge tables {table1_name} and {table2_name} based on the {merge_method} method"

        # Create DataFrameOperation record with IN_PROGRESS status
        df_operation = DataFrameOperation(
            process_id=process_id,
            dataframe_id=df1.id,  # Using first table as reference
            operation_type=OperationType.MERGE_FILES.value,
            operation_subtype=MergeSubType.HORIZONTAL.value if merge_type == 'horizontal' else MergeSubType.VERTICAL.value,
            payload=data,
            message=message
        )
        df_operation.user_id = user.id
        
        # Save initial operation record
        db.session.add(df_operation)
        db.session.commit()

        try:
            result = process_merge_tables(
                email=email,
                process_id=process_id,
                table1=df1,
                table2=df2,
                merge_type=merge_type,
                merge_method=data.get('mergeMethod', 'inner'),
                key_pairs=data.get('keyPairs', []),
                show_count_summary=data.get('showCountSummary', False),
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
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def clean_dataframe(df):
    """
    Remove columns where:
    - The column name is 'Unnamed' or None.
    - All rows in the column are empty (NaN or None).
    """
    # Identify valid columns where the column name is not 'Unnamed' or None, and the column is not entirely empty
    valid_columns = [
        col for col in df.columns 
        if col is not None and not str(col).startswith("Unnamed") and not df[col].isna().all()
    ]
    # Filter the DataFrame to include only valid columns
    return df[valid_columns]

@merge_files_bp.route('/preview/', methods=['POST'])
def preview_file():
    """
    Generate a preview of the merged file using saved preview files from Firebase.
    """
    try:
        # Extract and validate input
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Extract request data
        file1_name = data.get('file1Name')
        file2_name = data.get('file2Name')
        merge_type = data.get('mergeType')
        file1_sheet_name = data.get('file1SheetName')
        file2_sheet_name = data.get('file2SheetName')
        key_pairs = data.get('keyPairs', [])
        merge_method = data.get('mergeMethod', 'inner')
        show_count_summary = data.get('showCountSummary', False)

        if not file1_name or not file2_name or not merge_type:
            return jsonify({"error": "Missing required inputs"}), 400

        # Load files from Firebase
        bucket = get_storage_bucket()

        def load_file(file_name, sheet_name=None):
            """Load file from Firebase and return as DataFrame"""
            blob = bucket.blob(f"{email}/uploaded_files/{file_name}")
            if not blob.exists():
                blob = bucket.blob(f"{email}/processed_files/{file_name}")
                if not blob.exists():
                    raise FileNotFoundError(f"File '{file_name}' not found.")
            
            file_content = blob.download_as_bytes()
            if file_name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(BytesIO(file_content), sheet_name=sheet_name)
            elif file_name.endswith('.csv'):
                return pd.read_csv(BytesIO(file_content))
            else:
                raise ValueError(f"Unsupported file format for {file_name}")

        # Load the files with specified sheets
        df1 = load_file(file1_name, file1_sheet_name)
        df2 = load_file(file2_name, file2_sheet_name)

        if merge_type.lower() == "horizontal":
            if not key_pairs:
                return jsonify({"error": "Key pairs required for horizontal merge."}), 400

            merged_df = df1
            for key_pair in key_pairs:
                left_key = key_pair.get("left")
                right_key = key_pair.get("right")

                if left_key not in df1.columns or right_key not in df2.columns:
                    return jsonify({"error": f"Invalid key pair: {left_key}, {right_key}"}), 400

                df1[left_key] = df1[left_key].astype(str)
                df2[right_key] = df2[right_key].astype(str)

                merged_df = pd.merge(
                    merged_df,
                    df2,
                    left_on=left_key,
                    right_on=right_key,
                    how=merge_method,
                    indicator=show_count_summary
                )

                # Get the final column order: original df1 + new (non-key) columns from df2
                left_columns = list(df1.columns)
                # Exclude right keys and also avoid duplicates
                right_new_columns = [col for col in df2.columns if col not in right_keys and col not in df1.columns]
                final_column_order = left_columns + right_new_columns

                # Filter merged_df to keep only those columns, if they exist in the result
                merged_df = merged_df[[col for col in final_column_order if col in merged_df.columns]]

            count_summary = {}
            if show_count_summary and "_merge" in merged_df.columns:
                count_summary = merged_df["_merge"].value_counts().to_dict()
                merged_df = merged_df.drop(columns=["_merge"])

        elif merge_type.lower() == "vertical":
            df1 = clean_dataframe(df1)
            df2 = clean_dataframe(df2)
            missing_in_df1 = set(df2.columns) - set(df1.columns)
            missing_in_df2 = set(df1.columns) - set(df2.columns)

            if missing_in_df1 or missing_in_df2:
                return jsonify({
                    "error": "Columns of the two files do not match for vertical merge.",
                    "missingInFile1": list(missing_in_df1),
                    "missingInFile2": list(missing_in_df2),
                }), 400

            merged_df = pd.concat([df1, df2], ignore_index=True)
            count_summary = {}

        else:
            return jsonify({"error": "Invalid merge type. Must be 'horizontal' or 'vertical'."}), 400

        # Format datetime columns
        for col in merged_df.columns:
            if merged_df[col].dtype == 'datetime64[ns]':
                merged_df[col] = merged_df[col].dt.strftime('%Y-%m-%d')
            elif 'year' in col.lower() and merged_df[col].dtype in ['int64', 'float64']:
                merged_df[col] = merged_df[col].astype(int)

        merged_df = merged_df.fillna("-")

        to_show = 25 if merge_type.lower() == "horizontal" else 50
        preview_data = {
            "sheetName": "Merged Data",
            "columns": list(merged_df.columns),
            "rows": merged_df.head(to_show).to_dict(orient="records")
        }

        return jsonify({
            "success": True,
            "previewData": [preview_data],
            "countSummary": count_summary,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def normalize_input_data(data):
    """Normalize input data formats."""
    # Normalize reconciliation settings
    if 'reconciliation_settings' in data:
        settings = data['reconciliation_settings']
        if settings is None:
            raise ValueError("Reconciliation settings cannot be None.")

        based_on_columns = settings.get('based_on_columns') or {}
        if not isinstance(based_on_columns, dict):
            based_on_columns = {}

        data['settings'] = {
            'method': settings.get('method', 'one-to-one'),
            'duplicate': settings.get('handling_duplicate_matches', 'first_occurrence') or 'first_occurrence',
            'basis_column': {
                'file1': based_on_columns.get('file1', ''),
                'file2': based_on_columns.get('file2', '')
            },
            'fuzzy_preference': []
        }

    # Normalize settings
    if 'settings' in data:
        settings = data['settings']
        if settings is None:
            raise ValueError("Settings cannot be None.")

        if 'duplicate' not in settings:
            settings['duplicate'] = settings.get('duplicateHandling', 'first_occurrence') or 'first_occurrence'
        if 'basis_column' not in settings:
            settings['basis_column'] = {
                'file1': settings.get('baseColumn1', ''),
                'file2': settings.get('baseColumn2', '')
            }
        if 'fuzzy_preference' not in settings:
            settings['fuzzy_preference'] = []

    # Normalize matching keys
    if 'matching_keys' in data:
        data['keys'] = [{
            'file1': key.get('file1_column'),
            'file2': key.get('file2_column'),
            'criteria': key.get('method', 'exact'),
            'case_sensitive': 'yes' if key.get('case_sensitive') else 'no',
            'ignore_special': 'yes' if key.get('ignore_special_characters') else 'no'
        } for key in data['matching_keys']]

    # Normalize values
    if 'values' in data:
        normalized_values = []
        for value in data['values']:
            normalized_value = {
                'file1': value.get('file1_column'),
                'file2': value.get('file2_column') or value.get('file2', ''),
                'threshold_type': value.get('threshold_type', '').lower() if value.get('threshold_type') else None,
                'threshold_value': value.get('threshold_value')
            }
            if normalized_value['threshold_type'] == 'percent':
                normalized_value['threshold_type'] = 'percent'
            elif normalized_value['threshold_type'] == 'amount':
                normalized_value['threshold_type'] = 'amount'
            normalized_values.append(normalized_value)
        data['values'] = normalized_values

    # Normalize cross_reference
    if isinstance(data.get('cross_reference'), dict):
        data['cross_reference'] = [{
            'file1': data['cross_reference'].get('file1_column'),
            'file2': data['cross_reference'].get('file2_column', 0)
        }]
    elif not isinstance(data.get('cross_reference'), list):
        data['cross_reference'] = []

def load_file_from_firebase(bucket, email, file_info):
    """Load file from Firebase Storage."""
    file_name = file_info.get('file_name')
    sheet_name = file_info.get('sheet_name')
    blob = bucket.blob(f"{email}/uploaded_files/{file_name}")
    if not blob.exists():
        blob = bucket.blob(f"{email}/processed_files/{file_name}")
        if not blob.exists():
            raise FileNotFoundError(f"File '{file_name}' not found.")
    file_content = blob.download_as_bytes()
    return pd.read_excel(BytesIO(file_content), sheet_name=sheet_name)

def save_reconciled_files(email, df1, df2, data):
    """Save reconciled files to Firebase."""
    output_buffer = BytesIO()
    with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name=data['files'][0]['sheet_name'], index=False)
        df2.to_excel(writer, sheet_name=data['files'][1]['sheet_name'], index=False)

    output_buffer.seek(0)
    output_file_name = f"{data['output_file']}.xlsx"
    bucket = get_storage_bucket()
    blob = bucket.blob(f"{email}/processed_files/{output_file_name}")
    blob.upload_from_file(
        output_buffer,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return blob.generate_signed_url(expiration=3600), output_file_name

def process_reconciliation(email, data):
    """Core logic for reconciling files."""
    try:
        # Normalize input data
        normalize_input_data(data)

        # Load files
        bucket = get_storage_bucket()
        df1 = load_file_from_firebase(bucket, email, data['files'][0])
        df2 = load_file_from_firebase(bucket, email, data['files'][1])

        # Replace blanks
        df1.fillna('', inplace=True)
        df2.fillna('', inplace=True)

        # Handle non-case sensitive keys and ignore special characters
        for key in data['keys']:
            if key['case_sensitive'] == 'no':
                # Convert to string and handle NaN values
                df1[key['file1']] = df1[key['file1']].astype(str).fillna('')
                df2[key['file2']] = df2[key['file2']].astype(str).fillna('')
                df1[key['file1']] = df1[key['file1']].str.upper()
                df2[key['file2']] = df2[key['file2']].str.upper()
            if key['ignore_special'] == 'yes':
                # Convert to string and handle NaN values
                df1[key['file1']] = df1[key['file1']].astype(str).fillna('')
                df2[key['file2']] = df2[key['file2']].astype(str).fillna('')
                df1[key['file1']] = df1[key['file1']].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
                df2[key['file2']] = df2[key['file2']].str.replace(r'[^a-zA-Z0-9]', '', regex=True)

        # Initialize lists for keys
        left_keys = []
        right_keys = []
        left_block = []
        right_block = []
        fuzzy_keys = []

        for key in data['keys']:
            left_keys.append(key['file1'])
            right_keys.append(key['file2'])
            if key['criteria'] == 'exact':
                left_block.append(key['file1'])
                right_block.append(key['file2'])
            if key['criteria'] == 'fuzzy':
                fuzzy_keys.append(key['file1'] + '-' + key['file2'])

        # Debugging: Log the right_keys and df2 columns
        print("Right Keys:", right_keys)
        print("Columns in df2:", df2.columns.tolist())

        # Check for empty keys
        if any(key == '' for key in right_keys):
            return jsonify({"error": "One or more keys in 'right_keys' are empty."}), 400

        # Group by keys based on the method specified in settings
        agg_dict = {}
        if data['settings']['method'] == 'one-to-many':
            for value in data['values']:
                agg_dict[value['file2']] = 'sum'
            df2 = df2.groupby(right_keys).agg(agg_dict).reset_index()
            for ref in data['cross_reference']:
                if ref['file2'] == 0:
                    df2['Group-ID'] = 'Group-' + (df2.index + 1).astype(str)
                    ref['file2'] = 'Group-ID'

        elif data['settings']['method'] == 'many-to-one':
            for value in data['values']:
                agg_dict[value['file1']] = 'sum'
            df1 = df1.groupby(left_keys).agg(agg_dict).reset_index()
            for ref in data['cross_reference']:
                if ref['file1'] == 0:
                    df1['Group-ID'] = 'Group-' + (df1.index + 1).astype(str)
                    ref['file1'] = 'Group-ID'

        elif data['settings']['method'] == 'many-to-many':
            agg_dict_left = {}
            agg_dict_right = {}
            for value in data['values']:
                agg_dict_left[value['file1']] = 'sum'
                agg_dict_right[value['file2']] = 'sum'
            df1 = df1.groupby(left_keys).agg(agg_dict_left).reset_index()
            df2 = df2.groupby(right_keys).agg(agg_dict_right).reset_index()
            for ref in data['cross_reference']:
                if ref['file2'] == 0:
                    df2['Group-ID'] = 'Group-' + (df2.index + 1).astype(str)
                    ref['file2'] = 'Group-ID'
                if ref['file1'] == 0:
                    df1['Group-ID'] = 'Group-' + (df1.index + 1).astype(str)
                    ref['file1'] = 'Group-ID'

        # Initialize lists for required columns, filtering out None and empty values
        required_columns_file1 = [
            col for col in 
            list(set(left_keys + 
                    [value.get('file1') for value in data['values'] if value.get('file1')] + 
                    [ref.get('file1') for ref in data['cross_reference'] if ref.get('file1')]))
            if col is not None and col != ''
        ]
        
        required_columns_file2 = [
            col for col in 
            list(set(right_keys + 
                    [value.get('file2') for value in data['values'] if value.get('file2')] + 
                    [ref.get('file2') for ref in data['cross_reference'] if ref.get('file2')]))
            if col is not None and col != '' and col != 0
        ]

        # Verify that all required columns exist in the dataframes
        missing_cols_df1 = [col for col in required_columns_file1 if col not in df1.columns]
        missing_cols_df2 = [col for col in required_columns_file2 if col not in df2.columns]
        
        if missing_cols_df1 or missing_cols_df2:
            error_msg = []
            if missing_cols_df1:
                error_msg.append(f"Columns missing in first file: {', '.join(missing_cols_df1)}")
            if missing_cols_df2:
                error_msg.append(f"Columns missing in second file: {', '.join(missing_cols_df2)}")
            return jsonify({"error": " | ".join(error_msg)}), 400

        # Ensure all keys and columns are valid
        def validate_keys_and_columns(df, keys, context):
            for key in keys:
                if key is None or key not in df.columns:
                    raise KeyError(f"Invalid key '{key}' in {context}. Ensure all keys are valid and exist in the DataFrame.")

        # Validate keys and columns for both DataFrames
        validate_keys_and_columns(df1, left_keys, "left_keys")
        validate_keys_and_columns(df2, right_keys, "right_keys")

        # Reconciliation
        indexer = recordlinkage.Index()
        indexer.block(left_on=left_block, right_on=right_block)
        comparisons = indexer.index(df1, df2)
        compare = recordlinkage.Compare()
        for key in data['keys']:
            if key['criteria'] == 'exact':
                compare.exact(key['file1'], key['file2'], label=key['file1'] + '-' + key['file2'])
            elif key['criteria'] == 'fuzzy':
                compare.string(key['file1'], key['file2'], method='jarowinkler', threshold=0.85, label=key['file1'] + '-' + key['file2'])
        
        result = compare.compute(comparisons, df1, df2).reset_index()

        # Merge required columns with reconciliation result
        result = result.merge(df1[required_columns_file1], left_on='level_0', right_index=True, how='left', suffixes=('-1', '-2'))
        result = result.merge(df2[required_columns_file2], left_on='level_1', right_index=True, how='left', suffixes=('-1', '-2'))

        for fuzz in fuzzy_keys:
            result = result[result[fuzz] >= 0.8]

        # Initialize new columns to both dataframes for cross referencing, getting the difference in the values and reco status
        for ref in data['cross_reference']:
            if ref['file1'] == ref['file2']:
                df1[ref['file2'] + '-2'] = np.nan  # Ensure this column is created
                df2[ref['file1'] + '-1'] = np.nan  # Ensure this column is created
            else:
                df1[ref['file2']] = np.nan  # Ensure this column is created
                df2[ref['file1']] = np.nan  # Ensure this column is created

        df1['Reco_Status'] = np.nan
        df2['Reco_Status'] = np.nan

        # Cross-reference the best match between the two dataframes
        while len(result) > 0:
            for ref in data['cross_reference']:
                # Validate cross-reference keys
                if ref['file1'] is None or ref['file2'] is None:
                    continue  # Skip invalid references

                if ref['file1'] not in df1.columns or ref['file2'] not in df2.columns:
                    continue  # Skip if columns do not exist

                if ref['file1'] == ref['file2']:
                    df1.loc[result['level_0'].iloc[0], ref['file2'] + '-2'] = df2.loc[result['level_1'].iloc[0], ref['file2']]
                    df2.loc[result['level_1'].iloc[0], ref['file1'] + '-1'] = df1.loc[result['level_0'].iloc[0], ref['file1']]
                else:
                    df1.loc[result['level_0'].iloc[0], ref['file2']] = df2.loc[result['level_1'].iloc[0], ref['file2']]
                    df2.loc[result['level_1'].iloc[0], ref['file1']] = df1.loc[result['level_0'].iloc[0], ref['file1']]

            df1.loc[result['level_0'].iloc[0], 'Reco_Status'] = 'Matched'
            df2.loc[result['level_1'].iloc[0], 'Reco_Status'] = 'Matched'

            for value in data['values']:
                if value['file1'] == value['file2']:
                    df1.loc[result['level_0'].iloc[0], value['file2'] + '-2'] = df2.loc[result['level_1'].iloc[0], value['file2']]
                    df2.loc[result['level_1'].iloc[0], value['file1'] + '-1'] = df1.loc[result['level_0'].iloc[0], value['file1']]
                else:
                    df1.loc[result['level_0'].iloc[0], value['file2']] = df2.loc[result['level_1'].iloc[0], value['file2']]
                    df2.loc[result['level_1'].iloc[0], value['file1']] = df1.loc[result['level_0'].iloc[0], value['file1']]

            result = result[(result['level_0'] != result['level_0'].iloc[0]) & (result['level_1'] != result['level_1'].iloc[0])]

        # Get the difference in the values
        for value in data['values']:
            if value['file1'] == value['file2']:
                df1[value['file1'] + '-' + 'Difference'] = abs(df1[value['file1']] - df1[value['file2'] + '-2'])
                df2[value['file2'] + '-' + 'Difference'] = abs(df2[value['file2']] - df2[value['file1'] + '-1'])

                # Ensure the columns exist before accessing them
                if value['threshold_type'] == 'percent':
                    df1[value['file1'] + '-' + 'Match'] = np.where(
                        abs(df1[value['file1'] + '-' + 'Difference']) <= abs((df1[value['file1']] + df1[value['file2'] + '-2']) / 2 * value['threshold_value'] / 100),
                        'Matched', 'Unmatched'
                    )
                    df2[value['file2'] + '-' + 'Match'] = np.where(
                        abs(df2[value['file2'] + '-' + 'Difference']) <= abs((df2[value['file1'] + '-1'] + df2[value['file2']]) / 2 * value['threshold_value'] / 100),
                        'Matched', 'Unmatched'
                    )
                elif value['threshold_type'] == 'amount':
                    df1[value['file1'] + '-' + 'Match'] = np.where(
                        abs(df1[value['file1'] + '-' + 'Difference']) <= abs(value['threshold_value']),
                        'Matched', 'Unmatched'
                    )
                    df2[value['file2'] + '-' + 'Match'] = np.where(
                        abs(df2[value['file2'] + '-' + 'Difference']) <= abs(value['threshold_value']),
                        'Matched', 'Unmatched'
                    )
                else:
                    df1[value['file1'] + '-' + 'Match'] = np.where(
                        abs(df1[value['file1'] + '-' + 'Difference']) == 0,
                        'Matched', 'Unmatched'
                    )
                    df2[value['file2'] + '-' + 'Match'] = np.where(
                        abs(df2[value['file2'] + '-' + 'Difference']) == 0,
                        'Matched', 'Unmatched'
                    )
            else:
                df1[value['file1'] + '-' + 'Difference'] = abs(df1[value['file1']] - df1[value['file2']])
                df2[value['file2'] + '-' + 'Difference'] = abs(df2[value['file2']] - df2[value['file1']])

                # Ensure the columns exist before accessing them
                if value['threshold_type'] == 'percent':
                    df1[value['file1'] + '-' + 'Match'] = np.where(
                        abs(df1[value['file1'] + '-' + 'Difference']) <= abs((df1[value['file1']] + df1[value['file2']]) / 2 * value['threshold_value'] / 100),
                        'Matched', 'Unmatched'
                    )
                    df2[value['file2'] + '-' + 'Match'] = np.where(
                        abs(df2[value['file2'] + '-' + 'Difference']) <= abs((df2[value['file1']] + df2[value['file2']]) / 2 * value['threshold_value'] / 100),
                        'Matched', 'Unmatched'
                    )
                elif value['threshold_type'] == 'amount':
                    df1[value['file1'] + '-' + 'Match'] = np.where(
                        abs(df1[value['file1'] + '-' + 'Difference']) <= abs(value['threshold_value']),
                        'Matched', 'Unmatched'
                    )
                    df2[value['file2'] + '-' + 'Match'] = np.where(
                        abs(df2[value['file2'] + '-' + 'Difference']) <= abs(value['threshold_value']),
                        'Matched', 'Unmatched'
                    )
                else:
                    df1[value['file1'] + '-' + 'Match'] = np.where(
                        abs(df1[value['file1'] + '-' + 'Difference']) == 0,
                        'Matched', 'Unmatched'
                    )
                    df2[value['file2'] + '-' + 'Match'] = np.where(
                        abs(df2[value['file2'] + '-' + 'Difference']) == 0,
                        'Matched', 'Unmatched'
                    )

        # Save results
        download_url, output_file_name = save_reconciled_files(email, df1, df2, data)

        return {
            "success": True,
            "message": "Files reconciled successfully.",
            "downloadUrl": download_url,
            "fileName": output_file_name
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@merge_files_bp.route('/process/reconcile/', methods=['POST'])
def reconcile_files():
    """Endpoint for reconciling files within a process."""
    try:
        data = request.json
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected a JSON object."}), 400

        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        required_fields = [
            'processId', 
            'sourceTableNames',  # List of table names to reconcile
            'keys',             # Key columns for matching
            'values',           # Value columns to compare
            'settings',         # Reconciliation settings
            'crossReference',   # Cross-reference rules
            'outputTableName'   # Name for the reconciled table
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        process_id = data.get('processId')
        source_table_names = data.get('sourceTableNames')
        output_table_name = data.get('outputTableName', '').strip()

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get source DataFrame records
        source_dfs = []
        for table_name in source_table_names:
            df = DataFrame.query.filter_by(
                process_id=process_id,
                name=table_name
            ).first()
            if not df:
                return jsonify({"error": f"Table '{table_name}' not found in process"}), 404
            source_dfs.append(df)

        # Update basis_table in settings based on source table names
        if 'settings' in data and 'basis_table' in data['settings']:
            basis_table_name = data['settings']['basis_table']
            if basis_table_name == source_table_names[0]:
                data['settings']['basis_table'] = 'left'
            elif basis_table_name == source_table_names[1]:
                data['settings']['basis_table'] = 'right'
            else:
                return jsonify({"error": f"Basis table '{basis_table_name}' not found in source tables"}), 400

        # Check if output table name already exists
        existing_df = DataFrame.query.filter_by(
            process_id=process_id,
            name=output_table_name
        ).first()

        # Generate message for reconciliation operation
        method = data.get('settings', {}).get('method', 'one-to-one')
        message = f"Reconcile tables {source_table_names[0]} and {source_table_names[1]} on the specified columns using {method} method"

        # Create DataFrameOperation record
        df_operation = DataFrameOperation(
            process_id=process_id,
            dataframe_id=source_dfs[0].id,  # Use first source table as reference
            operation_type=OperationType.RECONCILE_FILES.value,
            payload={
                **data,
                'sourceTables': [{'id': df.id, 'name': df.name} for df in source_dfs]
            },
            message=message
        )
        
        # Save initial operation record
        db.session.add(df_operation)
        db.session.commit()

        # Add validation for matching keys
        exact_match_count = sum(1 for key in data['keys'] if key.get('criteria') == 'exact')
        if exact_match_count == 0:
            return jsonify({"error": "At least one exact match key is required"}), 400

        # Validate fuzzy ranking basis
        fuzzy_keys = [key for key in data['keys'] if key.get('criteria') == 'fuzzy']
        if len(fuzzy_keys) > 1:
            ranking_basis = sum(1 for key in fuzzy_keys if key.get('fuzzy_ranking'))
            if ranking_basis != 1:
                return jsonify({"error": "One fuzzy key must be selected as ranking basis"}), 400

        # Validate reconciliation settings for one-to-one method
        if data.get('settings', {}).get('method') == 'one-to-one':
            duplicate_settings = data.get('settings', {}).get('duplicate')
            if not duplicate_settings:
                return jsonify({"error": "Duplicate matching settings required for one-to-one method"}), 400

        try:
            # Process the reconciliation
            result = process_dataframe_reconciliation(
                email=email,
                process_id=process_id,
                source_dfs=source_dfs,
                keys=data['keys'],
                values=data['values'],
                settings=data['settings'],
                cross_reference=data['crossReference'],
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
            result.update({
                'operationId': df_operation.id,
                'operationStatus': df_operation.status,
                'processId': process_id,
                'sourceTables': [
                    {'id': df.id, 'name': df.name} 
                    for df in source_dfs
                ],
                'message': message
            })

            return jsonify(result)

        except Exception as e:
            df_operation.set_error(str(e))
            db.session.commit()
            traceback.print_exc()
            return jsonify({
                "error": f"An unexpected error occurred: {str(e)}",
                "operationId": df_operation.id,
                "operationStatus": "ERROR"
            }), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def detect_column_type(series):
    """Helper function to detect column type including boolean, numeric, date, and string types"""
    try:
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'string'
        
        # Check for boolean
        unique_values = set(non_null.astype(str).str.lower())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
        if unique_values.issubset(boolean_values):
            return 'boolean'
        
        # Check for numeric
        try:
            if all(non_null.astype(float).apply(lambda x: x.is_integer())):
                return 'integer'
            pd.to_numeric(non_null)
            return 'float'
        except:
            # Check for date
            try:
                pd.to_datetime(non_null)
                return 'date'
            except:
                return 'string'
    except:
        return 'string'

def process_dataframe_reconciliation(email, process_id, source_dfs, keys, values, settings, cross_reference, output_table_name, existing_df=None):
    """Process reconciliation between two DataFrames."""
    try:
        bucket = get_storage_bucket()

        # Read DataFrames from storage
        def read_dataframe(df_record):
            content = bucket.blob(df_record.storage_path).download_as_string()
            return pd.read_csv(BytesIO(content))

        df_left = read_dataframe(source_dfs[0])
        df_right = read_dataframe(source_dfs[1])

        # Replace blanks based on data type - use consistent logic
        for col in df_left.columns:
            if pd.api.types.is_numeric_dtype(df_left[col]):
                df_left[col] = df_left[col].fillna(0)  # Use 0 for numeric NaN values
            elif pd.api.types.is_datetime64_any_dtype(df_left[col]):
                df_left[col] = df_left[col].fillna(pd.Timestamp('1900-01-01'))  # Use default date for date NaN values
            else:
                df_left[col] = df_left[col].fillna('')  # Use empty string for string NaN values
                
        for col in df_right.columns:
            if pd.api.types.is_numeric_dtype(df_right[col]):
                df_right[col] = df_right[col].fillna(0)  # Use 0 for numeric NaN values
            elif pd.api.types.is_datetime64_any_dtype(df_right[col]):
                df_right[col] = df_right[col].fillna(pd.Timestamp('1900-01-01'))  # Use default date for date NaN values
            else:
                df_right[col] = df_right[col].fillna('')  # Use empty string for string NaN values

        # Handle non-case sensitive keys and special characters
        for key in keys:
            if key.get('case_sensitive') == 'no':
                df_left[key['left']] = df_left[key['left']].astype(str).str.upper()
                df_right[key['right']] = df_right[key['right']].astype(str).str.upper()
            if key.get('ignore_special') == 'yes':
                df_left[key['left']] = df_left[key['left']].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
                df_right[key['right']] = df_right[key['right']].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)

        # Initialize key lists
        left_keys = [k['left'] for k in keys]
        right_keys = [k['right'] for k in keys]
        left_block = [k['left'] for k in keys if k.get('criteria') == 'exact']
        right_block = [k['right'] for k in keys if k.get('criteria') == 'exact']
        fuzzy_keys = [f"{k['left']}-{k['right']}" for k in keys if k.get('criteria') == 'fuzzy']

        # Handle fuzzy ranking basis
        fuzzy_ranking = []
        fuzzy_key_found = False
        for key in keys:
            if key.get('criteria') == 'fuzzy':
                if not fuzzy_key_found:
                    key['fuzzy_ranking_basis'] = True
                    fuzzy_ranking.append(f"{key['left']}-{key['right']}")
                    fuzzy_key_found = True
                else:
                    key['fuzzy_ranking_basis'] = False

        # Handle grouping based on method
        method = settings.get('method', 'one-to-one')

        if method == 'one-to-many':
            agg_dict = {v['right']: 'sum' for v in values}
            df_right = df_right.groupby(right_keys).agg(agg_dict).reset_index()

        elif method == 'many-to-one':
            agg_dict = {v['left']: 'sum' for v in values}
            df_left = df_left.groupby(left_keys).agg(agg_dict).reset_index()

        elif method == 'many-to-many':
            agg_dict_left = {v['left']: 'sum' for v in values}
            agg_dict_right = {v['right']: 'sum' for v in values}
            df_left = df_left.groupby(left_keys).agg(agg_dict_left).reset_index()
            df_right = df_right.groupby(right_keys).agg(agg_dict_right).reset_index()

        # Handle Custom IDs
        if cross_reference:
            if '__Custom__' in cross_reference.get('left', []):
                df_left['Custom-ID'] = source_dfs[0].name + '-' + (df_left.index + 1).astype(str)
                cross_reference['left'] = ['Custom-ID' if x == '__Custom__' else x for x in cross_reference['left']]

            if '__Custom__' in cross_reference.get('right', []):
                df_right['Custom-ID'] = source_dfs[1].name + '-' + (df_right.index + 1).astype(str)
                cross_reference['right'] = ['Custom-ID' if x == '__Custom__' else x for x in cross_reference['right']]

        # Validate cross-reference columns based on method
        if cross_reference:
            # Get value columns
            value_cols_left = [v['left'] for v in values]
            value_cols_right = [v['right'] for v in values]

            # Validate left table cross-references
            if method in ['one-to-many', 'many-to-many']:
                # For 'many' side, only allow key columns
                invalid_left_refs = [ref for ref in cross_reference.get('left', []) 
                                   if ref not in left_keys and ref != '__Custom__']
                if invalid_left_refs:
                    raise ValueError(f"For {method} method, left table cross-references must be from key columns only. Invalid references: {invalid_left_refs}")
            else:
                # For 'one' side, allow any column except value columns
                invalid_left_refs = [ref for ref in cross_reference.get('left', []) 
                                   if ref in value_cols_left]
                if invalid_left_refs:
                    raise ValueError(f"Left table cross-references cannot include value columns. Invalid references: {invalid_left_refs}")

            # Validate right table cross-references
            if method in ['many-to-one', 'many-to-many']:
                # For 'many' side, only allow key columns
                invalid_right_refs = [ref for ref in cross_reference.get('right', []) 
                                    if ref not in right_keys and ref != '__Custom__']
                if invalid_right_refs:
                    raise ValueError(f"For {method} method, right table cross-references must be from key columns only. Invalid references: {invalid_right_refs}")
            else:
                # For 'one' side, allow any column except value columns
                invalid_right_refs = [ref for ref in cross_reference.get('right', []) 
                                    if ref in value_cols_right]
                if invalid_right_refs:
                    raise ValueError(f"Right table cross-references cannot include value columns. Invalid references: {invalid_right_refs}")

        # Get required columns and filter out None values
        required_columns_left = list(filter(None, set(
            left_keys +
            [v['left'] for v in values] +
            (cross_reference.get('left', []) if cross_reference else []) +
            ([settings.get('basis_column', {}).get('left')] if method == 'one-to-one' else [])
        )))
        required_columns_right = list(filter(None, set(
            right_keys +
            [v['right'] for v in values] +
            (cross_reference.get('right', []) if cross_reference else []) +
            ([settings.get('basis_column', {}).get('right')] if method == 'one-to-one' else [])
        )))

        # Validate that all required columns exist
        missing_cols_left = [col for col in required_columns_left if col not in df_left.columns]
        missing_cols_right = [col for col in required_columns_right if col not in df_right.columns]
        
        if missing_cols_left or missing_cols_right:
            error_msg = []
            if missing_cols_left:
                error_msg.append(f"Columns missing in left table: {', '.join(missing_cols_left)}")
            if missing_cols_right:
                error_msg.append(f"Columns missing in right table: {', '.join(missing_cols_right)}")
            raise ValueError(" | ".join(error_msg))

        # Perform reconciliation
        indexer = recordlinkage.Index()
        indexer.block(left_on=left_block, right_on=right_block)
        comparisons = indexer.index(df_left, df_right)
        
        compare = recordlinkage.Compare()
        for key in keys:
            label = f"{key['left']}-{key['right']}"
            if key['criteria'] == 'exact':
                compare.exact(key['left'], key['right'], label=label)
            elif key['criteria'] == 'fuzzy':
                compare.string(key['left'], key['right'], method='jarowinkler', 
                             threshold=0.85, label=label)

        result = compare.compute(comparisons, df_left, df_right)
        
        # Reset index and ensure no duplicate labels
        result = result.reset_index()
        
        # Merge required columns with reconciliation result
        result = result.merge(
            df_left[required_columns_left].reset_index(), 
            left_on='level_0', 
            right_on='index',
            how='left',
            suffixes=('-1', '-2')
        ).drop('index', axis=1)
        
        result = result.merge(
            df_right[required_columns_right].reset_index(),
            left_on='level_1',
            right_on='index',
            how='left',
            suffixes=('-1', '-2')
        ).drop('index', axis=1)

        # Apply fuzzy matching threshold for each fuzzy key
        for fuzz in fuzzy_keys:
            if fuzz in result.columns:
                result = result[result[fuzz] >= 0.8].reset_index(drop=True)

        # Handle duplicate matching logic for one-to-one method
        if method == 'one-to-one' and settings.get('duplicate'):
            handle_duplicate_matching(result, settings, fuzzy_ranking)

        # Initialize reconciliation columns
        initialize_reconciliation_columns(df_left, df_right, cross_reference or {}, values)

        # Process matches
        process_matches(result, df_left, df_right, cross_reference or {}, values)

        # Calculate value differences and matches
        calculate_value_differences(df_left, df_right, values)

        # Save results
        storage_path = f"{email}/process/{process_id}/dataframes/{output_table_name}.csv"
        
        # Combine results into a single DataFrame
        result_df = pd.concat([df_left, df_right], axis=0, ignore_index=True)
        
        # Save to storage
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        bucket.blob(storage_path).upload_from_file(
            csv_buffer,
            content_type='text/csv'
        )

        # Generate and store metadata
        metadata = {
            'tableName': output_table_name,
            'description': f'Reconciliation result of {source_dfs[0].name} and {source_dfs[1].name}',
            'sourceFileId': [df.id for df in source_dfs],
            'columns': [
                {
                    'name': col,
                    'type': str(result_df[col].dtype)
                }
                for col in result_df.columns
            ],
            'summary': {
                'nullCounts': result_df.isnull().sum().to_dict(),
                'uniqueCounts': result_df.nunique().to_dict()
            }
        }

        # Save metadata to storage
        metadata_path = f"{email}/process/{process_id}/metadata/{output_table_name}.json"
        metadata_blob = bucket.blob(metadata_path)
        metadata_blob.upload_from_string(
            json.dumps(metadata),
            content_type='application/json'
        )

        # Create or update DataFrame record
        if existing_df:
            # Update existing DataFrame record
            existing_df.row_count = len(result_df)
            existing_df.column_count = len(result_df.columns)
            existing_df.updated_at = datetime.now(timezone.utc)
            df_record = existing_df
            db.session.add(existing_df)  # Add existing record to session
        else:
            # Create new DataFrame record
            df_record = DataFrame.create_from_pandas(
                df=result_df,
                process_id=process_id,
                name=output_table_name,
                email=email,
                storage_path=storage_path,
                user_id=source_dfs[0].user_id
            )
            db.session.add(df_record)  # Add new record to session

        # Commit the changes
        db.session.commit()

        return {
            'success': True,
            'storage_path': storage_path,
            'row_count': len(result_df),
            'column_count': len(result_df.columns),
            'statistics': {
                'matched_count': len(df_left[df_left['Reco_Status'] == 'Matched']),
                'unmatched_count': len(df_left[df_left['Reco_Status'].isna()]),
                'source_tables': [
                    {'id': df.id, 'name': df.name, 'row_count': len(read_dataframe(df))}
                    for df in source_dfs
                ]
            }
        }

    except Exception as e:
        print(f"Error in process_dataframe_reconciliation: {str(e)}")
        traceback.print_exc()  # Add stack trace for better debugging
        return {
            'success': False,
            'error': str(e)
        }

def handle_duplicate_matching(result, settings, fuzzy_ranking):
    """Handle duplicate matching logic for one-to-one reconciliation."""
    basis_table = settings.get('basis_table')
    duplicate_handling = settings.get('duplicate')
    basis_column = settings.get('basis_column', {})
    
    level_col = 'level_0' if basis_table == 'left' else 'level_1'
    basis_col_name = basis_column.get('left') if basis_table == 'left' else basis_column.get('right')
    
    if duplicate_handling == 'closest':
        result['Difference'] = calculate_difference(result, settings)
        result.sort_values(by=[level_col] + fuzzy_ranking + ['Difference'], 
                         ascending=[True, False, True], inplace=True)
    
    elif duplicate_handling == 'first_occurance':
        sort_col = f"{basis_col_name}-2" if basis_table == 'left' else f"{basis_col_name}-1"
        result.sort_values(by=[level_col] + fuzzy_ranking + [sort_col], 
                         ascending=[True, False, True], inplace=True)
    
    elif duplicate_handling in ['immedidately_before', 'immedidately_after']:
        result['Difference'] = calculate_difference(result, settings)
        if duplicate_handling == 'immedidately_before':
            result = result[result['Difference'] >= 0]
        else:
            result = result[result['Difference'] <= 0]
        result.sort_values(by=[level_col] + fuzzy_ranking + ['Difference'], 
                         ascending=[True, False, True], inplace=True)
    
    result.reset_index(drop=True, inplace=True)

def initialize_reconciliation_columns(df_left, df_right, cross_reference, values):
    """Initialize columns for reconciliation results."""
    # Initialize cross-reference columns
    for ref in cross_reference.get('right', []):
        if ref in cross_reference.get('left', []):
            df_left[ref+'-2'] = np.nan
        else:
            df_left[ref] = np.nan
            
    for ref in cross_reference.get('left', []):
        if ref in cross_reference.get('right', []):
            df_right[ref+'-1'] = np.nan
        else:
            df_right[ref] = np.nan

    # Initialize value comparison columns
    for value in values:
        if value['left'] == value['right']:
            df_left[value['right']+'-2'] = np.nan
            df_right[value['left']+'-1'] = np.nan
        else:
            df_left[value['right']] = np.nan
            df_right[value['left']] = np.nan

    # Initialize status columns
    df_left['Reco_Status'] = np.nan
    df_right['Reco_Status'] = np.nan

def process_matches(result, df_left, df_right, cross_reference, values):
    """Process matches between DataFrames."""
    if result is None or len(result) == 0:
        return
        
    while len(result) > 0:
        try:
            idx_left = result['level_0'].iloc[0]
            idx_right = result['level_1'].iloc[0]
            
            # Skip if indices are None
            if pd.isna(idx_left) or pd.isna(idx_right):
                result = result.iloc[1:]
                continue
            
            # Update cross-reference values
            update_cross_references(df_left, df_right, cross_reference, idx_left, idx_right)
            
            # Update status
            df_left.loc[idx_left, 'Reco_Status'] = 'Matched'
            df_right.loc[idx_right, 'Reco_Status'] = 'Matched'
            
            # Update values
            update_values(df_left, df_right, values, idx_left, idx_right)
            
            # Remove processed match
            result = result[
                (result['level_0'] != idx_left) & 
                (result['level_1'] != idx_right)
            ]
        except Exception as e:
            print(f"Error processing match: {str(e)}")
            # Skip problematic row and continue
            result = result.iloc[1:]
            continue

def calculate_value_differences(df_left, df_right, values):
    """Calculate differences and matches for value columns."""
    for value in values:
        calculate_single_value_difference(df_left, df_right, value)

def calculate_difference(result, settings):
    """Calculate difference between basis columns for duplicate matching.
    
    Args:
        result (DataFrame): Reconciliation result DataFrame
        settings (dict): Settings containing basis_table and basis_column info
    """
    basis_table = settings.get('basis_table')
    basis_column = settings.get('basis_column', {})
    
    if basis_table == 'left':
        col1 = basis_column.get('left')
        col2 = basis_column.get('right')
        if col1 == col2:
            return abs(result[f"{col1}-1"] - result[f"{col2}-2"])
        return abs(result[col1] - result[col2])
    else:
        col1 = basis_column.get('right')
        col2 = basis_column.get('left')
        if col1 == col2:
            return abs(result[f"{col1}-2"] - result[f"{col2}-1"])
        return abs(result[col1] - result[col2])

def update_cross_references(df_left, df_right, cross_reference, idx_left, idx_right):
    """Update cross-reference values between DataFrames.
    
    Args:
        df_left (DataFrame): Left DataFrame
        df_right (DataFrame): Right DataFrame
        cross_reference (dict): Cross-reference configuration
        idx_left (int): Index in left DataFrame
        idx_right (int): Index in right DataFrame
    """
    # Handle right references
    for ref in cross_reference.get('right', []):
        if ref in cross_reference.get('left', []):
            # Same column name in both DataFrames
            df_left.loc[idx_left, f"{ref}-2"] = df_right.loc[idx_right, ref]
        else:
            # Different column names
            df_left.loc[idx_left, ref] = df_right.loc[idx_right, ref]
    
    # Handle left references
    for ref in cross_reference.get('left', []):
        if ref in cross_reference.get('right', []):
            # Same column name in both DataFrames
            df_right.loc[idx_right, f"{ref}-1"] = df_left.loc[idx_left, ref]
        else:
            # Different column names
            df_right.loc[idx_right, ref] = df_left.loc[idx_left, ref]

def update_values(df_left, df_right, values, idx_left, idx_right):
    """Update value columns between DataFrames.
    
    Args:
        df_left (DataFrame): Left DataFrame
        df_right (DataFrame): Right DataFrame
        values (list): List of value column mappings
        idx_left (int): Index in left DataFrame
        idx_right (int): Index in right DataFrame
    """
    for value in values:
        if value['left'] == value['right']:
            # Same column name in both DataFrames
            df_left.loc[idx_left, f"{value['right']}-2"] = df_right.loc[idx_right, value['right']]
            df_right.loc[idx_right, f"{value['left']}-1"] = df_left.loc[idx_left, value['left']]
        else:
            # Different column names
            df_left.loc[idx_left, value['right']] = df_right.loc[idx_right, value['right']]
            df_right.loc[idx_right, value['left']] = df_left.loc[idx_left, value['left']]

def calculate_single_value_difference(df_left, df_right, value):
    """Calculate differences and matches for a single value column pair.
    
    Args:
        df_left (DataFrame): Left DataFrame
        df_right (DataFrame): Right DataFrame
        value (dict): Value column configuration with threshold settings
    """
    # Handle same column names
    if value['left'] == value['right']:
        col_name = value['left']
        df_left[f"{col_name}-Difference"] = abs(
            df_left[col_name] - df_left[f"{value['right']}-2"]
        )
        df_right[f"{col_name}-Difference"] = abs(
            df_right[col_name] - df_right[f"{value['left']}-1"]
        )
        
        # Apply threshold checks
        if value.get('threshold_type') == 'percent':
            # Calculate percent difference threshold
            left_threshold = abs(
                (df_left[col_name] + df_left[f"{value['right']}-2"]) / 2 * 
                value['threshold_value'] / 100
            )
            right_threshold = abs(
                (df_right[f"{value['left']}-1"] + df_right[col_name]) / 2 * 
                value['threshold_value'] / 100
            )
            
            df_left[f"{col_name}-Match"] = np.where(
                df_left[f"{col_name}-Difference"] <= left_threshold,
                'Matched', 'Unmatched'
            )
            df_right[f"{col_name}-Match"] = np.where(
                df_right[f"{col_name}-Difference"] <= right_threshold,
                'Matched', 'Unmatched'
            )
            
        elif value.get('threshold_type') == 'amount':
            # Use fixed amount threshold
            threshold = abs(value['threshold_value'])
            df_left[f"{col_name}-Match"] = np.where(
                df_left[f"{col_name}-Difference"] <= threshold,
                'Matched', 'Unmatched'
            )
            df_right[f"{col_name}-Match"] = np.where(
                df_right[f"{col_name}-Difference"] <= threshold,
                'Matched', 'Unmatched'
            )
            
        else:
            # Exact match required
            df_left[f"{col_name}-Match"] = np.where(
                df_left[f"{col_name}-Difference"] == 0,
                'Matched', 'Unmatched'
            )
            df_right[f"{col_name}-Match"] = np.where(
                df_right[f"{col_name}-Difference"] == 0,
                'Matched', 'Unmatched'
            )
    
    # Handle different column names
    else:
        df_left[f"{value['left']}-Difference"] = abs(
            df_left[value['left']] - df_left[value['right']]
        )
        df_right[f"{value['right']}-Difference"] = abs(
            df_right[value['right']] - df_right[value['left']]
        )
        
        # Apply threshold checks
        if value.get('threshold_type') == 'percent':
            # Calculate percent difference threshold
            left_threshold = abs(
                (df_left[value['left']] + df_left[value['right']]) / 2 * 
                value['threshold_value'] / 100
            )
            right_threshold = abs(
                (df_right[value['left']] + df_right[value['right']]) / 2 * 
                value['threshold_value'] / 100
            )
            
            df_left[f"{value['left']}-Match"] = np.where(
                df_left[f"{value['left']}-Difference"] <= left_threshold,
                'Matched', 'Unmatched'
            )
            df_right[f"{value['right']}-Match"] = np.where(
                df_right[f"{value['right']}-Difference"] <= right_threshold,
                'Matched', 'Unmatched'
            )
            
        elif value.get('threshold_type') == 'amount':
            # Use fixed amount threshold
            threshold = abs(value['threshold_value'])
            df_left[f"{value['left']}-Match"] = np.where(
                df_left[f"{value['left']}-Difference"] <= threshold,
                'Matched', 'Unmatched'
            )
            df_right[f"{value['right']}-Match"] = np.where(
                df_right[f"{value['right']}-Difference"] <= threshold,
                'Matched', 'Unmatched'
            )
            
        else:
            # Exact match required
            df_left[f"{value['left']}-Match"] = np.where(
                df_left[f"{value['left']}-Difference"] == 0,
                'Matched', 'Unmatched'
            )
            df_right[f"{value['right']}-Match"] = np.where(
                df_right[f"{value['right']}-Difference"] == 0,
                'Matched', 'Unmatched'
            )

def validate_cross_references(cross_reference, method, keys, values):
    """Validate cross reference selections based on reconciliation method."""
    if not cross_reference:
        return True
        
    for ref in cross_reference:
        # Validate 'many' side references are from keys only
        if method == 'one-to-many':
            if ref['file2'] not in [k['right'] for k in keys] and ref['file2'] != '__Custom__':
                return False, "For one-to-many, file2 cross references must be from key columns only"
        elif method == 'many-to-one':
            if ref['file1'] not in [k['left'] for k in keys] and ref['file1'] != '__Custom__':
                return False, "For many-to-one, file1 cross references must be from key columns only"
                
        # Validate references don't include value columns
        value_cols = [v['left'] for v in values] + [v['right'] for v in values]
        if ref['file1'] in value_cols or ref['file2'] in value_cols:
            return False, "Cross references cannot include value columns"
            
    return True, ""

def validate_fuzzy_threshold(key):
    """Validate fuzzy matching threshold."""
    if key.get('criteria') == 'fuzzy':
        threshold = key.get('threshold', 0.85)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return False, "Fuzzy matching threshold must be between 0 and 1"
    return True, ""

def get_detailed_error(e, context):
    """Get detailed error message with context."""
    return {
        "error": str(e),
        "context": context,
        "suggestion": get_error_suggestion(str(e))
    }

def get_error_suggestion(error):
    """Get suggestion based on error type."""
    suggestions = {
        "column not found": "Please verify column names in both tables",
        "invalid data type": "Ensure columns have compatible data types",
        "duplicate matches": "Check duplicate handling settings",
        # Add more suggestions
    }
    return next((v for k, v in suggestions.items() if k in error.lower()), "Please check input parameters")
