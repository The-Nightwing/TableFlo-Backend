from flask import Blueprint, request, jsonify
import pandas as pd
from io import BytesIO
from firebase_config import get_storage_bucket
import os
import json
from datetime import datetime, timezone
from models import db
from models import (
    User, 
    UserProcess, 
    DataFrame, 
    DataFrameOperation, 
    OperationType,
)
import traceback

sort_filter_bp = Blueprint('sort_filter', __name__, url_prefix='/api/sort-filter/')

def load_preview_file(bucket, email, file_name, sheet_name=None):
    """Load preview file from Firebase."""
    preview_path = f"{email}/previews/{file_name}/{sheet_name or 'Sheet1'}_preview.csv"
    blob = bucket.blob(preview_path)
    if not blob.exists():
        raise FileNotFoundError(f"Preview file not found at: {preview_path}")

    content = blob.download_as_text()
    df = pd.read_csv(BytesIO(content.encode('utf-8')), na_values=['NA', ''])
    
    # Clean data based on column types - preserve data types
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)  # Use 0 for numeric NaN values
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(pd.Timestamp('1900-01-01'))  # Use default date for date NaN values
        else:
            df[col] = df[col].fillna('')  # Use empty string for string NaN values
    
    return df


@sort_filter_bp.route('/preview/sort/', methods=['POST'])
def preview_sort_data():
    """
    Preview sorted data from the preview folder.
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        file_name = data.get('fileName')
        sheet_name = data.get('sheet', None)
        sort_config = data.get('sortConfig', [])

        if not file_name:
            return jsonify({"error": "Missing required inputs: fileName."}), 400

        # Load file from Firebase
        bucket = get_storage_bucket()
        df = load_preview_file(bucket, email, file_name, sheet_name)

        # Validate and apply sorting
        sort_by = [config['column'] for config in sort_config]
        ascending = [config['order'] == 'asc' for config in sort_config]
        df = df.sort_values(by=sort_by, ascending=ascending)

        # Return preview (first 25 rows)
        preview_data = {
            "columns": list(df.columns),
            "rows": df.head(25).to_dict(orient="records")
        }
        return jsonify({"success": True, "data": preview_data})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@sort_filter_bp.route('/preview/filter/', methods=['POST'])
def preview_filter_data():
    """
    Preview filtered data from the preview folder.
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        file_name = data.get('fileName')
        sheet_name = data.get('sheet', None)
        filter_config = data.get('filterConfig', [])

        if not file_name:
            return jsonify({"error": "Missing required inputs: fileName."}), 400

        # Load file from Firebase
        bucket = get_storage_bucket()
        df = load_preview_file(bucket, email, file_name, sheet_name)

        # Apply filtering
        for config in filter_config:
            column = config['column']
            criteria = config['criteria']
            value = config['value']

            if criteria == "equals":
                if pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        df = df[df[column] == float(value)]
                    except (ValueError, TypeError):
                        df = df[df[column] == value]
                else:
                    df = df[df[column] == value]
            elif criteria == "does not equal":
                if pd.api.types.is_numeric_dtype(df[column]):
                    try:
                        df = df[df[column] != float(value)]
                    except (ValueError, TypeError):
                        df = df[df[column] != value]
                else:
                    df = df[df[column] != value]
            elif criteria == "greater than":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] > filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] > float(value)]
                else:
                    df = df[df[column] > float(value)]
            elif criteria == "greater than or equal to":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] >= filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] >= float(value)]
                else:
                    df = df[df[column] >= float(value)]
            elif criteria == "less than":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] < filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] < float(value)]
                else:
                    df = df[df[column] < float(value)]
            elif criteria == "less than or equal to":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] <= filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] <= float(value)]
                else:
                    df = df[df[column] <= float(value)]
            elif criteria == "begins with":
                df = df[df[column].str.startswith(value, na=False)]
            elif criteria == "does not begin with":
                df = ~df[column].str.startswith(value, na=False)
            elif criteria == "ends with":
                df = df[df[column].str.endswith(value, na=False)]
            elif criteria == "does not end with":
                df = ~df[column].str.endswith(value, na=False)
            elif criteria == "contains":
                df = df[df[column].str.contains(value, na=False)]
            elif criteria == "does not contain":
                df = df[~df[column].str.contains(value, na=False)]
            elif criteria == "is null":
                df = df[df[column].isna()]
            elif criteria == "is not null":
                df = df[~df[column].isna()]
            else:
                raise ValueError(f"Unsupported filter criteria: {criteria}")

        # Return preview (first 25 rows)
        preview_data = {
            "columns": list(df.columns),
            "rows": df.head(25).to_dict(orient="records")
        }
        return jsonify({"success": True, "data": preview_data})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@sort_filter_bp.route('/final/sort/', methods=['POST'])
def final_save_sort_data():
    """
    Save sorted data to the processed_files folder.
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        file_name = data.get('fileName')
        sort_config = data.get('sortConfig', [])
        sheet_name = data.get('sheet', None)

        if not file_name:
            return jsonify({"error": "Missing required inputs: fileName."}), 400

        # Load file from Firebase
        bucket = get_storage_bucket()
        blob = bucket.blob(f"{email}/uploaded_files/{file_name}")
        if not blob.exists():
            raise FileNotFoundError(f"File '{file_name}' not found.")
        file_content = blob.download_as_bytes()
        df = pd.read_excel(BytesIO(file_content), sheet_name=sheet_name) if sheet_name else pd.read_csv(BytesIO(file_content))

        # Apply sorting
        sort_by = [config['column'] for config in sort_config]
        ascending = [config['order'] == 'asc' for config in sort_config]
        df = df.sort_values(by=sort_by, ascending=ascending)

        # Save to Firebase
        output_buffer = BytesIO()
        df.to_excel(output_buffer, index=False, sheet_name="Sorted_Data")
        output_buffer.seek(0)
        output_file_name = f"sorted_{file_name}"
        processed_blob = bucket.blob(f"{email}/processed_files/{output_file_name}")
        processed_blob.upload_from_file(output_buffer, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Generate signed URL
        download_url = processed_blob.generate_signed_url(expiration=3600)
        return jsonify({"success": True, "downloadUrl": download_url})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@sort_filter_bp.route('/final/filter/', methods=['POST'])
def final_save_filter_data():
    """
    Save filtered data to the processed_files folder.
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        file_name = data.get('fileName')
        sheet_name = data.get('sheet', None)
        filter_config = data.get('filterConfig', [])

        if not file_name:
            return jsonify({"error": "Missing required inputs: fileName."}), 400

        # Load file from Firebase
        bucket = get_storage_bucket()
        blob = bucket.blob(f"{email}/uploaded_files/{file_name}")
        if not blob.exists():
            raise FileNotFoundError(f"File '{file_name}' not found.")
        file_content = blob.download_as_bytes()
        df = pd.read_excel(BytesIO(file_content), sheet_name=sheet_name) if sheet_name else pd.read_csv(BytesIO(file_content))

         # Apply filtering (if any)
        for config in filter_config:
            column = config['column']
            criteria = config['criteria']
            value = config['value']

            if criteria == "equals":
                df = df[df[column] == value]
            elif criteria == "does not equal":
                df = df[df[column] != value]
            elif criteria == "greater than":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] > filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] > float(value)]
                else:
                    df = df[df[column] > float(value)]
            elif criteria == "greater than or equal to":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] >= filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] >= float(value)]
                else:
                    df = df[df[column] >= float(value)]
            elif criteria == "less than":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] < filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] < float(value)]
                else:
                    df = df[df[column] < float(value)]
            elif criteria == "less than or equal to":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] <= filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] <= float(value)]
                else:
                    df = df[df[column] <= float(value)]
            elif criteria == "begins with":
                df = df[df[column].str.startswith(value, na=False)]
            elif criteria == "does not begin with":
                df = ~df[column].str.startswith(value, na=False)
            elif criteria == "ends with":
                df = df[df[column].str.endswith(value, na=False)]
            elif criteria == "does not end with":
                df = ~df[column].str.endswith(value, na=False)
            elif criteria == "contains":
                df = df[df[column].str.contains(value, na=False)]
            elif criteria == "does not contain":
                df = df[~df[column].str.contains(value, na=False)]
            elif criteria == "is null":
                df = df[df[column].isna()]
            elif criteria == "is not null":
                df = df[~df[column].isna()]
            else:
                raise ValueError(f"Unsupported filter criteria: {criteria}")


        # Save to Firebase
        output_buffer = BytesIO()
        df.to_excel(output_buffer, index=False, sheet_name="Filtered_Data")
        output_buffer.seek(0)
        output_file_name = f"filtered_{file_name}"
        processed_blob = bucket.blob(f"{email}/processed_files/{output_file_name}")
        processed_blob.upload_from_file(output_buffer, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Generate signed URL
        download_url = processed_blob.generate_signed_url(expiration=3600)
        return jsonify({"success": True, "downloadUrl": download_url})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    
    
@sort_filter_bp.route('/preview/combined/', methods=['POST'])
def preview_combined_data():
    """
    Preview data with both sorting and filtering applied, from the preview folder.
    """
    try:
        # Extract request data
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        file_name = data.get('fileName')
        sheet_name = data.get('sheet', None)
        sort_config = data.get('sortConfig', [])
        filter_config = data.get('filterConfig', [])

        if not file_name:
            return jsonify({"error": "Missing required input: fileName."}), 400

        # Initialize Firebase storage bucket
        bucket = get_storage_bucket()

        # Load preview file
        def load_preview_file(bucket, email, file_name, sheet_name=None):
            """
            Load a preview file from Firebase Storage and return it as a pandas DataFrame.
            """
            file_path = f"{email}/previews/{file_name}"
            if sheet_name and sheet_name != "CSV":
                file_path += f"/{sheet_name}_preview.csv"
            else:
                file_path += "/preview.csv"


            blob = bucket.blob(file_path)
            if not blob.exists():
                raise FileNotFoundError(f"Preview file not found: {file_path}")

            content = blob.download_as_text()
            return pd.read_csv(BytesIO(content.encode('utf-8'))).fillna("-")

        # Load the preview data
        df = load_preview_file(bucket, email, file_name, sheet_name)

         # Apply filtering (if any)
        for config in filter_config:
            column = config['column']
            criteria = config['criteria']
            value = config['value']

            if criteria == "equals":
                df = df[df[column] == value]
            elif criteria == "does not equal":
                df = df[df[column] != value]
            elif criteria == "greater than":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] > filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] > float(value)]
                else:
                    df = df[df[column] > float(value)]
            elif criteria == "greater than or equal to":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] >= filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] >= float(value)]
                else:
                    df = df[df[column] >= float(value)]
            elif criteria == "less than":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] < filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] < float(value)]
                else:
                    df = df[df[column] < float(value)]
            elif criteria == "less than or equal to":
                # Handle date filtering
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        filter_date = pd.to_datetime(value)
                        df = df[df[column] <= filter_date]
                    except (ValueError, TypeError):
                        # If date parsing fails, try numeric comparison
                        df = df[df[column] <= float(value)]
                else:
                    df = df[df[column] <= float(value)]
            elif criteria == "begins with":
                df = df[df[column].str.startswith(value, na=False)]
            elif criteria == "does not begin with":
                df = ~df[column].str.startswith(value, na=False)
            elif criteria == "ends with":
                df = df[df[column].str.endswith(value, na=False)]
            elif criteria == "does not end with":
                df = ~df[column].str.endswith(value, na=False)
            elif criteria == "contains":
                df = df[df[column].str.contains(value, na=False)]
            elif criteria == "does not contain":
                df = df[~df[column].str.contains(value, na=False)]
            else:
                raise ValueError(f"Unsupported filter criteria: {criteria}")

        # Apply sorting if configurations are provided
        if sort_config:
            sort_by = [config['column'] for config in sort_config if 'column' in config]
            ascending = [config['order'] == 'asc' for config in sort_config if 'order' in config]
            df = df.sort_values(by=sort_by, ascending=ascending)

        # Prepare preview data (returning only the first 25 rows)
        preview_data = {
            "columns": list(df.columns),
            "rows": df.head(25).to_dict(orient="records")
        }

        return jsonify({"success": True, "data": preview_data})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def get_column_type(series):
    """Helper function to determine column type (standardizing on 'date')."""
    try:
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'string'
        
        # Booleans
        dtype_str = str(series.dtype)
        if dtype_str.startswith('bool'):
            return 'boolean'
        
        # Date detection before numeric
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
        except Exception:
            pass
        
        # Numeric types
        if dtype_str.startswith('int'):
            return 'integer'
        if dtype_str.startswith('float'):
            return 'float'
        
        # Fallbacks
        try:
            pd.to_datetime(non_null, errors='raise')
            return 'date'
        except Exception:
            return 'string'
    except Exception:
        return 'string'

def apply_filter(df, column, operator, value):
    """Apply filter operation to DataFrame."""
    print(type(df[column]))
    if operator == "equals":
        col = df[column]
        # Numeric
        if pd.api.types.is_numeric_dtype(col):
            try:
                return df[col == float(value)]
            except (ValueError, TypeError):
                return df[col == value]
        # Datetime
        elif (
            pd.api.types.is_datetime64_any_dtype(col)
            or (
                pd.api.types.is_object_dtype(col)
                and pd.to_datetime(col, errors='coerce', infer_datetime_format=True, dayfirst=True).notna().any()
            )
        ):
            try:
                col_dt = pd.to_datetime(col, errors='coerce', infer_datetime_format=True, dayfirst=True)
                value_dt = pd.to_datetime(value, errors='coerce', infer_datetime_format=True, dayfirst=True)

                if pd.isna(value_dt):
                    # fallback to string comparison
                    return df[df[column].astype(str).str.strip() == str(value).strip()]

                mask = col_dt.dt.normalize() == value_dt.normalize()
                return df[mask]
            except Exception:
                return df[df[column].astype(str).str.strip() == str(value).strip()]

        # Boolean
        elif pd.api.types.is_bool_dtype(col):
            if isinstance(value, str):
                val = value.lower() in ["true", "1", "yes"]
            else:
                val = bool(value)
            return df[col == val]
        # String or object
        else:
            # Case-insensitive comparison
            return df[col.astype(str).str.lower() == str(value).lower()]
    elif operator == "not_equals":
        col = df[column]

        # ---------- BOOLEAN COLUMN HANDLING ----------
        if pd.api.types.is_bool_dtype(col):
            # Convert user input to boolean safely
            if isinstance(value, str):
                val = value.lower().strip() in ["true", "1", "yes"]
            else:
                val = bool(value)
            return df[col != val]

        # ---------- TRY DATETIME COLUMN LOGIC ----------
        dt_col = pd.to_datetime(col, errors='coerce')

        # If entire column is NOT datetime → skip date handling
        if not dt_col.isna().all():

            # Helper to parse user-provided dates (DD-MM-YYYY supported)
            def parse_user_date(val):
                # Try pandas first with dayfirst=True
                try:
                    return pd.to_datetime(val, dayfirst=True)
                except:
                    pass

                # Try explicit formats
                formats = [
                    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
                    "%Y-%m-%d", "%Y/%m/%d"
                ]
                for fmt in formats:
                    try:
                        return pd.to_datetime(datetime.strptime(val, fmt))
                    except:
                        pass
                return None

            filter_date = parse_user_date(value)

            # If parsed successfully → use datetime comparison
            if filter_date is not None:
                return df[dt_col != filter_date]

        # ---------- NUMERIC FALLBACK ----------
        try:
            return df[pd.to_numeric(col, errors='coerce') != float(value)]
        except:
            pass

        # ---------- STRING FALLBACK ----------
        # Case-insensitive comparison
        return df[col.astype(str).str.lower() != str(value).lower()]

    elif operator == "contains":
        return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
    elif operator == "not_contains":
        return df[~df[column].astype(str).str.contains(str(value), case=False, na=False)]
    elif operator == "greater_than":
        # Handle date filtering
        col = pd.to_datetime(df[column], errors='coerce')
        # If the entire column failed to convert, fallback to numeric
        if col.isna().all():
            return df[pd.to_numeric(df[column], errors='coerce') > float(value)]

        # Now handle date filtering
        try:
            filter_date = pd.to_datetime(value)
            return df[col > filter_date]
        except:
            return df[pd.to_numeric(df[column], errors='coerce') > float(value)]

    elif operator == "less_than":
        # Handle date filtering
        col = pd.to_datetime(df[column], errors='coerce')
        # If the entire column failed to convert, fallback to numeric
        if col.isna().all():
            return df[pd.to_numeric(df[column], errors='coerce') < float(value)]

        # Now handle date filtering
        try:
            filter_date = pd.to_datetime(value)
            return df[col < filter_date]
        except:
            return df[pd.to_numeric(df[column], errors='coerce') < float(value)]
    elif operator == "greater_equals":
        # Handle date filtering
        col = pd.to_datetime(df[column], errors='coerce')
        # If the entire column failed to convert, fallback to numeric
        if col.isna().all():
            return df[pd.to_numeric(df[column], errors='coerce') >= float(value)]

        # Now handle date filtering
        try:
            filter_date = pd.to_datetime(value)
            return df[col >= filter_date]
        except:
            return df[pd.to_numeric(df[column], errors='coerce') >= float(value)]
    elif operator == "less_equals":
        # Handle date filtering
        col = pd.to_datetime(df[column], errors='coerce')
        # If the entire column failed to convert, fallback to numeric
        if col.isna().all():
            return df[pd.to_numeric(df[column], errors='coerce') <= float(value)]

        # Now handle date filtering
        try:
            filter_date = pd.to_datetime(value)
            return df[col <= filter_date]
        except:
            return df[pd.to_numeric(df[column], errors='coerce') <= float(value)]
    elif operator == "starts_with":
        # Case-insensitive startswith
        return df[df[column].astype(str).str.lower().str.startswith(str(value).lower())]
    elif operator == "ends_with":
        # Case-insensitive endswith
        return df[df[column].astype(str).str.lower().str.endswith(str(value).lower())]
    elif operator == "is_null":
        return df[df[column].isna()]
    elif operator == "is_not_null":
        return df[~df[column].isna()]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def apply_sort(df, sort_config):
    """Apply sort operations to DataFrame"""
    if sort_config:
        sort_by = [config['column'] for config in sort_config]
        ascending = [config['order'] == 'asc' for config in sort_config]
        return df.sort_values(by=sort_by, ascending=ascending)
    return df

def save_processed_file(email, df, output_file_name, output_extension):
    """Save processed file and generate metadata and previews"""
    bucket = get_storage_bucket()
    output_buffer = BytesIO()
    
    # Save to Firebase based on format
    if output_extension in ['.xlsx', '.xls']:
        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Processed_Data', index=False)
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        sheet_name = "Processed_Data"
    else:  # CSV
        df.to_csv(output_buffer, index=False)
        content_type = "text/csv"
        sheet_name = "CSV"

    # Generate metadata
    metadata = {
        "sheets": {
            sheet_name: {
                "columns": df.columns.tolist(),
                "columnTypes": {col: get_column_type(df[col]) for col in df.columns}
            }
        }
    }

    # Save file
    output_buffer.seek(0)
    processed_blob = bucket.blob(f"{email}/processed_files/{output_file_name}")
    processed_blob.upload_from_file(output_buffer, content_type=content_type)

    # Save metadata
    metadata_blob_path = f"{email}/metadata/{output_file_name}.json"
    metadata_blob = bucket.blob(metadata_blob_path)
    metadata_blob.upload_from_string(json.dumps(metadata), content_type='application/json')

    # Save preview
    preview_buffer = BytesIO()
    df.head(50).to_csv(preview_buffer, index=False)
    preview_buffer.seek(0)
    preview_blob_path = f"{email}/previews/{output_file_name}/{sheet_name}_preview.csv"
    preview_blob = bucket.blob(preview_blob_path)
    preview_blob.upload_from_file(preview_buffer, content_type='text/csv')

    return processed_blob.generate_signed_url(expiration=3600, version='v4')

def process_sort_filter_data(email, process_id, source_df, sort_config, filter_config, output_table_name, existing_df=None):
    """Process sorting and filtering of DataFrame data."""
    try:
        bucket = get_storage_bucket()

        # Read source DataFrame
        content = bucket.blob(source_df.storage_path).download_as_string()
        df = pd.read_csv(BytesIO(content))

        # Apply filters
        if filter_config:
            for filter_rule in filter_config:
                column = filter_rule.get('column')
                operator = filter_rule.get('operator')
                value = filter_rule.get('value')
                
                if column not in df.columns:
                    return {"success": False, "error": f"Column '{column}' not found"}

                df = apply_filter(df, column, operator, value)

        # Apply sorting
        if sort_config:
            columns = [rule['column'] for rule in sort_config]
            ascending = [rule.get('direction', 'asc') == 'asc' for rule in sort_config]
            df = df.sort_values(by=columns, ascending=ascending)

        # Generate storage paths
        storage_path = f"{email}/process/{process_id}/dataframes/{output_table_name}.csv"
        metadata_path = f"{email}/process/{process_id}/metadata/{output_table_name}.json"

        # Save DataFrame and metadata
        try:
            # Save DataFrame as CSV
            df_buffer = BytesIO()
            df.to_csv(df_buffer, index=False)
            df_buffer.seek(0)

            # Upload to Firebase
            df_blob = bucket.blob(storage_path)
            df_blob.upload_from_file(df_buffer, content_type='text/csv')

            # Create metadata
            metadata = {
                "type": "processed_table",
                "description": f"Processed table from {source_df.name}",
                "processId": process_id,
                "createdAt": (existing_df.created_at if existing_df else datetime.now(timezone.utc)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                "updatedAt": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                "rowCount": len(df),
                "columnCount": len(df.columns),
                "columns": df.columns.tolist(),
                "columnTypes": {col: str(df[col].dtype) for col in df.columns},
                "operation": {
                    "type": "sort_filter",
                    "sortConfig": sort_config,
                    "filterConfig": filter_config,
                    "sourceTable": {
                        "id": source_df.id,
                        "name": source_df.name
                    }
                }
            }

            # Save metadata
            metadata_blob = bucket.blob(metadata_path)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )

            # Update or create DataFrame record
            if existing_df:
                existing_df.row_count = len(df)
                existing_df.column_count = len(df.columns)
                existing_df.updated_at = datetime.now(timezone.utc)
                dataframe_record = existing_df
            else:
                dataframe_record = DataFrame.create_from_pandas(
                    df=df,
                    process_id=process_id,
                    name=output_table_name,
                    email=email,
                    storage_path=storage_path,
                    user_id=source_df.user_id,
                    is_temporary=True
                )
                db.session.add(dataframe_record)

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
            if df_blob.exists():
                df_blob.delete()
            if metadata_blob.exists():
                metadata_blob.delete()
            raise e

    except Exception as e:
        raise Exception(f"Error processing table: {str(e)}")

@sort_filter_bp.route('/apply/', methods=['POST'])
def process_sort_filter():
    """Apply sorting and filtering to a DataFrame within a process."""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        process_id = data.get('processId')
        table_name = data.get('tableName')
        output_table_name = data.get('outputTableName', '').strip()
        sort_config = data.get('sortConfig', [])
        filter_config = data.get('filterConfig', [])

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

        # Generate message based on operation types
        message_parts = []
        if sort_config:
            sort_columns = [config['column'] for config in sort_config]
            message_parts.append(f"Sort table {table_name} on column(s) {', '.join(sort_columns)}")
        if filter_config:
            filter_columns = [config['column'] for config in filter_config]
            message_parts.append(f"Filter table {table_name} on column(s) {', '.join(filter_columns)}")
        
        message = " and ".join(message_parts)

        # Create DataFrameOperation record with IN_PROGRESS status
        df_operation = DataFrameOperation(
            process_id=process_id,
            dataframe_id=source_df.id,
            operation_type=OperationType.SORT_FILTER.value,
            payload=data,
            message=message
        )
        df_operation.user_id = user.id
        
        # Save initial operation record
        db.session.add(df_operation)
        db.session.commit()

        try:
            result = process_sort_filter_data(
                email=email,
                process_id=process_id,
                source_df=source_df,
                sort_config=sort_config,
                filter_config=filter_config,
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
