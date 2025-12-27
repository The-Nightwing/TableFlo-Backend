from flask import Blueprint, request, jsonify
from firebase_config import get_storage_bucket
import pandas as pd
import json
import traceback
from datetime import datetime
from models import db
from models import (
    User, 
    UserProcess, 
    DataFrame, 
    DataFrameOperation, 
    OperationType,
)
from io import BytesIO
import io

group_pivot_bp = Blueprint('group_pivot', __name__, url_prefix='/api/group-pivot')

def create_pivot_table(df, row_index, column_index, pivot_values):
    """Create a clean pivot table from DataFrame."""

    # Build aggregation dictionary
    aggfunc = {}

    for item in pivot_values:
        col = item['column']
        user_agg = item['aggregation']

        # Fix invalid aggregation: SUM on strings
        if df[col].dtype == 'object' and user_agg == 'sum':
            aggfunc[col] = 'first'
        else:
            aggfunc[col] = user_agg

    # Ensure column_index is a list or None
    if column_index in [None, "None", ""]:
        columns_arg = None
    elif isinstance(column_index, str):
        columns_arg = [column_index]
    else:
        columns_arg = list(column_index)

    # Build pivot table
    pivot_table = pd.pivot_table(
        df,
        index=row_index,
        columns=columns_arg,
        values=[item['column'] for item in pivot_values],
        aggfunc=aggfunc
    )
    # ----- CLEAN & REORDER COLUMN HEADERS -----

    # Build a list of original pivot columns (tuples for MultiIndex, single for Index)
    original_cols = list(pivot_table.columns)

    # Map original columns to the "clean" name used previously
    mapped_names = []
    for col in original_cols:
        if isinstance(col, tuple):
            # For multi-column, col is a tuple of (value_col, col_idx1, col_idx2, ...)
            val_col = col[0]
            col_idxs = col[1:]
        else:
            val_col = col
            col_idxs = ()

        if not col_idxs or all(idx in [None, "None", ""] for idx in col_idxs):
            # No column index: just use the value column name
            mapped_names.append(str(val_col))
        else:
            # Has column index: combine value column name with column index values
            # Format: ValueCol|idx1|idx2...
            idx_str = "|".join(str(idx) for idx in col_idxs if idx not in [None, "None", ""])
            mapped_names.append(f"{val_col}|{idx_str}")

    # Determine row index column names (after reset_index these will be the first columns)
    if isinstance(row_index, (list, tuple)):
        row_index_cols = list(row_index)
    elif row_index is None or row_index == []:
        row_index_cols = []
    else:
        row_index_cols = [row_index]

    # Build ordered list of mapped column names following the order of pivot_values
    ordered_value_cols = []
    # Also track the original column positions to reorder the DataFrame
    ordered_original_cols = []

    # For matching, normalize original_cols into tuples (val_col, *col_idxs)
    # Note: pandas pivot_table MultiIndex is (value_col, col_idx_value1, col_idx_value2, ...)
    # The aggregation is NOT part of the column index
    normalized_original = []
    for col in original_cols:
        if isinstance(col, tuple):
            # MultiIndex: (value_col, ...column_index_values)
            val_col = col[0]
            col_idxs = col[1:] if len(col) > 1 else ()
            normalized_original.append((val_col, col_idxs))
        else:
            normalized_original.append((col, ()))

    # For each pivot_value entry, collect ALL matching columns (all column index values for that value column)
    # then append them in the order they appear in the pivot table
    for pv in pivot_values:
        pv_col = pv.get('column')
        # Collect all mapped names that match this value column, preserving their original order
        for i, ((orig_val_col, orig_col_idxs), mapped_name) in enumerate(zip(normalized_original, mapped_names)):
            if str(orig_val_col) == str(pv_col):
                ordered_value_cols.append(mapped_name)
                ordered_original_cols.append(original_cols[i])

    # Construct final column list: row index columns first, then ordered value columns.
    final_columns = list(row_index_cols) + ordered_value_cols

    # Reset index to turn index levels into columns
    pivot_table = pivot_table.reset_index()

    # Reorder the DataFrame columns to match the desired order
    # First, create a mapping of current columns after reset_index
    current_cols_after_reset = list(pivot_table.columns)
    
    # Build the actual column order: row index cols (already in position) + reordered value cols
    reorder_cols = current_cols_after_reset[:len(row_index_cols)] + ordered_original_cols
    
    # Reorder the DataFrame
    if len(reorder_cols) == pivot_table.shape[1]:
        pivot_table = pivot_table[reorder_cols]
        # Now assign the clean names
        pivot_table.columns = final_columns
    else:
        # Fallback if something went wrong
        if len(final_columns) == pivot_table.shape[1]:
            pivot_table.columns = final_columns

    return pivot_table



def create_config_data(row_index, column_index, pivot_values):
    """Create configuration data for pivot table"""
    return {
        'rowIndex': row_index,
        'columnIndex': column_index,
        'pivotValues': pivot_values,
        'createdAt': datetime.now().isoformat()
    }

def save_process_pivot_data(email, process_id, pivot_table_name, pivot_df, config_data):
    """Save pivot table data and configuration to process storage"""
    try:
        bucket = get_storage_bucket()

        # Save pivot table data as CSV
        df_buffer = BytesIO()
        pivot_df.to_csv(df_buffer, index=False)
        df_buffer.seek(0)
        
        data_path = f"{email}/process/{process_id}/dataframes/{pivot_table_name}.csv"
        data_blob = bucket.blob(data_path)
        data_blob.upload_from_file(df_buffer, content_type='text/csv')

        # Save metadata
        # Build serializable column names and safe column type mapping.
        # Use positional access (iloc) to avoid cases where pivot_df[col]
        # returns a DataFrame (happens when column labels are duplicated),
        # which would not have a single `dtype` attribute.

        # Ensure all column names and keys are strings (handles MultiIndex and tuples)
        def stringify_col(col):
            if isinstance(col, tuple):
                return '|'.join([str(x) for x in col])
            return str(col)

        columns_list = [stringify_col(c) for c in pivot_df.columns.tolist()]
        column_types = {}
        for idx, col in enumerate(pivot_df.columns.tolist()):
            col_name = stringify_col(col)
            try:
                col_dtype = pivot_df.iloc[:, idx].dtype
            except Exception:
                col_dtype = None
            column_types[col_name] = str(col_dtype) if col_dtype is not None else 'unknown'

        metadata = {
            "type": "pivot_table",
            "columns": columns_list,
            "columnTypes": column_types,
            "rowCount": len(pivot_df),
            "configuration": config_data
        }
        
        metadata_path = f"{email}/process/{process_id}/metadata/{pivot_table_name}.json"
        metadata_blob = bucket.blob(metadata_path)
        metadata_blob.upload_from_string(
            json.dumps(metadata),
            content_type='application/json'
        )

        return {
            "success": True
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to save pivot table data: {str(e)}"
        }

def process_pivot_table(
    email,
    process_id,
    source_table_name,
    row_index,
    column_index,
    pivot_values,
    output_table_name
):
    """
    Process pivot table generation
    """
    try:
        bucket = get_storage_bucket()
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        # Source table path
        source_path = f"{email}/process/{process_id}/dataframes/{source_table_name}.csv"
        source_blob = bucket.blob(source_path)
        
        if not source_blob.exists():
            return {
                'success': False,
                'error': f"Source table {source_table_name} not found"
            }

        # Get source table data
        source_data = get_process_table_data(
            email=email,
            process_id=process_id,
            table_name=source_table_name,
            page=None,
            per_page=None
        )
        
        if 'error' in source_data:
            return {
                'success': False,
                'error': f"Error loading source table: {source_data['error']}"
            }

        # Convert to DataFrame
        df = pd.DataFrame(source_data['data'])

        # Validate that math functions are not applied to string columns
        math_aggs = {'sum', 'mean', 'median', 'min', 'max', 'std', 'var'}
        sample_bytes = source_blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(sample_bytes), nrows=10)
        for pv in pivot_values:
            col = pv.get('column')
            agg = pv.get('aggregation')
            if agg in math_aggs and col in df.columns:
                if str(df[col].dtype) == 'object':
                    return {
                        "success": False,
                        "error": f"The function '{agg}' cannot be applied to text field '{col}'. This function cannot be applied to a text field."
                    }


        # Create pivot table
        pivot_df = create_pivot_table(df, row_index, column_index, pivot_values)
        config_data = create_config_data(row_index, column_index, pivot_values)

        # Generate output table name if not provided
        if not output_table_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_table_name = f"pivot_{source_table_name}_{timestamp}"

        # Save output
        storage_path = f"{email}/process/{process_id}/dataframes/{output_table_name}.csv"
        
        # Save pivot table data
        result = save_process_pivot_data(
            email=email,
            process_id=process_id,
            pivot_table_name=output_table_name,
            pivot_df=pivot_df,
            config_data=config_data
        )

        if not result.get('success'):
            return result

        # Create DataFrame record
        try:
            dataframe_record = DataFrame.create_from_pandas(
                df=pivot_df,
                process_id=process_id,
                name=output_table_name,
                email=email,
                storage_path=storage_path,
                user_id=user.id 
            )
            db.session.add(dataframe_record)
            db.session.commit()

            return {
                "success": True,
                "message": "Pivot table generated successfully.",
                "id": dataframe_record.id,
                "name": output_table_name,
                "rowCount": len(pivot_df),
                "columnCount": len(pivot_df.columns)
            }

        except Exception as e:
            db.session.rollback()
            # Clean up uploaded files if they exist
            blob = bucket.blob(storage_path)
            if blob.exists():
                blob.delete()
            metadata_blob = bucket.blob(f"{email}/process/{process_id}/metadata/{output_table_name}.json")
            if metadata_blob.exists():
                metadata_blob.delete()
            raise e

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_process_table_data(email, process_id, table_name, page=None, per_page=None):
    """
    Retrieves data from a process table, with optional pagination
    
    Args:
        email (str): User's email
        process_name (str): Name of the process
        table_name (str): Name of the table
        page (int, optional): Page number for pagination (1-based)
        per_page (int, optional): Number of items per page
    
    Returns:
        dict: Table data and metadata
            If paginated:
            {
                'data': list of records,
                'totalRows': total number of rows,
                'currentPage': current page number,
                'totalPages': total number of pages,
                'perPage': items per page
            }
            If not paginated:
            {
                'data': list of records
            }
            If error:
            {
                'error': error message
            }
    """
    try:
        bucket = get_storage_bucket()
        
        # Get table data
        data_path = f"{email}/process/{process_id}/dataframes/{table_name}.csv"
        data_blob = bucket.blob(data_path)
        
        if not data_blob.exists():
            return {
                'error': f"Table {table_name} not found in process {process_id}"
            }
        
        try:
            # Load CSV data directly into DataFrame
            content = data_blob.download_as_string()
            df = pd.read_csv(BytesIO(content))
            
            # Convert DataFrame to records if needed
            all_data = df.to_dict('records')
            
            # Get metadata for column types
            metadata_path = f"{email}/process/{process_id}/metadata/{table_name}.json"
            metadata_blob = bucket.blob(metadata_path)
            
            if metadata_blob.exists():
                metadata_content = metadata_blob.download_as_text()
                metadata = json.loads(metadata_content)
                
                # Handle different metadata structures
                column_types = {}
                if 'columnTypes' in metadata:
                    # Old format: columnTypes dictionary
                    column_types = metadata.get('columnTypes', {})
                elif 'columns' in metadata:
                    # New format: columns array with name and type
                    for col in metadata.get('columns', []):
                        if 'name' in col and 'type' in col:
                            column_types[col['name']] = col['type']
                
                # Convert data types based on metadata
                for record in all_data:
                    for column, dtype in column_types.items():
                        if column in record:
                            try:
                                if dtype == 'date':
                                    record[column] = pd.to_datetime(record[column])
                                elif dtype in ['integer', 'int64']:
                                    record[column] = pd.to_numeric(record[column], errors='coerce')
                                elif dtype in ['float', 'float64']:
                                    record[column] = pd.to_numeric(record[column], errors='coerce')
                                elif dtype == 'boolean':
                                    record[column] = bool(record[column])
                            except Exception as e:
                                print(f"Error converting column {column} to type {dtype}: {str(e)}")
            
            # Clean the data - preserve numeric types, only convert NaN to empty strings for non-numeric columns
            if isinstance(all_data, list):
                cleaned_data = []
                for record in all_data:
                    cleaned_record = {}
                    for k, v in record.items():
                        if pd.isna(v):
                            # Check if this column should be numeric based on metadata
                            if k in column_types and column_types[k] in ['integer', 'float', 'int64', 'float64']:
                                cleaned_record[k] = 0  # Use 0 for numeric NaN values
                            elif k in column_types and column_types[k] == 'date':
                                cleaned_record[k] = '1900-01-01'  # Use default date for empty date values
                            else:
                                cleaned_record[k] = ''  # Use empty string for non-numeric NaN values
                        else:
                            cleaned_record[k] = v
                    cleaned_data.append(cleaned_record)
                all_data = cleaned_data
            
            # If pagination is not requested, return all data
            if page is None or per_page is None:
                return {
                    'data': all_data,
                    'totalRows': len(all_data)
                }
            
            # Handle pagination
            total_rows = len(all_data)
            total_pages = (total_rows + per_page - 1) // per_page
            
            # Validate page number
            if page < 1:
                page = 1
            elif page > total_pages:
                page = total_pages if total_pages > 0 else 1
            
            # Calculate slice indices
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_rows)
            
            # Return paginated data with metadata
            return {
                'data': all_data[start_idx:end_idx],
                'totalRows': total_rows,
                'currentPage': page,
                'totalPages': total_pages,
                'perPage': per_page
            }
            
        except json.JSONDecodeError as e:
            return {
                'error': f"Failed to parse table data: {str(e)}"
            }
        except Exception as e:
            return {
                'error': f"Failed to read table data: {str(e)}"
            }
            
    except Exception as e:
        return {
            'error': f"Unexpected error: {str(e)}"
        }

@group_pivot_bp.route('/generate/', methods=['POST'])
def generate_pivot():
    """Endpoint for generating pivot tables within a process"""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        process_id = data.get('processId')
        source_table_name = data.get('tableName')
        row_index = data.get('rowIndex', [])
        column_index = data.get('columnIndex', None)
        # Accept both string and list, always convert to list if not None
        if column_index not in [None, "None", ""] and not isinstance(column_index, list):
            column_index = [column_index]
        pivot_values = data.get('pivotValues', [])
        output_table_name = data.get('outputTableName', '').strip()

        if not all([process_id, source_table_name, row_index, pivot_values]):
            return jsonify({
                "error": "Missing required parameters: processId, tableName, rowIndex, or pivotValues."
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
            name=source_table_name
        ).first()
        if not source_df:
            return jsonify({"error": f"Table '{source_table_name}' not found in process"}), 404
        
        existing_df = DataFrame.query.filter_by(
            process_id=process_id,
            name=output_table_name
        ).first()

        if existing_df:
            if existing_df.is_temporary == False:
                return jsonify({"error": f"Table with name {output_table_name} already exists."}), 409

        # Generate message for the pivot operation
        message = f"Create a pivot table from table {source_table_name} using specified column headers and rows, columns and values"

        # Create DataFrameOperation record with IN_PROGRESS status
        df_operation = DataFrameOperation(
            process_id=process_id,
            dataframe_id=source_df.id,
            operation_type=OperationType.GROUP_PIVOT.value,
            payload=data,
            message=message
        )
        df_operation.user_id = user.id
        
        # Save initial operation record
        db.session.add(df_operation)
        db.session.commit()

        try:
            result = process_pivot_table(
                email=email,
                process_id=process_id,
                source_table_name=source_table_name,
                row_index=row_index,
                column_index=column_index,
                pivot_values=pivot_values,
                output_table_name=output_table_name
            )

            if not result.get('success'):
                df_operation.set_error(result.get('error'))
                db.session.commit()
                return jsonify({"error": result.get('error')}), 409

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
