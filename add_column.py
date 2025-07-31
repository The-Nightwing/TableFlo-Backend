from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timezone
from firebase_config import get_storage_bucket
import json
from io import BytesIO
from models import User, UserProcess, DataFrame, db, DataFrameOperation, OperationType, AddColumnSubType

logger = logging.getLogger(__name__)
add_column_bp = Blueprint('add_column', __name__, url_prefix='/api/add-column')

def apply_calculation(df, operations):
    """
    Apply calculation operation with multiple steps
    operations: List of dicts with format:
    [
        {
            'column1': 'column_name',
            'operator': '+|-|*|/',
            'column2': 'column_name' or None,
            'fixed_value': float or None
        }
    ]
    """
    result = None
    for op in operations:
        value1 = df[op['column1']] if result is None else result
        
        # Handle value2 based on whether we have column2 or fixed_value
        if op.get('column2') is not None:
            value2 = df[op['column2']]
        elif 'fixed_value' in op:
            value2 = float(op['fixed_value'])
        else:
            raise ValueError("Either column2 or fixed_value must be provided")
        
        if op['operator'] == '+':
            result = value1 + value2
        elif op['operator'] == '-':
            result = value1 - value2
        elif op['operator'] == '*':
            result = value1 * value2
        elif op['operator'] == '/':
            result = value1 / value2
    return result

def apply_concatenation(df, operations):
    """
    Apply concatenation with substring operations
    operations: List of dicts with format:
    [
        {
            'column': 'column_name',
            'type': 'Left|Right|Full text',
            'chars': int or None
        }
    ]
    """
    parts = []
    for op in operations:
        text = df[op['column']].astype(str)
        if op['type'] == 'Left':
            parts.append(text.str[:op['chars']])
        elif op['type'] == 'Right':
            parts.append(text.str[-op['chars']:])
        else:  # Full text
            parts.append(text)
    
    return pd.Series([''.join(row) for row in zip(*parts)])

def apply_conditional(df, source_column, conditions, residual_value):
    """
    Apply nested conditional operations
    conditions: List of dicts with format:
    [
        {
            'operator': 'equals|does not equal|...',
            'reference_value': value,
            'conditional_value': value
        }
    ]
    """
    operator_map = {
        'equals': '==',
        'does not equal': '!=',
        'greater than': '>',
        'greater than or equal to': '>=',
        'less than': '<',
        'less than or equal to': '<=',
        'begins with': 'startswith',
        'does not begin with': 'not startswith',
        'ends with': 'endswith',
        'does not end with': 'not endswith',
        'contains': 'contains',
        'does not contain': 'not contains'
    }

    def apply_string_operation(series, operator, value):
        if operator in ['startswith', 'not startswith']:
            result = series.str.startswith(value)
            return ~result if operator.startswith('not') else result
        elif operator in ['endswith', 'not endswith']:
            result = series.str.endswith(value)
            return ~result if operator.startswith('not') else result
        elif operator in ['contains', 'not contains']:
            result = series.str.contains(value, regex=False)
            return ~result if operator.startswith('not') else result
        return None

    result = pd.Series([residual_value] * len(df), index=df.index)
    
    for condition in conditions:
        op = operator_map[condition['operator']]
        ref_val = condition['reference_value']
        cond_val = condition['conditional_value']

        # Handle numeric comparisons
        if op in ['==', '!=', '>', '>=', '<', '<=']:
            try:
                ref_val = float(ref_val)
                mask = eval(f"df['{source_column}'] {op} {ref_val}")
            except ValueError:
                mask = eval(f"df['{source_column}'].astype(str) {op} '{ref_val}'")
        # Handle string operations
        else:
            mask = apply_string_operation(df[source_column].astype(str), op, ref_val)

        result = result.mask(mask, cond_val)

    return result

def apply_pattern(df, source_column, pattern):
    """
    Extract pattern from source column using regex
    """
    try:
        processed_pattern = pattern.replace("\\\\", "\\")
        # Get all matches but only use the first column
        matches = df[source_column].astype(str).str.extract(processed_pattern, expand=True)
        return matches.iloc[:, 0]  # Return only the first column of matches
    except Exception as e:
        raise ValueError(f"Invalid regex pattern: {str(e)}")

def update_process_table_data(email, process_name, table_name, data, new_column):
    """
    Updates process table data and metadata with new column information
    
    Args:
        email (str): User's email
        process_name (str): Name of the process
        table_name (str): Name of the table
        data (list): List of dictionaries containing the updated table data
        new_column (dict): Dictionary containing new column information
            {
                'name': column name,
                'type': column type (string|float)
            }
    
    Returns:
        dict: Result of the operation
            {
                'success': bool,
                'error': str (optional)
            }
    """
    try:
        bucket = get_storage_bucket()
        
        # Find the DataFrame record in the database
        dataframe = DataFrame.query.filter_by(process_id=process_name, name=table_name).first()
        if not dataframe:
            return {'success': False, 'error': "DataFrame not found in database"}
        
        # Update metadata first - from storage
        metadata_path = f"{email}/process/{process_name}/metadata/{table_name}.json"
        metadata_blob = bucket.blob(metadata_path)
        
        if not metadata_blob.exists():
            return {
                'success': False,
                'error': f"Metadata not found for table {table_name}"
            }
        
        # Get existing metadata from storage
        try:
            metadata_content = metadata_blob.download_as_text()
            metadata = json.loads(metadata_content)
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to read metadata: {str(e)}"
            }
        
        # Update metadata with new column
        new_column_info = {
            "name": new_column['name'],
            "type": new_column['type']
        }
        
        # Update columns list if it doesn't exist
        if not any(col.get('name') == new_column['name'] for col in metadata['columns']):
            metadata['columns'].append(new_column_info)
        
        # Update column count
        metadata['columnCount'] = len(metadata['columns'])
        current_timestamp = datetime.now(timezone.utc)
        metadata['updatedAt'] = current_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Convert data to DataFrame and save as CSV
        df = pd.DataFrame(data)
        df_buffer = BytesIO()
        df.to_csv(df_buffer, index=False)
        df_buffer.seek(0)
        
        try:
            # Save updated metadata to storage
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type='application/json'
            )
            
            # Save updated DataFrame to storage
            data_path = f"{email}/process/{process_name}/dataframes/{table_name}.csv"
            data_blob = bucket.blob(data_path)
            data_blob.upload_from_file(
                df_buffer,
                content_type='text/csv'
            )
            
            # Update essential fields in DataFrame record
            dataframe.column_count = len(df.columns)
            dataframe.row_count = len(df)
            dataframe.updated_at = current_timestamp
            
            # Commit database changes
            db.session.commit()
            
            return {
                'success': True,
                'dataframeId': dataframe.id
            }
            
        except Exception as e:
            # Attempt to rollback metadata changes
            try:
                metadata['columns'] = [col for col in metadata['columns'] 
                                     if col.get('name') != new_column['name']]
                metadata['columnCount'] = len(metadata['columns'])
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type='application/json'
                )
                db.session.rollback()  # Rollback database changes
            except:
                pass  # If rollback fails, continue with error reporting
            
            return {
                'success': False,
                'error': f"Failed to update data: {str(e)}"
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }

def get_process_table_data(email, process_name, table_name, page=None, per_page=None):
    """
    Retrieves data from a process table, with optional pagination
    """
    try:
        bucket = get_storage_bucket()
        
        # Get table data from CSV file
        data_path = f"{email}/process/{process_name}/dataframes/{table_name}.csv"
        data_blob = bucket.blob(data_path)
        
        if not data_blob.exists():
            return {
                'error': f"Table {table_name} not found"
            }
        
        try:
            # Load CSV data directly into DataFrame
            data_content = data_blob.download_as_string()
            df = pd.read_csv(BytesIO(data_content))
            
            # Convert DataFrame to records
            all_data = df.to_dict('records')
            
            # When performing operations, we always want all data regardless of pagination
            if page is None or per_page is None:
                return {
                    'data': all_data,
                    'totalRows': len(all_data)
                }
            
            # Pagination only for display purposes
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
            
        except Exception as e:
            return {
                'error': f"Failed to read table data: {str(e)}"
            }
            
    except Exception as e:
        return {
            'error': f"Unexpected error: {str(e)}"
        }

def process_add_column(email, process_name, table_name, new_column_name, operation_type, operation_params):
    """
    Process the addition of a new column to a table
    
    Args:
        email (str): User's email
        process_name (str): Name of the process
        table_name (str): Name of the table
        new_column_name (str): Name of the new column
        operation_type (str): Type of operation (calculate|concatenate|conditional|pattern)
        operation_params (dict): Parameters specific to the operation type
            For calculate: {'operations': [...]}
            For concatenate: {'operations': [...]}
            For conditional: {'sourceColumn': str, 'conditions': [...], 'residualValue': any}
            For pattern: {'sourceColumn': str, 'pattern': str}
    
    Returns:
        dict: Result of the operation
            {
                'success': bool,
                'message': str,
                'columnName': str,
                'columnType': str,
                'error': str (optional)
            }
    """
    try:
        # Get the table data
        table_data = get_process_table_data(
            email=email,
            process_name=process_name,
            table_name=table_name,
            page=None,
            per_page=None
        )
        
        if 'error' in table_data:
            return {
                'success': False,
                'error': table_data['error']
            }
        
        df = pd.DataFrame(table_data['data'])
        
        try:
            if operation_type == 'calculate':
                operations = operation_params.get('operations', [])
                if not operations:
                    return {'success': False, 'error': "Calculation operations required"}
                result = apply_calculation(df, operations)
                column_type = 'float'

            elif operation_type == 'concatenate':
                operations = operation_params.get('operations', [])
                if not operations:
                    return {'success': False, 'error': "Concatenation operations required"}
                result = apply_concatenation(df, operations)
                column_type = 'string'

            elif operation_type == 'conditional':
                source_column = operation_params.get('sourceColumn')
                conditions = operation_params.get('conditions', [])
                residual_value = operation_params.get('residualValue')
                if not all([source_column, conditions, residual_value is not None]):
                    return {'success': False, 'error': "Source column, conditions, and residual value required"}
                result = apply_conditional(df, source_column, conditions, residual_value)
                column_type = 'string'

            elif operation_type == 'pattern':
                source_column = operation_params.get('sourceColumn')
                pattern = operation_params.get('pattern')
                if not source_column or not pattern:
                    return {'success': False, 'error': "Source column and pattern required"}
                result = apply_pattern(df, source_column, pattern)
                column_type = 'string'

            else:
                return {'success': False, 'error': "Invalid operation type"}

            # Add the new column and update the table
            df[new_column_name] = result
            update_result = update_process_table_data(
                email=email,
                process_name=process_name,
                table_name=table_name,
                data=df.to_dict('records'),
                new_column={
                    'name': new_column_name,
                    'type': column_type
                }
            )

            if not update_result.get('success'):
                return {
                    'success': False,
                    'error': update_result.get('error')
                }

            return {
                'success': True,
                'message': f"Column '{new_column_name}' added successfully",
                'columnName': new_column_name,
                'columnType': column_type
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error applying operation: {str(e)}"
            }

    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }

@add_column_bp.route('/apply/', methods=['POST'])
def apply_operation():
    """
    Endpoint for adding a new column to a table in a process
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Validate required parameters
        process_id = data.get('processId')
        table_name = data.get('tableName')
        new_column_name = data.get('newColumnName')
        operation_type = data.get('operationType')

        if not all([process_id, table_name, new_column_name, operation_type]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Get process and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get DataFrame record
        dataframe = DataFrame.query.filter_by(
            process_id=process_id,
            name=table_name
        ).first()
        if not dataframe:
            return jsonify({"error": "Table not found in process"}), 404

        # Map operation type to AddColumnSubType
        operation_subtype_map = {
            'calculate': AddColumnSubType.APPLY_CALCULATION.value,
            'conditional': AddColumnSubType.APPLY_CONDITIONAL.value,
            'pattern': AddColumnSubType.APPLY_PATTERN.value,
            'concatenate' : AddColumnSubType.APPLY_CONCAT.value
        }

        operation_subtype = operation_subtype_map.get(operation_type)
        if not operation_subtype:
            return jsonify({"error": "Invalid operation type"}), 400

        # Generate message based on operation type
        message = None
        if operation_type == 'calculate':
            operations = data.get('operations', [])
            columns = set()
            for op in operations:
                if op.get('column1'):
                    columns.add(op['column1'])
                if op.get('column2'):
                    columns.add(op['column2'])
            message = f'Add column {new_column_name} from {", ".join(sorted(columns))}'
        elif operation_type == 'concatenate':
            operations = data.get('operations', [])
            columns = [op['column'] for op in operations if op.get('column')]
            message = f'Add column {new_column_name} from {", ".join(columns)}'
        elif operation_type == 'conditional':
            source_column = data.get('sourceColumn')
            message = f'Add column {new_column_name} from {source_column}'
        elif operation_type == 'pattern':
            source_column = data.get('sourceColumn')
            message = f'Add column {new_column_name} from {source_column}'

        # Create DataFrameOperation record with IN_PROGRESS status
        df_operation = DataFrameOperation(
            process_id=process_id,
            dataframe_id=dataframe.id,
            operation_type=OperationType.ADD_COLUMN.value,
            operation_subtype=operation_subtype,
            payload=data,
            message=message
        )
        
        # Set the user_id after creation
        df_operation.user_id = user.id
        
        # Save initial operation record
        db.session.add(df_operation)
        db.session.commit()

        # Get the table data
        try:
            bucket = get_storage_bucket()
            csv_blob = bucket.blob(dataframe.storage_path)
            if not csv_blob.exists():
                df_operation.set_error("Table data not found")
                db.session.commit()
                return jsonify({"error": "Table data not found"}), 404

            # Read CSV data
            content = csv_blob.download_as_string()
            df = pd.read_csv(BytesIO(content))

            # Get existing metadata
            metadata_path = f"{email}/process/{process_id}/metadata/{table_name}.json"
            metadata_blob = bucket.blob(metadata_path)
            if metadata_blob.exists():
                metadata = json.loads(metadata_blob.download_as_string())
            else:
                df_operation.set_error("Metadata not found")
                db.session.commit()
                return jsonify({"error": "Metadata not found"}), 404

        except Exception as e:
            df_operation.set_error(f"Failed to load table data: {str(e)}")
            db.session.commit()
            return jsonify({"error": f"Failed to load table data: {str(e)}"}), 500

        try:
            if operation_type == 'calculate':
                operations = data.get('operations', [])
                if not operations:
                    df_operation.set_error("Calculation operations required")
                    db.session.commit()
                    return jsonify({"error": "Calculation operations required"}), 400
                result = apply_calculation(df, operations)
                column_type = 'float'

            elif operation_type == 'concatenate':
                operations = data.get('operations', [])
                if not operations:
                    df_operation.set_error("Concatenation operations required")
                    db.session.commit()
                    return jsonify({"error": "Concatenation operations required"}), 400
                result = apply_concatenation(df, operations)
                column_type = 'string'

            elif operation_type == 'conditional':
                source_column = data.get('sourceColumn')
                conditions = data.get('conditions', [])
                residual_value = data.get('residualValue')
                if not all([source_column, conditions, residual_value is not None]):
                    df_operation.set_error("Source column, conditions, and residual value required")
                    db.session.commit()
                    return jsonify({"error": "Source column, conditions, and residual value required"}), 400
                result = apply_conditional(df, source_column, conditions, residual_value)
                column_type = 'string'

            elif operation_type == 'pattern':
                source_column = data.get('sourceColumn')
                pattern = data.get('pattern')
                if not source_column or not pattern:
                    df_operation.set_error("Source column and pattern required")
                    db.session.commit()
                    return jsonify({"error": "Source column and pattern required"}), 400
                result = apply_pattern(df, source_column, pattern)
                column_type = 'string'

            else:
                return jsonify({"error": "Invalid operation type"}), 400

            # Add the new column to DataFrame
            df[new_column_name] = result

            # Update metadata
            metadata['columnCount'] = len(df.columns)
            metadata['columns'].append({
                "name": new_column_name,
                "type": column_type,
                "operation": operation_type
            })
            metadata['updatedAt'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

            try:
                # Save updated DataFrame
                df_buffer = BytesIO()
                df.to_csv(df_buffer, index=False)
                df_buffer.seek(0)
                csv_blob.upload_from_file(df_buffer, content_type='text/csv')

                # Save updated metadata
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type='application/json'
                )

                # Update DataFrame record and mark operation as successful
                dataframe.column_count = len(df.columns)
                dataframe.updated_at = datetime.now(timezone.utc)
                df_operation.set_success()
                db.session.commit()

                # After operations are successfully applied and before returning success
                # Add this code to gather metadata for the resulting DataFrame
                result_metadata = {
                    "columns": df.columns.tolist(),
                    "columnTypes": {col: str(df[col].dtype) for col in df.columns},
                    "summary": {
                        "rowCount": len(df),
                        "nullCounts": {col: int(df[col].isna().sum()) for col in df.columns},
                        "uniqueCounts": {col: int(df[col].nunique()) for col in df.columns}
                    }
                }
                
                return jsonify({
                    "success": True,
                    "message": message,  # Use the same message in the response
                    "columnName": new_column_name,
                    "columnType": column_type,
                    "dataframeId": dataframe.id,
                    "operationId": df_operation.id,
                    "operationStatus": df_operation.status,
                    "metadata": result_metadata
                })

            except Exception as e:
                df_operation.set_error(f"Failed to save updates: {str(e)}")
                db.session.commit()
                return jsonify({"error": f"Failed to save updates: {str(e)}"}), 500

        except Exception as e:
            df_operation.set_error(f"Error applying operation: {str(e)}")
            db.session.commit()
            return jsonify({"error": f"Error applying operation: {str(e)}"}), 400

    except Exception as e:
        logger.error(f"Apply operation error: {str(e)}")
        return jsonify({"error": str(e)}), 500
