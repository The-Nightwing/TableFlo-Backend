from flask import Blueprint, request, jsonify
from models import User, UserProcess, ProcessOperation, ProcessFileKey, DataFrame, db, DataFrameOperation, DataFrameBatchOperation, FormattingStep
from firebase_config import get_storage_bucket
import json
from datetime import datetime
import pandas as pd
from io import BytesIO
import os
import traceback
from formatting import format_excel_file
from edit_file import edit_file, process_columns_and_types
from sort_filter import process_sort_filter_data
from group_pivot import process_pivot_table
from add_column import process_add_column
from merge_files import process_merge_tables, process_reconciliation
from file_operations import process_file_operations
import copy
import time

process_bp = Blueprint('process', __name__, url_prefix='/api/process/')

def validate_sort_filter_operation(operation_data):
    """Validate sort/filter operation parameters"""
    params = operation_data.get('params', {})
    
    # Validate that at least one of sortConfig or filterConfig exists
    if not params.get('sortConfig') and not params.get('filterConfig'):
        raise ValueError("Either sort or filter configuration is required")
        
    return {
        'fileName': params.get('fileName'),
        'sheet': params.get('sheet'),
        'sortConfig': params.get('sortConfig', []),
        'filterConfig': params.get('filterConfig', [])
    }

def validate_formatting_operation(operation_data):
    """Validate formatting operation parameters"""
    params = operation_data.get('params', {})
    
    required_fields = ['type', 'location', 'format']
    for field in required_fields:
        if field not in params:
            raise ValueError(f"Missing required field: {field}")
            
    return {
        'fileName': params.get('fileName'),
        'sheet': params.get('sheet'),
        'formattingConfig': params
    }

def validate_add_column_operation(operation_data):
    """Validate add column operation parameters"""
    params = operation_data.get('params', {})
    
    if not params.get('type') or not params.get('newColumnName'):
        raise ValueError("Column type and new column name are required")
        
    valid_types = ['calculate', 'concatenate', 'conditional', 'pattern']
    if params['type'] not in valid_types:
        raise ValueError(f"Invalid column type. Must be one of: {valid_types}")
        
    return {
        'fileName': params.get('fileName'),
        'sheet': params.get('sheet'),
        'type': params['type'],
        'newColumnName': params['newColumnName'],
        'params': params.get('params', []),
        'sourceColumn': params.get('sourceColumn'),
        'pattern': params.get('pattern'),
        'residualValue': params.get('residualValue')
    }

def validate_merge_operation(operation_data):
    """Validate merge operation parameters"""
    params = operation_data.get('params', {})
    
    if not params.get('files') or len(params['files']) != 2:
        raise ValueError("Exactly two files are required for merge operation")
        
    if not params.get('keyPairs'):
        raise ValueError("Key pairs for merging are required")
        
    return {
        'files': params['files'],
        'keyPairs': params['keyPairs'],
        'method': params.get('method', 'inner'),
        'showCountSummary': params.get('showCountSummary', False)
    }

def validate_group_pivot_operation(operation_data):
    """Validate group and pivot operation parameters"""
    params = operation_data.get('params', {})
    
    if not params.get('rowIndex') or not params.get('pivotValues'):
        raise ValueError("Row index and pivot values are required")
        
    return {
        'fileName': params.get('fileName'),
        'sheet': params.get('sheet'),
        'rowIndex': params['rowIndex'],
        'columnIndex': params.get('columnIndex'),
        'pivotValues': params['pivotValues']
    }

def validate_file_operation(operation_data):
    """Validate file operation parameters"""
    params = operation_data.get('params', {})
    
    if not params.get('operation'):
        raise ValueError("Operation type is required")
        
    valid_operations = ['replace', 'rename', 'reorder']
    if params['operation'] not in valid_operations:
        raise ValueError(f"Invalid file operation. Must be one of: {valid_operations}")
    
    operation_data = params.get('data', {})
    
    # Validate based on operation type
    if params['operation'] == 'replace':
        if not operation_data.get('selectedColumns'):
            raise ValueError("Selected columns are required for replace operation")
        if 'oldValue' not in operation_data or 'newValue' not in operation_data:
            raise ValueError("Old value and new value are required for replace operation")
            
    elif params['operation'] == 'rename':
        if not operation_data.get('columns'):
            raise ValueError("Columns configuration is required for rename operation")
        for col_config in operation_data['columns']:
            if not col_config.get('column') or not col_config.get('newName'):
                raise ValueError("Column and newName are required for each rename configuration")
                
    elif params['operation'] == 'reorder':
        if not operation_data.get('columns'):
            raise ValueError("Columns configuration is required for reorder operation")
        for col_config in operation_data['columns']:
            if not col_config.get('column') or 'positions' not in col_config:
                raise ValueError("Column and positions are required for each reorder configuration")
    
    return {
        'fileName': params.get('file_name'),  # Note the different key names to match file_operations.py
        'sheet': params.get('sheet_name'),
        'operation': params['operation'],
        'data': operation_data,
        'newFileName': params.get('newFileName'),
        'replace_existing': params.get('replace_existing', False)
    }

def validate_edit_file_operation(operation_data):
    """Validate edit file operation parameters"""
    params = operation_data.get('params', {})
    
    if not params.get('tables'):
        raise ValueError("Tables configuration is required")
        
    # Validate each table's configuration
    for table_config in params['tables']:
        if not table_config.get('tableName'):
            raise ValueError("Table name is required for each table")
        if not table_config.get('columnSelections'):
            raise ValueError(f"Column selections are required for table {table_config['tableName']}")
            
    return params

def validate_reconcile_operation(operation_data):
    """Validate reconcile operation parameters"""
    params = operation_data.get('params', {})
    
    # Validate required fields
    if not params.get('files') or len(params.get('files', [])) != 2:
        raise ValueError("Exactly two files are required for reconciliation")
        
    if not params.get('keys'):
        raise ValueError("Matching keys are required")
        
    if not params.get('values'):
        raise ValueError("Values configuration is required")
        
    if not params.get('output_file'):
        raise ValueError("Output file name is required")
        
    # Validate files structure
    for file_info in params['files']:
        if not file_info.get('file_name'):
            raise ValueError("File name is required for both files")
        if not file_info.get('sheet_name'):
            raise ValueError("Sheet name is required for both files")
            
    # Validate keys structure
    for key in params['keys']:
        if not key.get('file1') or not key.get('file2'):
            raise ValueError("Both file1 and file2 columns are required for each key")
        if not key.get('criteria') in ['exact', 'fuzzy']:
            raise ValueError("Invalid criteria type for key matching. Must be 'exact' or 'fuzzy'")
            
    # Validate values structure
    for value in params['values']:
        if not value.get('file1') or not value.get('file2'):
            raise ValueError("Both file1 and file2 columns are required for each value")
        if value.get('threshold_type') and value.get('threshold_type') not in ['percent', 'amount']:
            raise ValueError("Invalid threshold type. Must be 'percent' or 'amount'")
            
    return {
        'files': params['files'],
        'keys': params['keys'],
        'values': params['values'],
        'cross_reference': params.get('cross_reference', []),
        'output_file': params['output_file'],
        'settings': params.get('settings', {
            'method': 'one-to-one',
            'duplicate': 'first_occurrence',
            'basis_column': {'file1': '', 'file2': ''},
            'fuzzy_preference': []
        })
    }

def validate_operation(operation):
    """Validate operation based on its type"""
    if not isinstance(operation, dict):
        raise ValueError("Invalid operation format")
        
    operation_type = operation.get('type')
    if not operation_type:
        raise ValueError("Operation type is required")
        
    # Updated validation map with all operation types
    validation_map = {
        'sort_filter': validate_sort_filter_operation,
        'formatting': validate_formatting_operation,
        'add_column': validate_add_column_operation,
        'merge': validate_merge_operation,
        'group_pivot': validate_group_pivot_operation,
        'file_operations': validate_file_operation,
        'edit_file': validate_edit_file_operation,
        'reconcile': validate_reconcile_operation
    }
    
    if operation_type not in validation_map:
        raise ValueError(f"Unsupported operation type: {operation_type}")
        
    validated_params = validation_map[operation_type](operation)
    return {
        'type': operation_type,
        'parameters': validated_params
    }

def execute_edit_file_operation(email, params, input_file=None):
    """Execute edit file operation"""
    try:
        # Create batch operation record
        batch_operation = DataFrameBatchOperation(
            process_id=params.get('processId'),
            payload=params,
            dataframe_ids=[],
            operation_ids=[],
            total_dataframes=len(params.get('tables', []))
        )
        
        db.session.add(batch_operation)
        db.session.commit()

        try:
            # Process each table
            for table_config in params.get('tables', []):
                result = process_columns_and_types(
            email=email,
                    process_id=params['processId'],
                    table_name=table_config['tableName'],
                    column_selections=table_config.get('columnSelections'),
                    column_types=table_config.get('columnTypes'),
                    datetime_formats=table_config.get('datetimeFormats')
                )
                
                if result.get('success'):
                    batch_operation.increment_success_count()
                    batch_operation.dataframe_ids.append(result['id'])
                else:
                    error_msg = f"Error processing {table_config['tableName']}: {result.get('error')}"
                    batch_operation.set_error(error_msg)
                    db.session.commit()
                    return {
                        "success": False,
                        "error": error_msg,
                        "batchOperationId": batch_operation.id
                    }

            db.session.commit()
            return {
                "success": True,
                "message": "Tables processed successfully",
                "batchOperationId": batch_operation.id,
                "successCount": batch_operation.successful_dataframes,
                "totalCount": batch_operation.total_dataframes
            }

        except Exception as e:
            batch_operation.set_error(str(e))
            db.session.commit()
            raise e

    except Exception as e:
        raise Exception(f"Error in edit_file operation: {str(e)}")

def execute_sort_filter_operation(email, params, input_file=None):
    """Execute sort and filter operation"""
    try:
        file_name = input_file if input_file else params.get('fileName')
        
        result = process_sort_filter_data(
            email=email,
            file_name=file_name,
            sheet_name=params.get('sheet'),
            sort_config=params.get('sortConfig', []),
            filter_config=params.get('filterConfig', []),
            new_file_name=params.get('newFileName', ''),
            output_format=params.get('format', '')
        )

        return result

    except Exception as e:
        raise Exception(f"Error in sort_filter operation: {str(e)}")

def execute_formatting_operation(email, params, input_file=None):
    """Execute formatting operation"""
    try:
        file_name = input_file if input_file else params.get('fileName')
        sheet = params.get('sheet')
        formatting_config = params.get('formattingConfig', {})
        new_file_name = params.get('newFileName', '')

        # Convert single formatting config to list format
        formatting_configs = [formatting_config] if formatting_config else []

        result = format_excel_file(
            email=email,
            file_name=file_name,
            sheet_name=sheet,
            formatting_configs=formatting_configs,
            new_file_name=new_file_name
        )

        if not result.get('success'):
            raise Exception(result.get('error'))

        return result

    except Exception as e:
        raise Exception(f"Error in formatting operation: {str(e)}")

def execute_add_column_operation(email, params, input_file=None):
    """Execute add column operation"""
    try:
        file_name = input_file if input_file else params.get('fileName')
        
        result = process_add_column(
            email=email,
            file_name=file_name,
            sheet_name=params.get('sheet'),
            operations=params.get('operations', []),
            output_format=params.get('format', 'xlsx'),
            new_file_name=params.get('newFileName', '')
        )

        if not result.get('success'):
            raise Exception(result.get('error'))

        return result

    except Exception as e:
        raise Exception(f"Error in add_column operation: {str(e)}")

def execute_merge_operation(email, params):
    """Execute merge operation"""
    try:
        result = process_merge_tables(
            email=email,
            file1_name=params.get('file1Name'),
            file2_name=params.get('file2Name'),
            merge_type=params.get('mergeType'),
            merge_method=params.get('mergeMethod', 'inner'),
            key_pairs=params.get('keyPairs', []),
            show_count_summary=params.get('showCountSummary', False),
            new_file_name=params.get('newFileName', '')
        )

        if not result.get('success'):
            raise Exception(result.get('error'))

        return result

    except Exception as e:
        raise Exception(f"Error in merge operation: {str(e)}")

def execute_group_pivot_operation(email, params, input_file=None):
    """Execute group pivot operation"""
    try:
        file_name = input_file if input_file else params.get('fileName')
        
        result = process_pivot_table(
            email=email,
            file_name=file_name,
            row_index=params.get('rowIndex', []),
            column_index=params.get('columnIndex'),
            pivot_values=params.get('pivotValues', []),
            new_file_name=params.get('newFileName', '')
        )

        if not result.get('success'):
            raise Exception(result.get('error'))

        return result

    except Exception as e:
        raise Exception(f"Error in group_pivot operation: {str(e)}")

def execute_file_operation(email, params):
    """Execute file operation"""
    try:
        result = process_file_operations(
            email=email,
            file_name=params.get('file_name'),
            operations=params.get('operation'),
            sheet_name=params.get('sheet_name'),
            new_file_name=params.get('newFileName', '')
        )

        if not result.get('success'):
            raise Exception(result.get('error'))

        return result

    except Exception as e:
        raise Exception(f"Error in file operation: {str(e)}")

def execute_reconciliation_operation(email, params):
    """Execute reconciliation operation"""
    try:
        result = process_reconciliation(email, params)
        
        if not result.get('success'):
            raise Exception(result.get('error'))

        return result

    except Exception as e:
        raise Exception(f"Error in reconciliation operation: {str(e)}")

def execute_operation(operation_type, email, parameters, input_file=None):
    """Execute an operation based on its type"""
    operation_map = {
        'edit_file': execute_edit_file_operation,
        'sort_filter': execute_sort_filter_operation,
        'formatting': execute_formatting_operation,
        'add_column': execute_add_column_operation,
        'merge': execute_merge_operation,
        'group_pivot': execute_group_pivot_operation,
        'file_operations': execute_file_operation,
        'reconcile': execute_reconciliation_operation
    }
    
    if operation_type not in operation_map:
        raise ValueError(f"Unsupported operation type: {operation_type}")
        
    return operation_map[operation_type](email, parameters, input_file)

def validate_file_compatibility(original_metadata, new_file_metadata):
    """
    Validate if new file has same structure as original file.
    Returns (is_compatible, error_message)
    """
    try:
        # Check if all sheets exist
        for sheet_name, original_sheet in original_metadata['sheets'].items():
            if sheet_name not in new_file_metadata['sheets']:
                return False, f"Missing sheet: {sheet_name}"
            
            new_sheet = new_file_metadata['sheets'][sheet_name]
            
            # Check if all columns exist
            missing_cols = set(original_sheet['columns']) - set(new_sheet['columns'])
            if missing_cols:
                return False, f"Missing columns in sheet {sheet_name}: {', '.join(missing_cols)}"
            
            # Check column types
            for col in original_sheet['columns']:
                if original_sheet['columnTypes'][col] != new_sheet['columnTypes'][col]:
                    return False, f"Column type mismatch for {col} in sheet {sheet_name}: expected {original_sheet['columnTypes'][col]}, got {new_sheet['columnTypes'][col]}"
        
        return True, None
    except Exception as e:
        return False, str(e)

def get_file_metadata(email, file_name):
    """Get metadata for a file from Firebase."""
    bucket = get_storage_bucket()
    metadata_blob = bucket.blob(f"{email}/metadata/{file_name}.json")
    if not metadata_blob.exists():
        raise FileNotFoundError(f"Metadata not found for file: {file_name}")
    
    metadata_content = metadata_blob.download_as_string()
    return json.loads(metadata_content)

def execute_process_with_files(process, file_mappings, email):
    """Execute a process with given file mappings."""
    try:
        intermediate_results = []
        current_file_mapping = file_mappings.copy()

        # Get operations in sequence order
        operations = sorted(process.operations, key=lambda x: x.sequence)

        for operation in operations:
            try:
                operation_params = copy.deepcopy(operation.parameters)
                
                # Get input and output key names
                input_key = operation_params.get('fileName')
                output_key = operation_params.get('newFileName')

                if input_key not in current_file_mapping:
                    return {
                        "success": False,
                        "error": f"Error in operation {int(operation.sequence) + 1}: Unknown file key '{input_key}'",
                        "intermediate_results": intermediate_results
                    }

                # Replace input key with actual filename for execution
                operation_params['fileName'] = current_file_mapping[input_key]
                
                # Execute the operation
                result = execute_operation(
                    email=email,
                    operation_type=operation.operation_name,
                    parameters=operation_params,
                    input_file=None
                )
                
                if not result.get('success'):
                    return {
                        "success": False,
                        "error": f"Error in operation {int(operation.sequence) + 1}: {result.get('error')}",
                        "intermediate_results": intermediate_results
                    }

                # Extract filename from download URL
                physical_file = None
                if result.get('downloadUrl'):
                    url_path = result['downloadUrl'].split('?')[0]
                    physical_file = url_path.split('/')[-1]

                # Update file mapping with the output
                if output_key and physical_file:
                    current_file_mapping[output_key] = physical_file

                # Add to intermediate results
                intermediate_results.append({
                    "step": int(operation.sequence) + 1,
                    "operation_type": operation.operation_name,
                    "input_key": input_key,
                    "output_key": output_key,
                    "physical_file": physical_file,
                    "download_url": result.get('downloadUrl'),
                    "preview_url": result.get('previewUrl', '')
                })

                # Debug logging
                print(f"Current file mappings after operation {operation.sequence}: {current_file_mapping}")

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error in operation {int(operation.sequence) + 1}: {str(e)}",
                    "intermediate_results": intermediate_results
                }

        return {
            "success": True,
            "file_mappings": current_file_mapping,
            "intermediate_results": intermediate_results,
            "final_result": {
                "file_name": intermediate_results[-1]["physical_file"] if intermediate_results else None,
                "file_key": intermediate_results[-1]["output_key"] if intermediate_results else None,
                "download_url": intermediate_results[-1]["download_url"] if intermediate_results else None,
                "preview_url": intermediate_results[-1]["preview_url"] if intermediate_results else None
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "intermediate_results": []
        }
    
@process_bp.route('/create', methods=['POST'])
def create_process():
    """Create a new process template."""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Create the process
        process = UserProcess(
            user_id=user.id,
            process_name=data['name']
        )

        # Extract and store file key requirements
        file_keys_metadata = {}
        file_mappings = data.get('file_mappings', {})

        # Collect metadata for all initial files
        for key_name, filename in file_mappings.items():
            try:
                metadata = get_file_metadata(email, filename)
                file_keys_metadata[key_name] = metadata
                
                # Create ProcessFileKey record
                process_file_key = ProcessFileKey(
                    key_name=key_name,
                    required_structure=metadata
                )
                process.file_keys.append(process_file_key)
                
            except FileNotFoundError as e:
                return jsonify({"error": f"File metadata not found for {filename}"}), 400

        # Add operations
        for i, operation in enumerate(data['operations']):
            operation_type = operation['type']
            parameters = operation['params']
            
            if operation_type == 'edit_file':
                # Create batch operation for edit_file type
                batch_operation = DataFrameBatchOperation(
                    process_id=process.id,
                    payload=parameters,
                    dataframe_ids=[],
                    operation_ids=[],
                    total_dataframes=len(parameters.get('tables', []))
                )
                db.session.add(batch_operation)
                db.session.commit()
                
                process_operation = ProcessOperation(
                sequence=float(i),
                    operation_name=operation_type,
                    parameters=parameters,
                    dataframe_operation_id=batch_operation.id
                )
            else:
                process_operation = ProcessOperation(
                    sequence=float(i),
                    operation_name=operation_type,
                    parameters=parameters
                )
            
            process.operations.append(process_operation)

        db.session.add(process)
        db.session.commit()

        # Execute the process with initial files
        result = execute_process_with_files(process, file_mappings, email)
        
        if not result.get('success'):
            return jsonify(result), 400

        return jsonify({
            "success": True,
            "message": "Process created and executed successfully",
            "process_id": process.id,
            "process_name": process.process_name,
            "file_keys": {key.key_name: key.required_structure for key in process.file_keys},
            "execution_results": result
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@process_bp.route('/list', methods=['GET'])
def list_processes():
    """List all non-draft processes for a user"""
    try:
        email = request.headers.get('X-User-Email')
        if not email:
            return jsonify({"error": "User email is required"}), 400
            
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        # Filter for non-draft processes and active processes
        processes = UserProcess.query.filter_by(
            user_id=user.id,
            is_draft=False,
            is_active=True,
            original_process_id = None
        ).order_by(UserProcess.updated_at.desc()).all()
        
        return jsonify({
            "success": True,
            "processes": [
                {
                    **process.to_dict(),
                }
                for process in processes
            ],
            "totalCount": len(processes)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>', methods=['GET'])
def get_process(process_id):
    """Get process details including file key names."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User {email} not found."}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found"}), 404

        process_data = json.loads(process.data)
        
        # Add file mappings to response
        response = {
            "id": process.id,
            "name": process.process_name,
            "operations": process_data['operations'],
            "file_mappings": process_data.get('file_mappings', {}),  # Map of key_names to original filenames
            "file_metadata": process_data.get('file_metadata', {})   # Original file metadata for each key_name
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@process_bp.route('/start', methods=['POST'])
def start_process():
    """Initialize a new process with basic information."""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Validate required fields
        process_name = data.get('name')
        if not process_name:
            return jsonify({"error": "Process name is required"}), 400

        # Check if process with same name exists for this user
        existing_process = UserProcess.query.filter_by(
            user_id=user.id,
            process_name=process_name
        ).first()

        if existing_process:
            return jsonify({
                "error": "A process with this name already exists",
                "existingProcess": {
                    "id": existing_process.id,
                    "name": existing_process.process_name,
                    "createdAt": existing_process.created_at.isoformat(),
                    "updatedAt": existing_process.updated_at.isoformat()
                }
            }), 409  # HTTP 409 Conflict

        # Create the process with basic info
        process = UserProcess(
            user_id=user.id,
            process_name=process_name
        )

        db.session.add(process)
        db.session.commit()

        # Create directory structure in Firebase Storage using process ID
        bucket = get_storage_bucket()
        base_path = f"{email}/process/{process.id}"
        required_dirs = [
            f"{base_path}/dataframes/",
            f"{base_path}/metadata/",
            f"{base_path}/previews/"
        ]

        # Create empty placeholder files to ensure directories exist
        for dir_path in required_dirs:
            placeholder = bucket.blob(f"{dir_path}.placeholder")
            placeholder.upload_from_string('')

        return jsonify({
            "success": True,
            "message": "Process initialized successfully",
            "process": {
                "id": process.id,
                "name": process.process_name,
                "userId": user.id,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>', methods=['PATCH'])
def update_process(process_id):
    """Update process information."""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get the process
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found"}), 404

        bucket = get_storage_bucket()
        old_base_path = f"{email}/process/{process.process_name}"

        # Update process name if provided
        if 'name' in data and data['name'] != process.process_name:
            new_base_path = f"{email}/process/{data['name']}"
            
            # List all blobs in the old directory
            blobs = bucket.list_blobs(prefix=old_base_path)
            
            # Move each file to the new location
            for blob in blobs:
                new_path = blob.name.replace(old_base_path, new_base_path)
                new_blob = bucket.blob(new_path)
                
                # Copy the file to new location
                bucket.copy_blob(blob, bucket, new_path)
                # Delete the old file
                blob.delete()

            # Update process name
            process.process_name = data['name']

        # Update file mappings if provided
        if 'file_mappings' in data:
            file_mappings = data['file_mappings']
            
            for key_name, filename in file_mappings.items():
                try:
                    metadata = get_file_metadata(email, filename)
                    
                    # Check if a file key with this name already exists
                    existing_key = ProcessFileKey.query.filter_by(
                        process_id=process.id,
                        key_name=key_name
                    ).first()
                    
                    if existing_key:
                        # Update existing key
                        existing_key.required_structure = metadata
                    else:
                        # Add new file key
                        process_file_key = ProcessFileKey(
                            key_name=key_name,
                            required_structure=metadata
                        )
                        process.file_keys.append(process_file_key)
                        
                except FileNotFoundError:
                    return jsonify({"error": f"File metadata not found for {filename}"}), 400

        # Update operations if provided
        if 'operations' in data:
            # Only update the fields of the edited operation, preserve sequence/order
            incoming_ops = data['operations']
            # Get all existing operations for this process
            existing_ops = {op.id: op for op in ProcessOperation.query.filter_by(process_id=process.id).all()}
            for op_data in incoming_ops:
                op_id = op_data.get('id')
                if not op_id or op_id not in existing_ops:
                    continue  # skip new ops or invalid ids
                op = existing_ops[op_id]
                # Only update fields except sequence
                if op_data['type'] == 'edit_file':
                    # If edit_file, update batch operation payload if needed
                    batch_op_id = op.dataframe_operation_id
                    batch_op = DataFrameBatchOperation.query.get(batch_op_id) if batch_op_id else None
                    if batch_op:
                        batch_op.payload = op_data['params']
                        db.session.add(batch_op)
                    op.parameters = op_data['params']
                else:
                    op.parameters = op_data['params']
                if 'title' in op_data:
                    op.title = op_data['title']
                if 'description' in op_data:
                    op.description = op_data['description']
                db.session.add(op)

        # Add operation to database
        db.session.add(process)
        
        # Commit both process update and new operation in one transaction
        try:
            db.session.commit()
        except Exception as commit_error:
            db.session.rollback()
            print(f"Error committing changes: {str(commit_error)}")
            raise commit_error

        current_base_path = f"{email}/process/{process.process_name}"
        
        # Ensure directory structure exists
        required_dirs = [
            f"{current_base_path}/dataframes/",
            f"{current_base_path}/metadata/",
            f"{current_base_path}/previews/"
        ]

        for dir_path in required_dirs:
            placeholder = bucket.blob(f"{dir_path}.placeholder")
            if not placeholder.exists():
                placeholder.upload_from_string('')

        # Fetch updated operations in correct sequence to maintain order
        sorted_operations = ProcessOperation.query.filter_by(process_id=process.id).order_by(ProcessOperation.sequence).all()

        return jsonify({
            "success": True,
            "message": "Process updated successfully",
            "process": {
                "id": process.id,
                "name": process.process_name,
                "userId": user.id,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat(),
                #"storagePath": current_base_path,
                "fileKeys": {key.key_name: key.required_structure for key in process.file_keys},
                "operations": [op.to_dict() for op in sorted_operations]
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/original-dataframes', methods=['GET'])
def list_original_dataframes(process_id):
    """List all originally uploaded DataFrames for a process."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get all originally uploaded DataFrames for this process
        original_dataframes = DataFrame.query.filter_by(
            process_id=process_id,
            is_originally_uploaded=True,
            is_active=True
        ).all()

        return jsonify({
            "success": True,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            },
            "dataframes": [
                {
                    "id": df.id,
                    "name": df.name,
                    "rowCount": df.row_count,
                    "columnCount": df.column_count,
                    "metadata": df.data_metadata,
                    "createdAt": df.created_at.isoformat() if df.created_at else None
                } for df in original_dataframes
            ]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@process_bp.route('/<process_id>/dataframes', methods=['GET'])
def list_process_dataframes(process_id):
    """
    List all DataFrames associated with a process, including their details and column information.
    """
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get the process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get all DataFrames for this process
        dataframes = DataFrame.query.filter_by(process_id=process_id, is_draft=False).all()
        
        bucket = get_storage_bucket()
        
        def process_dataframe(df):
            """Process a single DataFrame to get its metadata and download URL"""
            try:
                # Initialize metadata content with basic DataFrame info
                metadata_content = {
                    "columns": [],
                    "rowCount": df.row_count,
                    "columnCount": df.column_count,
                    "description": "",
                    "sourceTableName": None,
                    "originalFileName": None,
                    "originalSheetName": None
                }

                # Try to get metadata from Firebase
                metadata_path = f"{email}/process/{process_id}/metadata/{df.name}.json"
                metadata_blob = bucket.blob(metadata_path)
                
                if metadata_blob.exists():
                    # Update metadata content with file data if available
                    file_metadata = json.loads(metadata_blob.download_as_string())
                    metadata_content.update(file_metadata)

                # Generate signed URL for the DataFrame CSV
                csv_blob = bucket.blob(df.storage_path)
                download_url = None
                if csv_blob.exists():
                    download_url = csv_blob.generate_signed_url(
                        expiration=604800,
                        version='v4'
                    )

                return {
                    "id": df.id,
                    "name": df.name,
                    "processId": df.process_id,
                    "createdAt": df.created_at.isoformat() if df.created_at else None,
                    "updatedAt": df.updated_at.isoformat() if df.updated_at else None,
                    "rowCount": df.row_count,
                    "columnCount": df.column_count,
                    "description": metadata_content.get("description", ""),
                    "sourceTableName": metadata_content.get("sourceTableName"),
                    "downloadUrl": download_url,
                    "originalFileName": metadata_content.get("originalFileName"),
                    "originalSheetName": metadata_content.get("originalSheetName"),
                    "columns": metadata_content.get("columns", [])
                }
            except Exception as e:
                print(f"Error processing DataFrame {df.id}: {str(e)}")
                return None

        # Use ThreadPoolExecutor to process DataFrames in parallel
        from concurrent.futures import ThreadPoolExecutor
        dataframe_list = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all DataFrame processing tasks
            future_to_df = {executor.submit(process_dataframe, df): df for df in dataframes}
            
            # Collect results as they complete
            for future in future_to_df:
                result = future.result()
                if result is not None:
                    dataframe_list.append(result)

        return jsonify({
            "success": True,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            },
            "dataframes": dataframe_list,
            "totalCount": len(dataframe_list)
        })

    except Exception as e:
        print(f"Error in list_process_dataframes: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/operations', methods=['POST'])
def add_operation_to_process(process_id):
    """Associate an existing DataFrameOperation with a process."""
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Set process as draft and update timestamp
        process.is_draft = True
        process.updated_at = datetime.now()
        db.session.add(process)

        # Rest of the validation and operation creation...
        required_fields = ['dataframeOperationId']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Determine the next sequence number
        last_operation = ProcessOperation.query.filter_by(process_id=process_id).order_by(ProcessOperation.sequence.desc()).first()
        next_sequence = (last_operation.sequence + 1.0) if last_operation else 1.0

        operation_type = data.get('operationType')
        
        # Get optional title from request body
        title = data.get('title')
        
        if operation_type == 'edit_file':
            # Get the BatchOperation
            batch_operation = DataFrameBatchOperation.query.get(data['dataframeOperationId'])
            if not batch_operation:
                return jsonify({"error": "BatchOperation not found"}), 404

            # Verify operation ownership through process
            if batch_operation.process_id != process_id:
                return jsonify({"error": "BatchOperation does not belong to this process"}), 403

            # Create ProcessOperation without dataframe_operation_id
            process_operation = ProcessOperation(
                process_id=process_id,
                sequence=next_sequence,
                operation_name='edit_file',
                title=title or "Define Inputs",  # Use provided title or default
                description=batch_operation.message,
                parameters={
                    **batch_operation.payload,
                    'batchOperationId': batch_operation.id
                },
                is_active=True
            )
        else:
            # Get the DataFrameOperation for other operation types
            df_operation = DataFrameOperation.query.get(data['dataframeOperationId'])
            if not df_operation:
                return jsonify({"error": "DataFrameOperation not found"}), 404

            # Verify operation ownership through process
            if df_operation.process_id != process_id:
                return jsonify({"error": "Operation does not belong to this process"}), 403

            # Set title based on operation type and subtype
            operation_title = title  # Default to provided title
            if not operation_title:
                if df_operation.operation_type == "add_column":
                    subtype = df_operation.operation_subtype.lower()
                    if "calcul" in subtype:
                        operation_title = "Add Column – Calculate"
                    elif "concat" in subtype:
                        operation_title = "Add Column – Concatenate"
                    elif "conditional" in subtype:
                        operation_title = "Add Column – Conditional"
                    elif "pattern" in subtype:
                        operation_title = "Add Column – Pattern"
                elif df_operation.operation_type == "merge_files":
                    operation_title = "Merge"
                elif df_operation.operation_type == "group_pivot":
                    operation_title = "Pivot"
                elif df_operation.operation_type == "sort_filter":
                    payload_str = str(df_operation.payload).lower()
                    if "sortconfig" in payload_str:
                        operation_title = "Sort"
                    elif "filterconfig" in payload_str:
                        operation_title = "Filter"
                elif df_operation.operation_type == "replace_rename_reorder":
                    operation_title = "Replace Rename Reorder"
                elif df_operation.operation_type == "reconcile_files":
                    operation_title = "Reconcile"
                elif df_operation.operation_type == "apply_formatting":
                    operation_title = "Formatting"

            # Create ProcessOperation
            process_operation = ProcessOperation(
                process_id=process_id,
                sequence=next_sequence,
                operation_name=df_operation.operation_type,
                title=operation_title,
                description=df_operation.message,
                parameters=df_operation.payload,
                dataframe_operation_id=df_operation.id,
                is_active=True
            )

        # Add operation to database
        db.session.add(process_operation)
        
        # Commit both process update and new operation in one transaction
        try:
            db.session.commit()
        except Exception as commit_error:
            db.session.rollback()
            print(f"Error committing changes: {str(commit_error)}")
            raise commit_error

        # Prepare response based on operation type
        if operation_type == 'edit_file':
            operation_info = {
                "id": process_operation.id,
                "processId": process_id,
                "dataframeOperationId": batch_operation.id,
                "operationType": process_operation.operation_name,
                "title": process_operation.title,
                "description": process_operation.description,
                "sequence": process_operation.sequence,
                "parameters": process_operation.parameters,
                "isActive": process_operation.is_active,
                "createdAt": process_operation.created_at.isoformat() if process_operation.created_at else None,
                "updatedAt": process_operation.updated_at.isoformat() if process_operation.updated_at else None,
                "dataframeOperation": {
                    "id": batch_operation.id,
                    "operationType": "edit_file",
                    "status": batch_operation.status,
                    "payload": batch_operation.payload,
                    "successCount": batch_operation.successful_dataframes,
                    "totalCount": batch_operation.total_dataframes,
                    "createdAt": batch_operation.created_at.isoformat() if batch_operation.created_at else None,
                    "updatedAt": batch_operation.updated_at.isoformat() if batch_operation.updated_at else None
                }
            }
        else:
            operation_info = {
                "id": process_operation.id,
                "processId": process_id,
                "dataframeOperationId": df_operation.id,
                "operationType": process_operation.operation_name,
                "title": process_operation.title,
                "description": process_operation.description,
                "sequence": process_operation.sequence,
                "parameters": process_operation.parameters,
                "isActive": process_operation.is_active,
                "createdAt": process_operation.created_at.isoformat() if process_operation.created_at else None,
                "updatedAt": process_operation.updated_at.isoformat() if process_operation.updated_at else None,
                "dataframeOperation": {
                    "id": df_operation.id,
                    "operationType": df_operation.operation_type,
                    "operationSubtype": df_operation.operation_subtype,
                    "status": df_operation.status,
                    "payload": df_operation.payload,
                    "createdAt": df_operation.created_at.isoformat() if df_operation.created_at else None,
                    "updatedAt": df_operation.updated_at.isoformat() if df_operation.updated_at else None
                }
            }

        return jsonify({
            "success": True,
            "message": "Operation associated with process successfully",
            "operation": operation_info,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "isDraft": process.is_draft,
                "updatedAt": process.updated_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@process_bp.route('/<process_id>/operations', methods=['GET'])
def list_process_operations(process_id):
    """List all operations associated with a process."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get all operations for this process, ordered by sequence
        operations = ProcessOperation.query.filter_by(process_id=process_id)\
            .order_by(ProcessOperation.sequence)\
            .all()

        operation_list = []
        for op in operations:
            # Get the associated DataFrameOperation if it exists
            df_operation = None
            if op.dataframe_operation_id:
                df_operation = DataFrameOperation.query.get(op.dataframe_operation_id)

            operation_info = {
                "id": op.id,
                "processId": process_id,
                "sequence": float(op.sequence),
                "operationType": op.operation_name,
                "title": op.title,
                "description": op.description or (df_operation.message if df_operation else None),  # Use message from df_operation if available
                "parameters": op.parameters,
                "createdAt": op.created_at.isoformat() if op.created_at else None,
                "updatedAt": op.updated_at.isoformat() if op.updated_at else None,
                "dataframeOperation": {
                    "id": df_operation.id,
                    "operationType": df_operation.operation_type,
                    "operationSubtype": df_operation.operation_subtype,
                    "status": df_operation.status,
                    "payload": df_operation.payload,
                    "message": df_operation.message,  # Include message field
                    "createdAt": df_operation.created_at.isoformat() if df_operation.created_at else None,
                    "updatedAt": df_operation.updated_at.isoformat() if df_operation.updated_at else None
                } if df_operation else None
            }
            operation_list.append(operation_info)

        return jsonify({
            "success": True,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            },
            "operations": operation_list,
            "totalCount": len(operation_list)
        })

    except Exception as e:
        print(f"Error in list_process_operations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/operations/reorder', methods=['POST'])
def reorder_process_operations(process_id):
    """Reorder operations within a process based on new sequence numbers."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404
            
        data = request.json
        operation_sequences = data.get('operations', [])
        
        if not operation_sequences:
            return jsonify({"error": "Operation sequences are required"}), 400

        # Validate the operation sequence data structure
        for op_data in operation_sequences:
            if 'id' not in op_data or 'sequence' not in op_data:
                return jsonify({
                    "error": "Each operation must have 'id' and 'sequence' fields"
                }), 400
                
        # Get all existing operations for this process
        existing_operations = {
            op.id: op for op in ProcessOperation.query.filter_by(process_id=process_id).all()
        }

        # Validate all operation IDs belong to this process
        for op_data in operation_sequences:
            if op_data['id'] not in existing_operations:
                return jsonify({
                    "error": f"Operation {op_data['id']} does not belong to this process"
                }), 400
            
        # Update sequences
        for op_data in operation_sequences:
            operation = existing_operations[op_data['id']]
            operation.sequence = float(op_data['sequence'])

        db.session.commit()

        # Fetch updated operations in new sequence order
        updated_operations = ProcessOperation.query.filter_by(process_id=process_id)\
            .order_by(ProcessOperation.sequence)\
            .all()

        operation_list = []
        for op in updated_operations:
            # Get the associated DataFrameOperation if it exists
            df_operation = None
            if op.dataframe_operation_id:
                df_operation = DataFrameOperation.query.get(op.dataframe_operation_id)

            operation_info = {
                "id": op.id,
                "processId": process_id,
                "sequence": float(op.sequence),
                "operationType": op.operation_name,
                "title": op.title,
                "description": op.description or (df_operation.message if df_operation else None),  # Use message from df_operation if available
                "parameters": op.parameters,
                "createdAt": op.created_at.isoformat() if op.created_at else None,
                "updatedAt": op.updated_at.isoformat() if op.updated_at else None,
                "dataframeOperation": {
                    "id": df_operation.id,
                    "operationType": df_operation.operation_type,
                    "operationSubtype": df_operation.operation_subtype,
                    "status": df_operation.status,
                    "payload": df_operation.payload,
                    "message": df_operation.message,  # Include message field
                    "createdAt": df_operation.created_at.isoformat() if df_operation.created_at else None,
                    "updatedAt": df_operation.updated_at.isoformat() if df_operation.updated_at else None
                } if df_operation else None
            }
            operation_list.append(operation_info)
        
        return jsonify({
            "success": True,
            "message": "Operations reordered successfully",
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            },
            "operations": operation_list,
            "totalCount": len(operation_list)
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in reorder_process_operations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/copy', methods=['POST'])
def copy_process(process_id):
    """Create a new process as a copy of an existing process."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get original process and verify ownership
        original_process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not original_process:
            return jsonify({"error": "Original process not found or access denied"}), 404

        data = request.json
        new_process_name = data.get('name')
        if not new_process_name:
            return jsonify({"error": "New process name is required"}), 400

        # Check if process with same name exists
        existing_process = UserProcess.query.filter_by(
            user_id=user.id,
            process_name=new_process_name,
            id = process_id
        ).first()

        if existing_process:
            return jsonify({
                "error": "A process with this name already exists",
                "existingProcess": {
                    "id": existing_process.id,
                    "name": existing_process.process_name,
                    "createdAt": existing_process.created_at.isoformat(),
                    "updatedAt": existing_process.updated_at.isoformat()
                }
            }), 409

        # Create new process
        new_process = UserProcess(
            user_id=user.id,
            process_name=new_process_name,
            file_mappings=original_process.file_mappings,
            file_metadata=original_process.file_metadata,
            is_original=False,
            original_process_id=original_process.id
        )

        # Copy file keys
        for file_key in original_process.file_keys:
            new_file_key = ProcessFileKey(
                key_name=file_key.key_name,
                required_structure=file_key.required_structure
            )
            new_process.file_keys.append(new_file_key)

        db.session.add(new_process)
        db.session.commit()

        # Copy operations
        for operation in original_process.operations:
            if operation.operation_name == 'edit_file':
                # For edit_file operations, check if there's a batch operation ID in parameters
                batch_operation_id = operation.parameters.get('batchOperationId')
                if batch_operation_id:
                    original_batch = DataFrameBatchOperation.query.get(batch_operation_id)
                    if original_batch:
                        new_batch = DataFrameBatchOperation(
                            process_id=new_process.id,
                            payload=copy.deepcopy(original_batch.payload),
                            dataframe_ids=[],
                            operation_ids=[],
                            total_dataframes=original_batch.total_dataframes
                        )
                        db.session.add(new_batch)
                        db.session.commit()

                        # Create new operation with batch operation ID in parameters
                        new_operation = ProcessOperation(
                            process_id=new_process.id,
                            sequence=operation.sequence,
                            operation_name=operation.operation_name,
                            parameters={
                                **copy.deepcopy(operation.parameters),
                                'batchOperationId': new_batch.id
                            },
                            is_active=operation.is_active
                        )
                else:
                    # If no batch operation ID found, just copy the operation as is
                    new_operation = ProcessOperation(
                        process_id=new_process.id,
                        sequence=operation.sequence,
                        operation_name=operation.operation_name,
                        parameters=copy.deepcopy(operation.parameters),
                        is_active=operation.is_active
                    )
            else:
                # For non-edit-file operations
                new_operation = ProcessOperation(
                    process_id=new_process.id,
                    sequence=operation.sequence,
                    operation_name=operation.operation_name,
                    parameters=copy.deepcopy(operation.parameters),
                    dataframe_operation_id=operation.dataframe_operation_id,
                    is_active=operation.is_active
                )
            
            new_process.operations.append(new_operation)

        db.session.commit()

        # Create directory structure in Firebase Storage
        bucket = get_storage_bucket()
        base_path = f"{email}/process/{new_process.id}"
        required_dirs = [
            f"{base_path}/dataframes/",
            f"{base_path}/metadata/",
            f"{base_path}/previews/"
        ]

        # Create empty placeholder files
        for dir_path in required_dirs:
            placeholder = bucket.blob(f"{dir_path}.placeholder")
            placeholder.upload_from_string('')

        return jsonify({
            "success": True,
            "message": "Process copied successfully",
            "process": {
                "id": new_process.id,
                "name": new_process.process_name,
                "userId": user.id,
                "createdAt": new_process.created_at.isoformat(),
                "updatedAt": new_process.updated_at.isoformat(),
                "isOriginal": new_process.is_original,
                "originalProcessId": new_process.original_process_id,
                "fileKeys": {key.key_name: key.required_structure for key in new_process.file_keys},
                "operations": [
                    {
                        "id": op.id,
                        "sequence": float(op.sequence),
                        "operationType": op.operation_name,
                        "parameters": op.parameters,
                        "isActive": op.is_active,
                        "createdAt": op.created_at.isoformat() if op.created_at else None,
                        "updatedAt": op.updated_at.isoformat() if op.updated_at else None
                    } for op in new_process.operations
                ]
            },
            "originalProcess": {
                "id": original_process.id,
                "name": original_process.process_name,
                "createdAt": original_process.created_at.isoformat(),
                "updatedAt": original_process.updated_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@process_bp.route('/<process_id>/operations/<operation_id>/delete', methods=['POST'])
def delete_process_operation(process_id, operation_id):
    """Delete a process operation and resequence remaining operations."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get the operation to delete
        operation = ProcessOperation.query.filter_by(
            id=operation_id,
            process_id=process_id
        ).first()
        if not operation:
            return jsonify({"error": "Operation not found"}), 404
        deleted_operation = {
            "id": operation.id,
            "sequence": float(operation.sequence),
            "operationType": operation.operation_name,
            "title": operation.title,
            "description": operation.description
        }

        # Delete the operation
        db.session.delete(operation)

        # Get remaining operations ordered by sequence
        remaining_operations = ProcessOperation.query.filter_by(process_id=process_id)\
            .order_by(ProcessOperation.sequence)\
            .all()

        # Resequence remaining operations
        for index, op in enumerate(remaining_operations):
            if op.id != operation_id:  # Skip the deleted operation
                op.sequence = float(index + 1)

        db.session.commit()

        # Fetch updated operations for response
        updated_operations = ProcessOperation.query.filter_by(process_id=process_id)\
            .order_by(ProcessOperation.sequence)\
            .all()

        operation_list = []
        for op in updated_operations:
            # Get the associated DataFrameOperation if it exists
            df_operation = None
            if op.dataframe_operation_id:
                df_operation = DataFrameOperation.query.get(op.dataframe_operation_id)

            operation_info = {
                "id": op.id,
                "processId": process_id,
                "sequence": float(op.sequence),
                "operationType": op.operation_name,
                "title": op.title,
                "description": op.description,
                "parameters": op.parameters,
                "createdAt": op.created_at.isoformat() if op.created_at else None,
                "updatedAt": op.updated_at.isoformat() if op.updated_at else None,
                "dataframeOperation": {
                    "id": df_operation.id,
                    "operationType": df_operation.operation_type,
                    "operationSubtype": df_operation.operation_subtype,
                    "status": df_operation.status,
                    "payload": df_operation.payload,
                    "createdAt": df_operation.created_at.isoformat() if df_operation.created_at else None,
                    "updatedAt": df_operation.updated_at.isoformat() if df_operation.updated_at else None
                } if df_operation else None
            }
            operation_list.append(operation_info)
        return jsonify({
            "success": True,
            "message": "Operation deleted and sequences updated successfully",
            "deletedOperation": deleted_operation,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            },
            "operations": operation_list,
            "totalCount": len(operation_list)
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in delete_process_operation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/complete', methods=['POST'])
def complete_process(process_id):
    """Mark a process as completed by setting is_draft to False."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Update draft status
        process.is_draft = False
        process.updated_at = datetime.now()
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Process marked as completed",
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat(),
                "isDraft": process.is_draft,
                "isActive": process.is_active,
                "isOriginal": process.is_original,
                "originalProcessId": process.original_process_id,
                "operationCount": len(process.operations)
            }
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in complete_process: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/drafts', methods=['GET'])
def list_draft_processes():
    """List all draft processes for a user with basic details."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get all draft processes for the user
        draft_processes = UserProcess.query\
            .filter_by(user_id=user.id, is_draft=True, is_active=True)\
            .order_by(UserProcess.updated_at.desc())\
            .all()

        process_list = []
        for process in draft_processes:
            # Count operations
            operation_count = len(process.operations)

            # Get the original process details if this is a copy
            original_process = None
            if process.original_process_id:
                original = UserProcess.query.get(process.original_process_id)
                if original:
                    original_process = {
                        "id": original.id,
                        "name": original.process_name,
                        "createdAt": original.created_at.isoformat()
                    }

            process_info = {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat(),
                "isDraft": True,
                "isActive": process.is_active,
                "isOriginal": process.is_original,
                "originalProcess": original_process,
                "operationCount": operation_count,
                "lastModified": process.updated_at.isoformat(),
                "operations": [
                    {
                        "id": op.id,
                        "sequence": float(op.sequence),
                        "operationType": op.operation_name,
                        "createdAt": op.created_at.isoformat() if op.created_at else None
                    } for op in process.operations
                ]
            }
            process_list.append(process_info)

        return jsonify({
            "success": True,
            "totalCount": len(process_list),
            "processes": process_list
        })

    except Exception as e:
        print(f"Error in list_draft_processes: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/delete', methods=['POST'])
def delete_process(process_id):
    """Mark a process as inactive instead of deleting it."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Store process details for response
        process_info = {
            "id": process.id,
            "name": process.process_name,
            "createdAt": process.created_at.isoformat(),
            "updatedAt": process.updated_at.isoformat(),
            "isDraft": process.is_draft,
            "isOriginal": process.is_original,
            "originalProcessId": process.original_process_id,
            "operationCount": len(process.operations)
        }

        # Start cleanup operations in parallel
        import threading
        import time
        
        storage_cleanup_done = threading.Event()
        storage_error = None
        
        def cleanup_storage():
            nonlocal storage_error
            try:
                bucket = get_storage_bucket()
                base_path = f"{email}/process/{process.id}"
                blobs = list(bucket.list_blobs(prefix=base_path))
                
                # Delete blobs in batches for better performance
                batch_size = 50
                for i in range(0, len(blobs), batch_size):
                    batch = blobs[i:i + batch_size]
                    for blob in batch:
                        try:
                            blob.delete()
                        except Exception as e:
                            print(f"Warning: Failed to delete blob {blob.name}: {str(e)}")
            except Exception as e:
                storage_error = str(e)
                print(f"Warning: Error cleaning up storage for process {process_id}: {str(e)}")
            finally:
                storage_cleanup_done.set()
        
        # Start storage cleanup in background
        storage_thread = threading.Thread(target=cleanup_storage)
        storage_thread.start()
        
        # Optimize database cleanup with bulk operations
        try:
            # Use bulk delete for better performance
            DataFrameOperation.query.filter_by(process_id=process.id).delete(synchronize_session=False)
            DataFrameBatchOperation.query.filter_by(process_id=process.id).delete(synchronize_session=False)
            FormattingStep.query.filter_by(process_id=process.id).delete(synchronize_session=False)
            
            # Delete the process (cascade will handle remaining children)
            db.session.delete(process)
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            print(f"Error in database cleanup: {str(e)}")
            raise e
        
        # Wait for storage cleanup to complete (with timeout)
        storage_cleanup_done.wait(timeout=30)  # 30 second timeout
        if not storage_cleanup_done.is_set():
            print(f"Warning: Storage cleanup timed out for process {process_id}")
        
        if storage_error:
            print(f"Warning: Storage cleanup had errors: {storage_error}")

        return jsonify({
            "success": True,
            "message": "Process deleted successfully",
            "deletedProcess": process_info
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in delete_process: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/delete-async', methods=['POST'])
def delete_process_async(process_id):
    """Delete a process asynchronously for very large processes."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Start async deletion in background
        import threading
        def async_delete():
            try:
                # Mark process as being deleted
                process.is_active = False
                db.session.commit()
                
                # Perform cleanup
                bucket = get_storage_bucket()
                base_path = f"{email}/process/{process.id}"
                blobs = list(bucket.list_blobs(prefix=base_path))
                
                # Delete in smaller batches with delays
                batch_size = 20
                for i in range(0, len(blobs), batch_size):
                    batch = blobs[i:i + batch_size]
                    for blob in batch:
                        try:
                            blob.delete()
                        except Exception as e:
                            print(f"Warning: Failed to delete blob {blob.name}: {str(e)}")
                    time.sleep(0.1)  # Small delay between batches
                
                # Clean up database
                DataFrameOperation.query.filter_by(process_id=process.id).delete(synchronize_session=False)
                DataFrameBatchOperation.query.filter_by(process_id=process.id).delete(synchronize_session=False)
                FormattingStep.query.filter_by(process_id=process.id).delete(synchronize_session=False)
                db.session.delete(process)
                db.session.commit()
                
                print(f"Async deletion completed for process {process_id}")
            except Exception as e:
                print(f"Error in async deletion for process {process_id}: {str(e)}")
                db.session.rollback()
        
        # Start background thread
        thread = threading.Thread(target=async_delete)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Process deletion started in background",
            "processId": process_id
        })

    except Exception as e:
        print(f"Error starting async delete: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/operations/<operation_id>', methods=['GET'])
def get_process_operation_details(process_id, operation_id):
    """Get detailed information about a specific operation within a process."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get the specific operation
        operation = ProcessOperation.query.filter_by(
            id=operation_id,
            process_id=process_id
        ).first()
        if not operation:
            return jsonify({"error": "Operation not found"}), 404

        # Get associated DataFrameOperation if it exists
        df_operation = None
        batch_operation = None
        if operation.dataframe_operation_id:
            df_operation = DataFrameOperation.query.get(operation.dataframe_operation_id)
        elif operation.operation_name == 'edit_file' and operation.parameters.get('batchOperationId'):
            batch_operation = DataFrameBatchOperation.query.get(
                operation.parameters.get('batchOperationId')
            )

        # Build detailed response based on operation type
        operation_details = {
            "id": operation.id,
            "processId": process_id,
            "sequence": float(operation.sequence),
            "operationType": operation.operation_name,
            "title": operation.title,
            "description": operation.description,
            "parameters": json.loads(operation.parameters) if isinstance(operation.parameters, str) else operation.parameters,
            "createdAt": operation.created_at.isoformat() if operation.created_at else None,
            "updatedAt": operation.updated_at.isoformat() if operation.updated_at else None,
        }

        # Add operation-type specific details
        if operation.operation_name == 'edit_file' and batch_operation:
            operation_details["batchOperation"] = {
                "id": batch_operation.id,
                "status": batch_operation.status,
                "successCount": batch_operation.successful_dataframes,
                "totalCount": batch_operation.total_dataframes,
                "payload": json.loads(batch_operation.payload) if isinstance(batch_operation.payload, str) else batch_operation.payload,
                "createdAt": batch_operation.created_at.isoformat() if batch_operation.created_at else None,
                "updatedAt": batch_operation.updated_at.isoformat() if batch_operation.updated_at else None
            }
        elif df_operation:
            operation_details["dataframeOperation"] = {
                "id": df_operation.id,
                "operationType": df_operation.operation_type,
                "operationSubtype": df_operation.operation_subtype,
                "status": df_operation.status,
                "payload": json.loads(df_operation.payload) if isinstance(df_operation.payload, str) else df_operation.payload,
                "createdAt": df_operation.created_at.isoformat() if df_operation.created_at else None,
                "updatedAt": df_operation.updated_at.isoformat() if df_operation.updated_at else None
            }

        # Add selected parameters based on operation type
        params = operation_details["parameters"]
        
        if operation.operation_name == 'edit_file':
            operation_details["selectedData"] = {
                "tables": [{
                    "tableName": table.get('tableName'),
                    "columnSelections": table.get('columnSelections', []),
                    "columnTypes": table.get('columnTypes', {}),
                    "datetimeFormats": table.get('datetimeFormats', {})
                } for table in params.get('tables', [])]
            }
        elif operation.operation_name == 'add_column':
            operation_details["selectedData"] = {
                "newColumnName": params.get('newColumnName'),
                "operationType": params.get('type'),
                "sourceColumn": params.get('sourceColumn'),
                "operations": params.get('operations', []),
                "pattern": params.get('pattern'),
                "residualValue": params.get('residualValue')
            }
        elif operation.operation_name == 'merge_files':
            operation_details["selectedData"] = {
                "files": params.get('files', []),
                "keyPairs": params.get('keyPairs', []),
                "method": params.get('method', 'inner'),
                "showCountSummary": params.get('showCountSummary', False)
            }
        elif operation.operation_name == 'reconcile_files':
            operation_details["selectedData"] = {
                "sourceTableNames": params.get('sourceTableNames', []),
                "outputTableName": params.get('outputTableName'),
                "keys": params.get('keys', []),
                "values": params.get('values', []),
                "crossReference": params.get('crossReference', {})
            }
        elif operation.operation_name == 'sort_filter':
            operation_details["selectedData"] = {
                "fileName": params.get('fileName'),
                "sheet": params.get('sheet'),
                "sortConfig": params.get('sortConfig', []),
                "filterConfig": params.get('filterConfig', []),
                "newFileName": params.get('newFileName')
            }
        elif operation.operation_name == 'formatting':
            operation_details["selectedData"] = {
                "fileName": params.get('fileName'),
                "sheet": params.get('sheet'),
                "formattingConfig": params.get('formattingConfig', {}),
                "newFileName": params.get('newFileName')
            }
        elif operation.operation_name == 'group_pivot':
            operation_details["selectedData"] = {
                "fileName": params.get('fileName'),
                "sheet": params.get('sheet'),
                "rowIndex": params.get('rowIndex', []),
                "columnIndex": params.get('columnIndex'),
                "pivotValues": params.get('pivotValues', []),
                "newFileName": params.get('newFileName')
            }
        elif operation.operation_name == 'file_operations':
            operation_details["selectedData"] = {
                "fileName": params.get('fileName'),
                "sheet": params.get('sheet'),
                "operation": params.get('operation'),
                "data": params.get('data', {}),
                "newFileName": params.get('newFileName'),
                "replace_existing": params.get('replace_existing', False)
            }
        elif operation.operation_name == 'regex':
            operation_details["selectedData"] = {
                "pattern": params.get('pattern'),
                "explanation": params.get('explanation'),
                "tokens_used": params.get('tokens_used'),
                "confidence": params.get('confidence')
            }

        return jsonify({
            "success": True,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "createdAt": process.created_at.isoformat(),
                "updatedAt": process.updated_at.isoformat()
            },
            "operation": operation_details
        })

    except Exception as e:
        print(f"Error in get_process_operation_details: {str(e)}")
        return jsonify({"error": str(e)}), 500

@process_bp.route('/<process_id>/operations/replace', methods=['POST'])
def manage_process_operations(process_id):
    """
    Combined endpoint: Add a new operation and optionally delete an existing one in a process.
    Request can include:
        - 'dataframeOperationId', 'sequence', 'operationType' (required for add)
        - 'operationToDeleteId' (optional - to delete before adding)
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Authenticate user and check process ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        process.is_draft = True
        process.updated_at = datetime.now()
        db.session.add(process)

        deleted_operation_info = None
        operation_to_delete_id = data.get('operationToDeleteId')

        # === Deletion part (optional) ===
        if operation_to_delete_id:
            operation = ProcessOperation.query.filter_by(
                id=operation_to_delete_id,
                process_id=process_id
            ).first()
            if not operation:
                return jsonify({"error": "Operation to delete not found"}), 404

            deleted_operation_info = {
                "id": operation.id,
                "sequence": float(operation.sequence),
                "operationType": operation.operation_name,
                "title": operation.title,
                "description": operation.description
            }

            db.session.delete(operation)
            db.session.commit()

        # === Addition part ===
        required_fields = ['dataframeOperationId', 'sequence']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        operation_type = data.get('operationType')

        allDataframes = DataFrame.query.filter_by(
            process_id=process_id,
            is_active=True,
            is_originally_uploaded=True).all()
        
        tableNames = []
        fileNames = []
        for dataframe in allDataframes:
            metadata = dataframe.data_metadata
            tableNames.append(metadata.get('tableName'))
            fileNames.append(metadata.get('originalFileName'))

        description = f'Select files {", ".join(tableNames)} from files {", ".join(fileNames)}'

        if operation_type == 'edit_file':
            batch_operation = DataFrameBatchOperation.query.get(data['dataframeOperationId'])
            if not batch_operation or batch_operation.process_id != process_id:
                return jsonify({"error": "BatchOperation not found or does not belong to this process"}), 404

            process_operation = ProcessOperation(
                process_id=process_id,
                title = deleted_operation_info.get('title') or 'Define Inputs',
                description = description or '',
                sequence=float(data['sequence']),
                operation_name='edit_file',
                parameters={**batch_operation.payload, 'batchOperationId': batch_operation.id},
                is_active=True
            )
        else:
            df_operation = DataFrameOperation.query.get(data['dataframeOperationId'])
            if not df_operation or df_operation.process_id != process_id:
                return jsonify({"error": "DataFrameOperation not found or does not belong to this process"}), 404
            
            operation_title = None  # Default to provided title
            if not operation_title:
                if df_operation.operation_type == "add_column":
                    subtype = df_operation.operation_subtype.lower()
                    if "calcul" in subtype:
                        operation_title = "Add Column – Calculate"
                    elif "concat" in subtype:
                        operation_title = "Add Column – Concatenate"
                    elif "conditional" in subtype:
                        operation_title = "Add Column – Conditional"
                    elif "pattern" in subtype:
                        operation_title = "Add Column – Pattern"
                elif df_operation.operation_type == "merge_files":
                    operation_title = "Merge"
                elif df_operation.operation_type == "group_pivot":
                    operation_title = "Pivot"
                elif df_operation.operation_type == "sort_filter":
                    payload_str = str(df_operation.payload).lower()
                    if "sortconfig" in payload_str:
                        operation_title = "Sort"
                    elif "filterconfig" in payload_str:
                        operation_title = "Filter"
                elif df_operation.operation_type == "replace_rename_reorder":
                    operation_title = "Replace Rename Reorder"
                elif df_operation.operation_type == "reconcile_files":
                    operation_title = "Reconcile"
                elif df_operation.operation_type == "apply_formatting":
                    operation_title = "Formatting"

            process_operation = ProcessOperation(
                process_id=process_id,
                title = deleted_operation_info.get('title') or operation_title,
                description = deleted_operation_info.get('description') or '',
                sequence=float(data['sequence']),
                operation_name=df_operation.operation_type,
                parameters=df_operation.payload,
                dataframe_operation_id=df_operation.id,
                is_active=True
            )

        db.session.add(process_operation)
        
        # Commit both process update and new operation in one transaction
        try:
            db.session.commit()
        except Exception as commit_error:
            db.session.rollback()
            print(f"Error committing changes: {str(commit_error)}")
            raise commit_error

        # Format response for the newly added operation
        op_source = batch_operation if operation_type == 'edit_file' else df_operation
        op_info = {
            "id": process_operation.id,
            "processId": process_id,
            "sequence": process_operation.sequence,
            "operationType": process_operation.operation_name,
            "parameters": process_operation.parameters,
            "isActive": process_operation.is_active,
            "createdAt": process_operation.created_at.isoformat() if process_operation.created_at else None,
            "updatedAt": process_operation.updated_at.isoformat() if process_operation.updated_at else None,
            "dataframeOperation": {
                "id": op_source.id,
                "operationType": "edit_file" if operation_type == 'edit_file' else op_source.operation_type,
                "operationSubtype": None if operation_type == 'edit_file' else op_source.operation_subtype,
                "status": op_source.status,
                "payload": op_source.payload,
                "createdAt": op_source.created_at.isoformat() if op_source.created_at else None,
                "updatedAt": op_source.updated_at.isoformat() if op_source.updated_at else None
            }
        }

        return jsonify({
            "success": True,
            "message": "Operation added successfully" + (" and previous operation deleted" if operation_to_delete_id else ""),
            "operation": op_info,
            "deletedOperation": deleted_operation_info,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "updatedAt": process.updated_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@process_bp.route('/<process_id>/operations/replace', methods=['POST'])
def manage_process_operations(process_id):
    """
    Combined endpoint: Add a new operation and optionally delete an existing one in a process.
    Request can include:
        - 'dataframeOperationId', 'sequence', 'operationType' (required for add)
        - 'operationToDeleteId' (optional - to delete before adding)
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Authenticate user and check process ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        process.is_draft = True
        process.updated_at = datetime.now()
        db.session.add(process)

        deleted_operation_info = None
        operation_to_delete_id = data.get('operationToDeleteId')

        # === Deletion part (optional) ===
        if operation_to_delete_id:
            operation = ProcessOperation.query.filter_by(
                id=operation_to_delete_id,
                process_id=process_id
            ).first()
            if not operation:
                return jsonify({"error": "Operation to delete not found"}), 404

            deleted_operation_info = {
                "id": operation.id,
                "sequence": float(operation.sequence),
                "operationType": operation.operation_name,
                "title": operation.title,
                "description": operation.description
            }

            db.session.delete(operation)
            db.session.commit()

        # === Addition part ===
        required_fields = ['dataframeOperationId', 'sequence']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        operation_type = data.get('operationType')

        allDataframes = DataFrame.query.filter_by(
            process_id=process_id,
            is_active=True,
            is_originally_uploaded=True).all()
        
        tableNames = []
        fileNames = []
        for dataframe in allDataframes:
            metadata = dataframe.data_metadata
            tableNames.append(metadata.get('tableName'))
            fileNames.append(metadata.get('originalFileName'))

        description = f'Select files {", ".join(tableNames)} from files {", ".join(fileNames)}'

        if operation_type == 'edit_file':
            allDrafts = DataFrame.query.filter_by(
                process_id=process_id,
                is_originally_uploaded=False,
                is_draft=True,
            ).all()

            tables_payload = []

            for draft in allDrafts:
                metadata = draft.data_metadata or {}
                columns = metadata.get("columns", metadata.get("columnTypes", {}).keys())

                column_selections = {col: True for col in columns}
                column_types = {col: metadata.get("columnTypes", {}).get(col, "string") for col in columns}
                datetime_formats = {col: "" for col in columns}

                tables_payload.append({
                    "tableName": draft.name,
                    "columnSelections": column_selections,
                    "columnTypes": column_types,
                    "datetimeFormats": datetime_formats
                })

            payload = {
                "processId": process_id,
                "tables": tables_payload
            }

            process_operation = ProcessOperation(
                process_id=process_id,
                title = deleted_operation_info.get('title') or 'Define Inputs',
                description = description or '',
                sequence=float(data['sequence']),
                operation_name='edit_file',
                parameters={**payload, 'batchOperationId': None},
                is_active=True
            )
        else:
            df_operation = DataFrameOperation.query.get(data['dataframeOperationId'])
            if not df_operation or df_operation.process_id != process_id:
                return jsonify({"error": "DataFrameOperation not found or does not belong to this process"}), 404
            
            operation_title = None  # Default to provided title
            if not operation_title:
                if df_operation.operation_type == "add_column":
                    subtype = df_operation.operation_subtype.lower()
                    if "calcul" in subtype:
                        operation_title = "Add Column – Calculate"
                    elif "concat" in subtype:
                        operation_title = "Add Column – Concatenate"
                    elif "conditional" in subtype:
                        operation_title = "Add Column – Conditional"
                    elif "pattern" in subtype:
                        operation_title = "Add Column – Pattern"
                elif df_operation.operation_type == "merge_files":
                    operation_title = "Merge"
                elif df_operation.operation_type == "group_pivot":
                    operation_title = "Pivot"
                elif df_operation.operation_type == "sort_filter":
                    payload_str = str(df_operation.payload).lower()
                    if "sortconfig" in payload_str:
                        operation_title = "Sort"
                    elif "filterconfig" in payload_str:
                        operation_title = "Filter"
                elif df_operation.operation_type == "replace_rename_reorder":
                    operation_title = "Replace Rename Reorder"
                elif df_operation.operation_type == "reconcile_files":
                    operation_title = "Reconcile"
                elif df_operation.operation_type == "apply_formatting":
                    operation_title = "Formatting"

            process_operation = ProcessOperation(
                process_id=process_id,
                title = deleted_operation_info.get('title') or operation_title,
                description = deleted_operation_info.get('description') or '',
                sequence=float(data['sequence']),
                operation_name=df_operation.operation_type,
                parameters=df_operation.payload,
                dataframe_operation_id=df_operation.id,
                is_active=True
            )

        db.session.add(process_operation)
        
        # Commit both process update and new operation in one transaction
        try:
            db.session.commit()
        except Exception as commit_error:
            db.session.rollback()
            print(f"Error committing changes: {str(commit_error)}")
            raise commit_error

        # Format response for the newly added operation
        op_source = batch_operation if operation_type == 'edit_file' else df_operation
        op_info = {
            "id": process_operation.id,
            "processId": process_id,
            "sequence": process_operation.sequence,
            "operationType": process_operation.operation_name,
            "parameters": process_operation.parameters,
            "isActive": process_operation.is_active,
            "createdAt": process_operation.created_at.isoformat() if process_operation.created_at else None,
            "updatedAt": process_operation.updated_at.isoformat() if process_operation.updated_at else None,
            "dataframeOperation": {
                "id": op_source.id,
                "operationType": "edit_file" if operation_type == 'edit_file' else op_source.operation_type,
                "operationSubtype": None if operation_type == 'edit_file' else op_source.operation_subtype,
                "status": op_source.status,
                "payload": op_source.payload,
                "createdAt": op_source.created_at.isoformat() if op_source.created_at else None,
                "updatedAt": op_source.updated_at.isoformat() if op_source.updated_at else None
            }
        }

        return jsonify({
            "success": True,
            "message": "Operation added successfully" + (" and previous operation deleted" if operation_to_delete_id else ""),
            "operation": op_info,
            "deletedOperation": deleted_operation_info,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "updatedAt": process.updated_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@process_bp.route('/<process_id>/operations/replace1', methods=['POST'])
def manage_process_operations1(process_id):
    """
    Combined endpoint: Add a new operation, optionally delete an existing one,
    and finalize table drafts into the process.
    """
    try:
        data = request.json
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Authenticate user and check process ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        process.is_draft = True  # process stays draft until manually finalized
        process.updated_at = datetime.utcnow()
        db.session.add(process)

        deleted_operation_info = None
        operation_to_delete_id = data.get('operationToDeleteId')

        # === Optional deletion of previous operation ===
        if operation_to_delete_id:
            operation = ProcessOperation.query.filter_by(
                id=operation_to_delete_id,
                process_id=process_id
            ).first()
            if not operation:
                return jsonify({"error": "Operation to delete not found"}), 404

            deleted_operation_info = {
                "id": operation.id,
                "sequence": float(operation.sequence),
                "operationType": operation.operation_name,
                "title": operation.title,
                "description": operation.description
            }

            db.session.delete(operation)
            db.session.commit()

        # === Validate required fields for adding a new operation ===
        required_fields = ['dataframeOperationId', 'sequence']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        operation_type = data.get('operationType')

        # === Promote table drafts to finalized copies ===
        table_drafts = DataFrame.query.filter_by(
            process_id=process_id,
            is_draft=True,
            is_originally_uploaded=False
        ).all()

        finalized_table_ids = []

        for draft in table_drafts:
            # Check for existing finalized copy
            finalized = DataFrame.query.filter_by(
                process_id=process_id,
                name=draft.name,
                is_draft=False,
                is_originally_uploaded=False
            ).first()

            if finalized:
                # Overwrite finalized copy
                finalized.row_count = draft.row_count
                finalized.column_count = draft.column_count
                finalized.data_metadata = draft.data_metadata
                finalized.storage_path = draft.storage_path
                finalized.updated_at = datetime.utcnow()
                finalized_table_ids.append(finalized.id)
            else:
                # Promote draft to finalized
                draft.is_draft = False
                draft.updated_at = datetime.utcnow()
                db.session.add(draft)
                finalized_table_ids.append(draft.id)

        # Delete all drafts after finalization
        for draft in table_drafts:
            if draft.is_draft:  # leftover drafts
                db.session.delete(draft)

        db.session.commit()  # commit all draft promotions before creating ProcessOperation

        # === Add ProcessOperation ===
        if operation_type == 'edit_file':
            batch_operation = DataFrameBatchOperation.query.get(data['dataframeOperationId'])
            if not batch_operation or batch_operation.process_id != process_id:
                return jsonify({"error": "BatchOperation not found or does not belong to this process"}), 404

            description = f'Select tables {", ".join([d.name for d in table_drafts])}'

            process_operation = ProcessOperation(
                process_id=process_id,
                title = deleted_operation_info.get('title') or 'Define Inputs',
                description = description or '',
                sequence=float(data['sequence']),
                operation_name='edit_file',
                parameters={**batch_operation.payload, 'batchOperationId': batch_operation.id, 'finalizedTableIds': finalized_table_ids},
                is_active=True
            )
        else:
            df_operation = DataFrameOperation.query.get(data['dataframeOperationId'])
            if not df_operation or df_operation.process_id != process_id:
                return jsonify({"error": "DataFrameOperation not found or does not belong to this process"}), 404

            # Build default title based on operation type/subtype
            operation_title = None
            if df_operation.operation_type == "add_column":
                subtype = (df_operation.operation_subtype or "").lower()
                if "calcul" in subtype:
                    operation_title = "Add Column – Calculate"
                elif "concat" in subtype:
                    operation_title = "Add Column – Concatenate"
                elif "conditional" in subtype:
                    operation_title = "Add Column – Conditional"
                elif "pattern" in subtype:
                    operation_title = "Add Column – Pattern"
            elif df_operation.operation_type == "merge_files":
                operation_title = "Merge"
            elif df_operation.operation_type == "group_pivot":
                operation_title = "Pivot"
            elif df_operation.operation_type == "sort_filter":
                payload_str = str(df_operation.payload).lower()
                if "sortconfig" in payload_str:
                    operation_title = "Sort"
                elif "filterconfig" in payload_str:
                    operation_title = "Filter"
            elif df_operation.operation_type == "replace_rename_reorder":
                operation_title = "Replace Rename Reorder"
            elif df_operation.operation_type == "reconcile_files":
                operation_title = "Reconcile"
            elif df_operation.operation_type == "apply_formatting":
                operation_title = "Formatting"

            process_operation = ProcessOperation(
                process_id=process_id,
                title = deleted_operation_info.get('title') or operation_title,
                description = deleted_operation_info.get('description') or '',
                sequence=float(data['sequence']),
                operation_name=df_operation.operation_type,
                parameters={**df_operation.payload, 'finalizedTableIds': finalized_table_ids},
                dataframe_operation_id=df_operation.id,
                is_active=True
            )

        db.session.add(process_operation)
        db.session.commit()

        # Prepare response
        op_source = batch_operation if operation_type == 'edit_file' else df_operation
        op_info = {
            "id": process_operation.id,
            "processId": process_id,
            "sequence": process_operation.sequence,
            "operationType": process_operation.operation_name,
            "parameters": process_operation.parameters,
            "isActive": process_operation.is_active,
            "createdAt": process_operation.created_at.isoformat() if process_operation.created_at else None,
            "updatedAt": process_operation.updated_at.isoformat() if process_operation.updated_at else None,
            "dataframeOperation": {
                "id": op_source.id,
                "operationType": "edit_file" if operation_type == 'edit_file' else op_source.operation_type,
                "operationSubtype": None if operation_type == 'edit_file' else op_source.operation_subtype,
                "status": getattr(op_source, 'status', None),
                "payload": op_source.payload,
                "createdAt": op_source.created_at.isoformat() if op_source.created_at else None,
                "updatedAt": op_source.updated_at.isoformat() if op_source.updated_at else None
            }
        }

        return jsonify({
            "success": True,
            "message": "Operation added successfully" + (" and previous operation deleted" if operation_to_delete_id else ""),
            "operation": op_info,
            "deletedOperation": deleted_operation_info,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "updatedAt": process.updated_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def iso_or_none(dt):
    return dt.isoformat() if dt else None

def build_description_for_process(process_id):
    """
    Build a human-friendly description using all uploaded dataframes in the process.
    """
    all_dfs = DataFrame.query.filter_by(
        process_id=process_id,
        is_active=True,
        is_originally_uploaded=True
    ).all()

    table_names = []
    file_names = []
    for df in all_dfs:
        md = df.data_metadata or {}
        table_names.append(md.get("tableName") or df.name or "unknown")
        file_names.append(md.get("originalFileName") or getattr(df, "original_file_name", "unknown"))

    return f"Select files {', '.join(table_names)} from files {', '.join(file_names)}" if table_names else ""