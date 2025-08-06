from flask import Blueprint, request, jsonify
from flask import Blueprint, request, jsonify
from models import DataFrame, UserProcess, DataFrameOperation, OperationType, db, FormattingStep, AIRequest, User
from llm_chain import run_chain
import json
from datetime import datetime, timezone
from add_column import process_add_column
from sort_filter import process_sort_filter_data
from group_pivot import process_pivot_table
from merge_files import process_merge_tables, process_dataframe_reconciliation
from file_operations import process_dataframe_operations
from formatting import process_dataframe_formatting
from firebase_config import get_storage_bucket
from io import BytesIO

nlp_bp = Blueprint('nlp', __name__, url_prefix='/api/nlp')

def transform_metadata(metadata):
    """Transform metadata to match DataFrameMetadata model requirements"""
    # Extract column names from the column objects
    column_names = []
    column_types = {}
    
    # Handle the columns array which contains name/type objects
    for col in metadata.get('columns', []):
        if isinstance(col, dict) and 'name' in col and 'type' in col:
            col_name = col['name']
            col_type = col['type']
            column_names.append(col_name)
            
            # Map column types to pandas-compatible types
            if col_type in ['float', 'number', 'numeric']:
                column_types[col_name] = 'float64'
            elif col_type in ['int', 'integer']:
                column_types[col_name] = 'int64'
            elif col_type in ['datetime', 'date']:
                column_types[col_name] = 'datetime64[ns]'
            else:
                column_types[col_name] = 'object'

    # Create summary statistics
    summary = {
        'nullCounts': {col: 0 for col in column_names},  # Default to 0 if not available
        'uniqueCounts': {col: 0 for col in column_names}  # Default to 0 if not available
    }

    # If we have summary data in the metadata, use it
    if 'summary' in metadata:
        if 'nullCounts' in metadata['summary']:
            summary['nullCounts'].update(metadata['summary']['nullCounts'])
        if 'uniqueCounts' in metadata['summary']:
            summary['uniqueCounts'].update(metadata['summary']['uniqueCounts'])

    # Create the DataFrameMetadata structure
    return {
        'columns': column_names,  # List of strings
        'summary': summary,  # DataFrameSummary object
        'columnTypes': column_types  # Dict[str, str]
    }

def convert_table_names_to_fields(data):
    """Convert table_names array into appropriate field names based on operation type"""
    if 'table_names' not in data:
        return data
        
    table_names = data['table_names']
    operation_type = data.get('operation_type')
    
    if operation_type in ['merge-files', 'merge_files']:
        if len(table_names) >= 2:
            data['table1Name'] = table_names[0]
            data['table2Name'] = table_names[1]
    elif operation_type == 'reconcile':
        data['sourceTableNames'] = table_names
    else:
        if len(table_names) >= 1:
            data['tableName'] = table_names[0]
            
    return data

@nlp_bp.route("/process", methods=["POST"])
def process_natural_language():
    """Process natural language query with DataFrame context"""
    # Initialize variables at the top level
    ai_request = None
    result = None
    
    try:
        print("[DEBUG] Starting process_natural_language endpoint")
        data = request.json
        email = request.headers.get("X-User-Email")
        print(f"[DEBUG] Received request - Email: {email}, Data: {json.dumps(data, indent=2)}")

        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Convert table_names array to appropriate field names
        data = convert_table_names_to_fields(data)
        print(f"[DEBUG] Converted data: {json.dumps(data, indent=2)}")

        # Handle AI request tracking and retries
        original_request_id = data.get("aiRequestId")
        try:
            # Get user and verify ownership
            user = User.query.filter_by(email=email).first()
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Get process and verify ownership
            process = UserProcess.query.filter_by(id=data.get("process_id"), user_id=user.id).first()
            if not process:
                return jsonify({"error": "Process not found or access denied"}), 404

            if original_request_id:
                # Try to find existing request and verify ownership
                ai_request = db.session.query(AIRequest).filter(
                    AIRequest.id == original_request_id,
                    AIRequest.email == email,
                    AIRequest.process_id == data.get("process_id")
                ).first()
                
                if ai_request:
                    # Update existing request for retry
                    if ai_request.retry_count >= ai_request.max_retries:
                        raise ValueError(f"Maximum retry limit ({ai_request.max_retries}) reached for request {original_request_id}")
                    
                    # Remove status check to allow retries regardless of previous status
                    # Update retry count and timestamps
                    ai_request.retry_count += 1
                    ai_request.last_retry_time = datetime.now(timezone.utc)
                    ai_request.request_time = datetime.now(timezone.utc)
                    ai_request.status = 'in_progress'
                    ai_request.response = None
                    ai_request.error_message = None
                    ai_request.response_time = None
                    ai_request.processing_duration = None
                    print(f"[DEBUG] Updated request {ai_request.id} for retry, count: {ai_request.retry_count}")
                else:
                    # Create new request with the provided ID
                    ai_request = AIRequest(
                        process_id=data.get("process_id"),
                        user_id=user.id,
                        email=email,
                        operation_type=data.get('operation_type'),
                        query=data.get('query'),
                        table_name=data.get('tableName'),
                        second_table_name=data.get('table2Name'),
                        original_request_id=original_request_id,
                        custom_id=original_request_id
                    )
                    print(f"[DEBUG] Created new AI request with provided ID {ai_request.id}")
            else:
                # Create new AI request
                ai_request = AIRequest(
                    process_id=data.get("process_id"),
                    user_id=user.id,
                    email=email,
                    operation_type=data.get('operation_type'),
                    query=data.get('query'),
                    table_name=data.get('tableName'),
                    second_table_name=data.get('table2Name')
                )
                print(f"[DEBUG] Created new AI request {ai_request.id}")
            
            db.session.add(ai_request)
            db.session.commit()
        except ValueError as e:
            print(f"[DEBUG] Retry handling failed: {str(e)}")
            # Get the original request to include its details in the response
            original_request = db.session.query(AIRequest).get(original_request_id)
            if original_request:
                return jsonify({
                    "error": str(e),
                    "details": {
                        "originalRequestId": original_request_id,
                        "retryCount": original_request.retry_count,
                        "maxRetries": original_request.max_retries,
                        "lastRetryTime": original_request.last_retry_time.isoformat() if original_request.last_retry_time else None,
                        "status": original_request.status,
                        "errorMessage": original_request.error_message
                    }
                }), 429  # Too Many Requests
            else:
                return jsonify({"error": str(e)}), 400
        except Exception as e:
            print(f"[DEBUG] Unexpected error in AI request handling: {str(e)}")
            db.session.rollback()
            return jsonify({"error": f"Failed to handle AI request: {str(e)}"}), 500

        # Ensure ai_request is available for the rest of the function
        if not ai_request:
            return jsonify({"error": "Failed to initialize AI request"}), 500

        # Validate required fields based on operation type
        if data.get("operation_type") in ["merge-files", "merge_files"]:
            required_fields = ["query", "operation_type", "process_id", "table1Name", "table2Name", "output_table_name"]
        elif data.get("operation_type") == "reconcile":
            required_fields = ["query", "operation_type", "process_id", "sourceTableNames", "output_table_name"]
        else:
            required_fields = ["query", "operation_type", "tableName", "process_id"]

        if not all(k in data for k in required_fields):
            print(f"[DEBUG] Missing fields. Received: {list(data.keys())}")
            return jsonify({
                "error": f"Missing required fields. Required: {required_fields}"
            }), 400

        # Get optional output table name
        output_table_name = data.get("output_table_name")
        print(f"[DEBUG] Output table name: {output_table_name}")

        # Get user process and verify ownership
        print(f"[DEBUG] Fetching process with ID: {data['process_id']}")
        process = UserProcess.query.filter_by(
            id=data["process_id"]
        ).first()
        if not process:
            print(f"[DEBUG] Process not found: {data['process_id']}")
            return jsonify({"error": f"Process with id {data['process_id']} not found"}), 404
            
        # Initialize storage bucket
        bucket = get_storage_bucket()

        # Handle DataFrame fetching based on operation type
        if data["operation_type"] in ["merge"]:
            # Get first DataFrame
            print(f"[DEBUG] Fetching first DataFrame - Name: {data['table1Name']}")
            dataframe = DataFrame.query.filter_by(
                name=data["table1Name"],
                process_id=data["process_id"]
            ).first()
            if not dataframe:
                print(f"[DEBUG] First table not found")
                return jsonify({
                    "error": f"DataFrame {data['table1Name']} not found in process {data['process_id']}"
                }), 404

            # Get second DataFrame
            print(f"[DEBUG] Fetching second DataFrame - Name: {data['table2Name']}")
            dataframe2 = DataFrame.query.filter_by(
                name=data["table2Name"],
                process_id=data["process_id"]
            ).first()
            if not dataframe2:
                print(f"[DEBUG] Second table not found")
                return jsonify({
                    "error": f"DataFrame {data['table2Name']} not found in process {data['process_id']}"
                }), 404

            # Get metadata from storage files
            try:
                # Get first table metadata from storage
                metadata_path1 = f"{email}/process/{data['process_id']}/metadata/{data['table1Name']}.json"
                metadata_blob1 = bucket.blob(metadata_path1)
                
                if not metadata_blob1.exists():
                    print(f"[DEBUG] Metadata for first table not found: {metadata_path1}")
                    return jsonify({"error": f"Metadata for table {data['table1Name']} not found"}), 404
                    
                metadata = json.loads(metadata_blob1.download_as_string())
                
                # Ensure required metadata fields exist for first table
                if 'columns' not in metadata:
                    metadata['columns'] = []
                    print(f"[DEBUG] Warning: 'columns' field missing in first table metadata, using empty list")
                
                # Create columnTypes if missing for first table
                if 'columnTypes' not in metadata:
                    print(f"[DEBUG] Warning: 'columnTypes' field missing in first table metadata, inferring from columns")
                    metadata['columnTypes'] = {}
                    for col in metadata.get('columns', []):
                        if isinstance(col, dict) and 'name' in col and 'type' in col:
                            col_name = col['name']
                            col_type = col['type']
                            # Map column types to pandas-compatible types
                            if col_type in ['float', 'number', 'numeric']:
                                metadata['columnTypes'][col_name] = 'float64'
                            elif col_type in ['int', 'integer']:
                                metadata['columnTypes'][col_name] = 'int64'
                            elif col_type in ['datetime', 'date']:
                                metadata['columnTypes'][col_name] = 'datetime64[ns]'
                            else:
                                metadata['columnTypes'][col_name] = 'object'
                
                # Get second table metadata from storage
                metadata_path2 = f"{email}/process/{data['process_id']}/metadata/{data['table2Name']}.json"
                metadata_blob2 = bucket.blob(metadata_path2)
                
                if not metadata_blob2.exists():
                    print(f"[DEBUG] Metadata for second table not found: {metadata_path2}")
                    return jsonify({"error": f"Metadata for table {data['table2Name']} not found"}), 404
                    
                metadata2 = json.loads(metadata_blob2.download_as_string())
                
                # Ensure required metadata fields exist for second table
                if 'columns' not in metadata2:
                    metadata2['columns'] = []
                    print(f"[DEBUG] Warning: 'columns' field missing in second table metadata, using empty list")
                
                # Create columnTypes if missing for second table
                if 'columnTypes' not in metadata2:
                    print(f"[DEBUG] Warning: 'columnTypes' field missing in second table metadata, inferring from columns")
                    metadata2['columnTypes'] = {}
                    for col in metadata2.get('columns', []):
                        if isinstance(col, dict) and 'name' in col and 'type' in col:
                            col_name = col['name']
                            col_type = col['type']
                            # Map column types to pandas-compatible types
                            if col_type in ['float', 'number', 'numeric']:
                                metadata2['columnTypes'][col_name] = 'float64'
                            elif col_type in ['int', 'integer']:
                                metadata2['columnTypes'][col_name] = 'int64'
                            elif col_type in ['datetime', 'date']:
                                metadata2['columnTypes'][col_name] = 'datetime64[ns]'
                            else:
                                metadata2['columnTypes'][col_name] = 'object'
                
                print(f"[DEBUG] Metadata from storage loaded for both tables")
                print(f"[DEBUG] Table1 metadata keys: {list(metadata.keys())}")
                print(f"[DEBUG] Table2 metadata keys: {list(metadata2.keys())}")

                # Normalize operation type for run_chain
                operation_type = "merge-files"  # Always use hyphenated version for run_chain
                
                # Transform both metadata objects
                transformed_metadata = transform_metadata(metadata)
                transformed_metadata2 = transform_metadata(metadata2) if metadata2 else None

                # Call run_chain for merge operation
                result = run_chain(
                    user_input=data["query"],
                    operation_type=operation_type,
                    table_name=data["table1Name"],
                    process_id=data["process_id"],
                    dataframe_metadata=transformed_metadata,
                    table2_metadata=transformed_metadata2
                )
            except Exception as e:
                print(f"[DEBUG] Metadata processing error: {str(e)}")
                return jsonify({"error": f"Invalid metadata format: {str(e)}"}), 500

        elif data["operation_type"] == "reconcile":
            # Get source DataFrames and their metadata
            print(f"[DEBUG] Fetching source DataFrames for reconciliation")
            source_table_names = data.get("sourceTableNames", [])
            source_dfs = []
            metadata_list = []

            for table_name in source_table_names:
                print(f"[DEBUG] Fetching DataFrame - Name: {table_name}")
                df = DataFrame.query.filter_by(
                    name=table_name,
                    process_id=data["process_id"]
                ).first()
                if not df:
                    print(f"[DEBUG] Table not found: {table_name}")
                    return jsonify({
                        "error": f"DataFrame {table_name} not found in process {data['process_id']}"
                    }), 404
                source_dfs.append(df)
                
                # Get metadata from storage
                try:
                    metadata_path = f"{email}/process/{data['process_id']}/metadata/{table_name}.json"
                    metadata_blob = bucket.blob(metadata_path)
                    
                    if not metadata_blob.exists():
                        print(f"[DEBUG] Metadata not found: {metadata_path}")
                        return jsonify({"error": f"Metadata for table {table_name} not found"}), 404
                        
                    df_metadata = json.loads(metadata_blob.download_as_string())
                    metadata_list.append(df_metadata)
                except Exception as e:
                    print(f"[DEBUG] Metadata processing error for {table_name}: {str(e)}")
                    return jsonify({"error": f"Invalid metadata format for {table_name}: {str(e)}"}), 500

            if len(source_dfs) != 2:
                print(f"[DEBUG] Invalid number of source tables: {len(source_dfs)}")
                return jsonify({
                    "error": "Exactly two source tables are required for reconciliation"
                }), 400

            # Set metadata variables for run_chain
            metadata = metadata_list[0]
            metadata2 = metadata_list[1]
            dataframe = source_dfs[0]  # First DataFrame for reference

            # Transform both metadata objects
            transformed_metadata = transform_metadata(metadata)
            transformed_metadata2 = transform_metadata(metadata2) if metadata2 else None

            # Call run_chain with transformed metadata
            result = run_chain(
                user_input=data["query"],
                operation_type=data["operation_type"],
                table_name=source_table_names[0],
                process_id=data["process_id"],
                dataframe_metadata=transformed_metadata,
                table2_metadata=transformed_metadata2
            )
            print(f"[DEBUG] Reconciliation result: {result}")
            # Validate column names in the parameters
            if result.get("parameters"):
                # Get actual column names from metadata
                actual_columns1 = []
                actual_columns2 = []
                
                # Extract column names from first table metadata
                for col in metadata.get('columns', []):
                    if isinstance(col, dict) and 'name' in col:
                        actual_columns1.append(col['name'])
                    elif isinstance(col, str):
                        actual_columns1.append(col)
                
                # Extract column names from second table metadata
                for col in metadata2.get('columns', []):
                    if isinstance(col, dict) and 'name' in col:
                        actual_columns2.append(col['name'])
                    elif isinstance(col, str):
                        actual_columns2.append(col)

                print(f"[DEBUG] Available columns in first table: {actual_columns1}")
                print(f"[DEBUG] Available columns in second table: {actual_columns2}")

                # Validate keys
                for key in result["parameters"].get("keys", []):
                    if isinstance(key, dict):
                        if key.get('left') not in actual_columns1:
                            return jsonify({
                                "error": f"Column '{key.get('left')}' not found in first table. Available columns: {actual_columns1}"
                            }), 400
                        if key.get('right') not in actual_columns2:
                            return jsonify({
                                "error": f"Column '{key.get('right')}' not found in second table. Available columns: {actual_columns2}"
                            }), 400

                # Validate values
                for value in result["parameters"].get("values", []):
                    if isinstance(value, dict):
                        if value.get('left') not in actual_columns1:
                            return jsonify({
                                "error": f"Column '{value.get('left')}' not found in first table. Available columns: {actual_columns1}"
                            }), 400
                        if value.get('right') not in actual_columns2:
                            return jsonify({
                                "error": f"Column '{value.get('right')}' not found in second table. Available columns: {actual_columns2}"
                            }), 400

                # Validate cross-reference columns
                cross_reference = result["parameters"].get("crossReference", {})
                if isinstance(cross_reference, dict):
                    for ref in cross_reference.get("left", []):
                        if ref not in actual_columns1:
                            return jsonify({
                                "error": f"Cross-reference column '{ref}' not found in first table. Available columns: {actual_columns1}"
                            }), 400
                    for ref in cross_reference.get("right", []):
                        if ref not in actual_columns2:
                            return jsonify({
                                "error": f"Cross-reference column '{ref}' not found in second table. Available columns: {actual_columns2}"
                            }), 400

            # Create DataFrameOperation record
            df_operation = DataFrameOperation(
                process_id=data["process_id"],
                dataframe_id=source_dfs[0].id,  # Using first dataframe's ID
                operation_type=OperationType.RECONCILE_FILES.value,
                operation_subtype="reconcile",
                payload=result["parameters"]
            )
            db.session.add(df_operation)
            db.session.commit()
            print(f"[DEBUG] Created DataFrame Operation: {df_operation.id}")

            # Validate source DataFrames
            if not source_dfs or len(source_dfs) != 2:
                error_msg = "Source DataFrames list is empty or incomplete. Exactly two source tables are required for reconciliation."
                df_operation.set_error(error_msg)
                db.session.commit()
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "operationId": df_operation.id
                }), 400

            # Get source table names from the request
            request_source_table_names = data.get("sourceTableNames", [])
            if not request_source_table_names or len(request_source_table_names) != 2:
                error_msg = "Invalid source table names in request. Exactly two table names are required."
                df_operation.set_error(error_msg)
                db.session.commit()
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "operationId": df_operation.id
                }), 400

            # Update the source table names in the result parameters
            result["parameters"]["sourceTableNames"] = request_source_table_names

            # Check if keys and values are empty
            keys = result["parameters"].get("keys", [])
            values = result["parameters"].get("values", [])
            if not keys or not values:
                error_msg = "Cannot perform reconciliation without keys and values. Please specify the columns to match and compare."
                df_operation.set_error(error_msg)
                db.session.commit()
                return jsonify({
                    "success": False,
                    "error": error_msg,
                    "operationId": df_operation.id
                }), 400

            # Validate that no key or value has None for left or right
            for key in keys:
                if key.get('left') is None or key.get('right') is None:
                    error_msg = f"Invalid key pair: {key}. Both 'left' and 'right' must be specified."
                    df_operation.set_error(error_msg)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": error_msg,
                        "operationId": df_operation.id
                    }), 400

            for value in values:
                if value.get('left') is None or value.get('right') is None:
                    error_msg = f"Invalid value pair: {value}. Both 'left' and 'right' must be specified."
                    df_operation.set_error(error_msg)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": error_msg,
                        "operationId": df_operation.id
                    }), 400

            # Validate that all column names exist in their respective tables
            for key in keys:
                if key['left'] not in actual_columns1:
                    error_msg = f"Key column '{key['left']}' not found in first table. Available columns: {actual_columns1}"
                    df_operation.set_error(error_msg)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": error_msg,
                        "operationId": df_operation.id
                    }), 400
                if key['right'] not in actual_columns2:
                    error_msg = f"Key column '{key['right']}' not found in second table. Available columns: {actual_columns2}"
                    df_operation.set_error(error_msg)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": error_msg,
                        "operationId": df_operation.id
                    }), 400

            for value in values:
                if value['left'] not in actual_columns1:
                    error_msg = f"Value column '{value['left']}' not found in first table. Available columns: {actual_columns1}"
                    df_operation.set_error(error_msg)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": error_msg,
                        "operationId": df_operation.id
                    }), 400
                if value['right'] not in actual_columns2:
                    error_msg = f"Value column '{value['right']}' not found in second table. Available columns: {actual_columns2}"
                    df_operation.set_error(error_msg)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": error_msg,
                        "operationId": df_operation.id
                    }), 400

            # Add debug logging for reconciliation parameters
            print(f"[DEBUG] Reconciliation parameters:")
            print(f"[DEBUG] Keys: {json.dumps(keys, indent=2)}")
            print(f"[DEBUG] Values: {json.dumps(values, indent=2)}")
            print(f"[DEBUG] Settings: {json.dumps(result['parameters'].get('settings', {}), indent=2)}")
            print(f"[DEBUG] Cross reference: {json.dumps(result['parameters'].get('crossReference', {}), indent=2)}")

            operation_result = process_dataframe_reconciliation(
                email=email,
                process_id=data["process_id"],
                source_dfs=source_dfs,
                keys=keys,
                values=values,
                settings=result["parameters"].get('settings', {}),
                cross_reference=result["parameters"].get('crossReference', {}),
                output_table_name=output_table_name,
                existing_df=None
            )
            print(f"[DEBUG] Reconcile operation result: {operation_result}")

            if operation_result.get('success'):
                df_operation.set_success()
                
                # Store AI response with tokens and confidence
                ai_request.response = {
                    'ai_response': result,
                    'parameters': result.get("parameters"),
                    'metadata_used': result.get("metadata_used"),
                    'domain': result.get("domain"),
                    'tokens_used': result.get("domain", {}).get("tokens_used"),
                    'confidence': result.get("domain", {}).get("confidence")
                }
                ai_request.status = 'success'
                ai_request.response_time = datetime.now(timezone.utc)
                # Skip processing duration calculation to avoid timezone issues
                db.session.commit()
                
                # Fetch fresh metadata from storage after operation
                try:
                    output_name = data.get("output_table_name")
                    metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_name}.json"
                    metadata_blob = bucket.blob(metadata_path)
                    if metadata_blob.exists():
                        updated_metadata = json.loads(metadata_blob.download_as_string())
                    else:
                        updated_metadata = None
                except Exception as e:
                    print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                    updated_metadata = None
                
                return jsonify({
                    "success": True,
                    "operation_details": {
                        "id": df_operation.id,
                        "sourceTables": [df.name for df in source_dfs],
                        "dataframeId": df_operation.dataframe_id,
                        "message": operation_result.get('message'),
                        "newTableName": data.get("output_table_name"),
                        "statistics": operation_result.get('statistics'),
                        "aiRequestId": ai_request.id,
                        "retryCount": ai_request.retry_count,
                        "maxRetries": ai_request.max_retries,
                        "tokens_used": result.get("domain", {}).get("tokens_used"),
                        "confidence": result.get("domain", {}).get("confidence")
                    },
                    "parameters": result["parameters"],
                    "metadata_used": result.get("metadata_used"),
                    "domain": result.get("domain"),
                    "updated_metadata": updated_metadata
                })
            else:
                df_operation.set_error(operation_result.get('error'))
                # Store error in AI request
                ai_request.status = 'error'
                ai_request.error_message = operation_result.get('error')
                ai_request.response_time = datetime.now(timezone.utc)
                # Skip processing duration calculation to avoid timezone issues
                db.session.commit()
                return jsonify({
                    "success": False,
                    "error": operation_result.get('error'),
                    "operationId": df_operation.id,
                    "aiRequestId": ai_request.id
                }), 400

        elif data["operation_type"] == "add_column":
            try:
                print("[DEBUG] Processing add_column operation")
                
                # Get DataFrame
                print(f"[DEBUG] Fetching DataFrame - Name: {data['tableName']}")
                dataframe = DataFrame.query.filter_by(
                    name=data["tableName"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe:
                    print(f"[DEBUG] Table not found")
                    return jsonify({
                        "error": f"DataFrame {data['tableName']} not found in process {data['process_id']}"
                    }), 404
                
                # Get metadata from storage
                metadata_blob = None
                try:
                    metadata_path = f"{email}/process/{data['process_id']}/metadata/{data['tableName']}.json"
                    metadata_blob = bucket.blob(metadata_path)
                    
                    if not metadata_blob.exists():
                        print(f"[DEBUG] Metadata not found: {metadata_path}")
                        return jsonify({"error": f"Metadata for table {data['tableName']} not found"}), 404
                        
                    metadata = json.loads(metadata_blob.download_as_string())
                    print(f"[DEBUG] Metadata from storage loaded successfully")
                    
                    # Transform metadata before using it
                    transformed_metadata = transform_metadata(metadata)
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None
                    )
                    print(f"[DEBUG] Add column result: {result}")
                    if not result.get("success"):
                        print(f"[DEBUG] run_chain error: {result.get('error')}")
                        return jsonify({"error": result.get("error")}), 400
                        
                except Exception as e:
                    print(f"[DEBUG] Metadata processing error: {str(e)}")
                    return jsonify({"error": f"Invalid metadata format: {str(e)}"}), 500

                # Update parameters with output table name if provided
                if output_table_name:
                    result["parameters"]["newTableName"] = output_table_name
                
                # Fix operations format if needed
                if result["parameters"].get("operationType") == "calculate":
                    operations = result["parameters"].get("operations", [])
                    for op in operations:
                        # If column2 is a number, move it to fixed_value
                        if "column2" in op and isinstance(op["column2"], (int, float)):
                            op["fixed_value"] = op["column2"]
                            op["column2"] = None

                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.ADD_COLUMN.value,
                    operation_subtype=result["parameters"].get("operationType"),
                    payload=result["parameters"]
                )
                db.session.add(df_operation)
                db.session.commit()
                print(f"[DEBUG] Created DataFrame Operation: {df_operation.id}")

                operation_result = process_add_column(
                    email=email,
                    process_name=process.id,
                    table_name=result["parameters"].get("tableName"),
                    new_column_name=result["parameters"].get("newColumnName"),
                    operation_type=result["parameters"].get("operationType"),
                    operation_params=result["parameters"]
                )
                print(f"[DEBUG] Add column operation result: {operation_result}")

                if operation_result.get('success'):
                    df_operation.set_success()
                    
                    # Store AI response with tokens and confidence
                    ai_request.response = {
                        'ai_response': result,
                        'parameters': result.get("parameters"),
                        'metadata_used': result.get("metadata_used"),
                        'domain': result.get("domain"),
                        'tokens_used': result.get("domain", {}).get("tokens_used"),
                        'confidence': result.get("domain", {}).get("confidence")
                    }
                    ai_request.status = 'success'
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    
                    # Fetch fresh metadata from storage after operation
                    try:
                        # For add_column, the table name is the same (no new table is created)
                        table_name = result["parameters"].get("tableName")
                        metadata_path = f"{email}/process/{data['process_id']}/metadata/{table_name}.json"
                        metadata_blob = bucket.blob(metadata_path)
                        if metadata_blob.exists():
                            updated_metadata = json.loads(metadata_blob.download_as_string())
                        else:
                            updated_metadata = None
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                        updated_metadata = None
                    
                    return jsonify({
                        "success": True,
                        "operation_details": {
                            "id": df_operation.id,
                            "sourceTable": result["parameters"].get("tableName"),
                            "dataframeId": df_operation.dataframe_id,
                            "message": operation_result.get('message'),
                            "newTableName": result["parameters"].get("newTableName"),
                            "aiRequestId": ai_request.id,
                            "retryCount": ai_request.retry_count,
                            "maxRetries": ai_request.max_retries,
                            "tokens_used": result.get("domain", {}).get("tokens_used"),
                            "confidence": result.get("domain", {}).get("confidence")
                        },
                        "parameters": result["parameters"],
                        "metadata_used": result.get("metadata_used"),
                        "domain": result.get("domain"),
                        "updated_metadata": updated_metadata
                    })
                else:
                    df_operation.set_error(operation_result.get('error'))
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = operation_result.get('error')
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": operation_result.get('error'),
                        "operationId": df_operation.id,
                        "aiRequestId": ai_request.id
                    }), 400

            except Exception as e:
                print(f"[DEBUG] Error in add_column processing: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                
                # Rollback the session before attempting to set error
                db.session.rollback()
                
                try:
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = str(e)
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                except Exception as inner_e:
                    print(f"[DEBUG] Error setting error status: {str(inner_e)}")
                    db.session.rollback()
                    
                return jsonify({"error": str(e)}), 500

        elif data["operation_type"] == "sort_filter":
            try:
                print("[DEBUG] Processing sort_filter operation")
                
                # Get DataFrame
                print(f"[DEBUG] Fetching DataFrame - Name: {data['tableName']}")
                dataframe = DataFrame.query.filter_by(
                    name=data["tableName"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe:
                    print(f"[DEBUG] Table not found")
                    return jsonify({
                        "error": f"DataFrame {data['tableName']} not found in process {data['process_id']}"
                    }), 404
                
                # Get metadata from storage
                metadata_blob = None
                try:
                    metadata_path = f"{email}/process/{data['process_id']}/metadata/{data['tableName']}.json"
                    metadata_blob = bucket.blob(metadata_path)
                    
                    if not metadata_blob.exists():
                        print(f"[DEBUG] Metadata not found: {metadata_path}")
                        return jsonify({"error": f"Metadata for table {data['tableName']} not found"}), 404
                        
                    metadata = json.loads(metadata_blob.download_as_string())
                    print(f"[DEBUG] Metadata from storage loaded successfully")
                    
                    # Transform metadata before using it
                    transformed_metadata = transform_metadata(metadata)
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None
                    )
                    
                    if not result.get("success"):
                        print(f"[DEBUG] run_chain error: {result.get('error')}")
                        return jsonify({"error": result.get("error")}), 400
                        
                except Exception as e:
                    print(f"[DEBUG] Metadata processing error: {str(e)}")
                    return jsonify({"error": f"Invalid metadata format: {str(e)}"}), 500

                # Update parameters with output table name if provided
                if output_table_name:
                    result["parameters"]["output_table_name"] = output_table_name
                elif not result["parameters"].get("output_table_name"):
                    result["parameters"]["output_table_name"] = f"{data['tableName']}_filtered"

                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.SORT_FILTER.value,
                    operation_subtype="sort_filter",
                    payload=result["parameters"]
                )
                db.session.add(df_operation)
                db.session.commit()
                print(f"[DEBUG] Created DataFrame Operation: {df_operation.id}")

                operation_result = process_sort_filter_data(
                    email=email,
                    process_id=data["process_id"],
                    source_df=dataframe,
                    sort_config=result["parameters"].get("sort_config", []),
                    filter_config=result["parameters"].get("filter_config", []),
                    output_table_name=result["parameters"]["output_table_name"],
                    existing_df=None
                )
                print(f"[DEBUG] Sort filter operation result: {operation_result}")
                
                if operation_result.get('success'):
                    df_operation.set_success()
                    
                    # Store AI response with tokens and confidence
                    ai_request.response = {
                        'ai_response': result,
                        'parameters': result.get("parameters"),
                        'metadata_used': result.get("metadata_used"),
                        'domain': result.get("domain"),
                        'tokens_used': result.get("domain", {}).get("tokens_used"),
                        'confidence': result.get("domain", {}).get("confidence")
                    }
                    ai_request.status = 'success'
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    
                    # Fetch fresh metadata from storage after operation
                    try:
                        output_name = result["parameters"]["output_table_name"]
                        metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_name}.json"
                        metadata_blob = bucket.blob(metadata_path)
                        if metadata_blob.exists():
                            updated_metadata = json.loads(metadata_blob.download_as_string())
                        else:
                            updated_metadata = None
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                        updated_metadata = None
                    
                    return jsonify({
                        "success": True,
                        "operation_details": {
                            "id": df_operation.id,
                            "sourceTable": result["parameters"].get("table_name"),
                            "dataframeId": df_operation.dataframe_id,
                            "message": operation_result.get('message'),
                            "newTableName": result["parameters"].get("output_table_name"),
                            "aiRequestId": ai_request.id,
                            "retryCount": ai_request.retry_count,
                            "maxRetries": ai_request.max_retries,
                            "tokens_used": result.get("domain", {}).get("tokens_used"),
                            "confidence": result.get("domain", {}).get("confidence")
                        },
                        "parameters": result["parameters"],
                        "metadata_used": result.get("metadata_used"),
                        "domain": result.get("domain"),
                        "updated_metadata": updated_metadata
                    })
                else:
                    df_operation.set_error(operation_result.get('error'))
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = operation_result.get('error')
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": operation_result.get('error'),
                        "operationId": df_operation.id,
                        "aiRequestId": ai_request.id
                    }), 400

            except Exception as e:
                print(f"[DEBUG] Error in sort_filter processing: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                
                # Rollback the session before attempting to set error
                db.session.rollback()
                
                try:
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = str(e)
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                except Exception as inner_e:
                    print(f"[DEBUG] Error setting error status: {str(inner_e)}")
                    db.session.rollback()
                    
                return jsonify({"error": str(e)}), 500

        elif data["operation_type"] == "group_pivot":
            try:
                print("[DEBUG] Processing group_pivot operation")
                
                # Get DataFrame
                print(f"[DEBUG] Fetching DataFrame - Name: {data['tableName']}")
                dataframe = DataFrame.query.filter_by(
                    name=data["tableName"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe:
                    print(f"[DEBUG] Table not found")
                    return jsonify({
                        "error": f"DataFrame {data['tableName']} not found in process {data['process_id']}"
                    }), 404
                
                # Get metadata from storage
                metadata_blob = None
                try:
                    metadata_path = f"{email}/process/{data['process_id']}/metadata/{data['tableName']}.json"
                    metadata_blob = bucket.blob(metadata_path)
                    
                    if not metadata_blob.exists():
                        print(f"[DEBUG] Metadata not found: {metadata_path}")
                        return jsonify({"error": f"Metadata for table {data['tableName']} not found"}), 404
                        
                    metadata = json.loads(metadata_blob.download_as_string())
                    print(f"[DEBUG] Metadata from storage loaded successfully")
                    
                    # Transform metadata before using it
                    transformed_metadata = transform_metadata(metadata)
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None
                    )
                    
                    if not result.get("success"):
                        print(f"[DEBUG] run_chain error: {result.get('error')}")
                        return jsonify({"error": result.get("error")}), 400
                        
                except Exception as e:
                    print(f"[DEBUG] Metadata processing error: {str(e)}")
                    return jsonify({"error": f"Invalid metadata format: {str(e)}"}), 500

                # Update parameters with output table name if provided
                if output_table_name:
                    result["parameters"]["outputTableName"] = output_table_name
                elif not result["parameters"].get("outputTableName"):
                    result["parameters"]["outputTableName"] = f"{data['tableName']}_pivot"

                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.GROUP_PIVOT.value,
                    operation_subtype="group_pivot",
                    payload=result["parameters"]
                )
                db.session.add(df_operation)
                db.session.commit()
                print(f"[DEBUG] Created DataFrame Operation: {df_operation.id}")

                operation_result = process_pivot_table(
                    email=email,
                    process_id=data["process_id"],
                    source_table_name=result["parameters"].get("tableName"),
                    row_index=result["parameters"].get("rowIndex"),
                    column_index=result["parameters"].get("columnIndex"),
                    pivot_values=result["parameters"].get("pivotValues"),
                    output_table_name=output_table_name
                )
                print(f"[DEBUG] Group pivot operation result: {operation_result}")

                if operation_result.get('success'):
                    df_operation.set_success()
                    
                    # Store AI response with tokens and confidence
                    ai_request.response = {
                        'ai_response': result,
                        'parameters': result.get("parameters"),
                        'metadata_used': result.get("metadata_used"),
                        'domain': result.get("domain"),
                        'tokens_used': result.get("domain", {}).get("tokens_used"),
                        'confidence': result.get("domain", {}).get("confidence")
                    }
                    ai_request.status = 'success'
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    
                    # Fetch fresh metadata from storage after operation
                    try:
                        output_name = result["parameters"]["outputTableName"]
                        metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_name}.json"
                        metadata_blob = bucket.blob(metadata_path)
                        if metadata_blob.exists():
                            updated_metadata = json.loads(metadata_blob.download_as_string())
                        else:
                            updated_metadata = None
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                        updated_metadata = None
                    
                    return jsonify({
                        "success": True,
                        "operation_details": {
                            "id": df_operation.id,
                            "sourceTable": result["parameters"].get("tableName"),
                            "dataframeId": df_operation.dataframe_id,
                            "message": operation_result.get('message'),
                            "newTableName": result["parameters"].get("outputTableName"),
                            "aiRequestId": ai_request.id,
                            "retryCount": ai_request.retry_count,
                            "maxRetries": ai_request.max_retries,
                            "tokens_used": result.get("domain", {}).get("tokens_used"),
                            "confidence": result.get("domain", {}).get("confidence")
                        },
                        "parameters": result["parameters"],
                        "metadata_used": result.get("metadata_used"),
                        "domain": result.get("domain"),
                        "updated_metadata": updated_metadata
                    })
                else:
                    df_operation.set_error(operation_result.get('error'))
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = operation_result.get('error')
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": operation_result.get('error'),
                        "operationId": df_operation.id,
                        "aiRequestId": ai_request.id
                    }), 400

            except Exception as e:
                print(f"[DEBUG] Error in group_pivot processing: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                
                # Rollback the session before attempting to set error
                db.session.rollback()
                
                try:
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = str(e)
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                except Exception as inner_e:
                    print(f"[DEBUG] Error setting error status: {str(inner_e)}")
                    db.session.rollback()
                    
                return jsonify({"error": str(e)}), 500

        elif data["operation_type"] in ["merge-files", "merge_files"]:
            try:
                print("[DEBUG] Processing merge operation")
                
                # Get first DataFrame
                print(f"[DEBUG] Fetching first DataFrame - Name: {data['table1Name']}")
                dataframe = DataFrame.query.filter_by(
                    name=data["table1Name"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe:
                    print(f"[DEBUG] First table not found")
                    return jsonify({
                        "error": f"DataFrame {data['table1Name']} not found in process {data['process_id']}"
                    }), 404

                # Get second DataFrame
                print(f"[DEBUG] Fetching second DataFrame - Name: {data['table2Name']}")
                dataframe2 = DataFrame.query.filter_by(
                    name=data["table2Name"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe2:
                    print(f"[DEBUG] Second table not found")
                    return jsonify({
                        "error": f"DataFrame {data['table2Name']} not found in process {data['process_id']}"
                    }), 404
                
                # Initialize result if not already set
                if not result:
                    # Get metadata from storage for both tables
                    try:
                        # Get first table metadata
                        metadata_path1 = f"{email}/process/{data['process_id']}/metadata/{data['table1Name']}.json"
                        metadata_blob1 = bucket.blob(metadata_path1)
                        if not metadata_blob1.exists():
                            return jsonify({"error": f"Metadata for table {data['table1Name']} not found"}), 404
                        metadata1 = json.loads(metadata_blob1.download_as_string())
                        
                        # Get second table metadata
                        metadata_path2 = f"{email}/process/{data['process_id']}/metadata/{data['table2Name']}.json"
                        metadata_blob2 = bucket.blob(metadata_path2)
                        if not metadata_blob2.exists():
                            return jsonify({"error": f"Metadata for table {data['table2Name']} not found"}), 404
                        metadata2 = json.loads(metadata_blob2.download_as_string())
                        
                        # Transform metadata
                        transformed_metadata1 = transform_metadata(metadata1)
                        transformed_metadata2 = transform_metadata(metadata2)
                        
                        # Call run_chain to get merge parameters
                        result = run_chain(
                            user_input=data["query"],
                            operation_type="merge-files",
                            table_name=data["table1Name"],
                            process_id=data["process_id"],
                            dataframe_metadata=transformed_metadata1,
                            table2_metadata=transformed_metadata2
                        )
                        
                        if not result or not result.get("parameters"):
                            return jsonify({"error": "Failed to generate merge parameters"}), 500
                            
                    except Exception as e:
                        print(f"[DEBUG] Error getting metadata: {str(e)}")
                        return jsonify({"error": f"Error processing metadata: {str(e)}"}), 500

                # Use table names from request body
                table1_name = data["table1Name"]
                table2_name = data["table2Name"]
                output_table_name = data["output_table_name"]

                # Update parameters with table names from request
                result["parameters"].update({
                    "table1Name": table1_name,
                    "table2Name": table2_name,
                    "outputTableName": output_table_name
                })

                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,  # Using first dataframe's ID
                    operation_type=OperationType.MERGE_FILES.value,
                    operation_subtype=result["parameters"].get("mergeType"),
                    payload=result["parameters"]
                )
                db.session.add(df_operation)
                db.session.commit()
                print(f"[DEBUG] Created DataFrame Operation: {df_operation.id}")

                operation_result = process_merge_tables(
                    email=email,
                    process_id=data["process_id"],
                    table1=dataframe,  # Pass DataFrame object directly
                    table2=dataframe2,  # Pass DataFrame object directly
                    merge_type=result["parameters"].get("mergeType"),
                    merge_method=result["parameters"].get("mergeMethod"),
                    key_pairs=result["parameters"].get("keyPairs"),
                    show_count_summary=result["parameters"].get("showCountSummary", False),
                    output_table_name=output_table_name
                )
                print(f"[DEBUG] Merge files operation result: {operation_result}")

                if operation_result.get('success'):
                    df_operation.set_success()
                    
                    # Store AI response with tokens and confidence
                    ai_request.response = {
                        'ai_response': result,
                        'parameters': result.get("parameters"),
                        'metadata_used': result.get("metadata_used"),
                        'domain': result.get("domain"),
                        'tokens_used': result.get("domain", {}).get("tokens_used"),
                        'confidence': result.get("domain", {}).get("confidence")
                    }
                    ai_request.status = 'success'
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    
                    # Fetch fresh metadata from storage after operation
                    try:
                        output_name = result["parameters"]["outputTableName"]
                        metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_name}.json"
                        metadata_blob = bucket.blob(metadata_path)
                        if metadata_blob.exists():
                            updated_metadata = json.loads(metadata_blob.download_as_string())
                        else:
                            updated_metadata = None
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                        updated_metadata = None
                    
                    return jsonify({
                        "success": True,
                        "operation_details": {
                            "id": df_operation.id,
                            "sourceTable": table1_name,
                            "secondTable": table2_name,
                            "dataframeId": df_operation.dataframe_id,
                            "message": operation_result.get('message'),
                            "newTableName": output_table_name,
                            "aiRequestId": ai_request.id,
                            "retryCount": ai_request.retry_count,
                            "maxRetries": ai_request.max_retries,
                            "tokens_used": result.get("domain", {}).get("tokens_used"),
                            "confidence": result.get("domain", {}).get("confidence")
                        },
                        "parameters": result["parameters"],
                        "metadata_used": result.get("metadata_used"),
                        "domain": result.get("domain"),
                        "updated_metadata": updated_metadata
                    })
                else:
                    df_operation.set_error(operation_result.get('error'))
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = operation_result.get('error')
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": operation_result.get('error'),
                        "operationId": df_operation.id,
                        "aiRequestId": ai_request.id
                    }), 400

            except Exception as e:
                print(f"[DEBUG] Error in merge operation: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                
                # Rollback the session before attempting to set error
                db.session.rollback()
                
                try:
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = str(e)
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                except Exception as inner_e:
                    print(f"[DEBUG] Error setting error status: {str(inner_e)}")
                    db.session.rollback()
                    
                return jsonify({"error": str(e)}), 500

        elif data["operation_type"] == "replace_rename_reorder":
            try:
                print("[DEBUG] Processing replace_rename_reorder operation")
                
                # Get DataFrame
                print(f"[DEBUG] Fetching DataFrame - Name: {data['tableName']}")
                dataframe = DataFrame.query.filter_by(
                    name=data["tableName"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe:
                    print(f"[DEBUG] Table not found")
                    return jsonify({
                        "error": f"DataFrame {data['tableName']} not found in process {data['process_id']}"
                    }), 404
                
                # Initialize variables
                metadata = None
                result = None
                
                try:
                    # Get metadata from storage
                    metadata_path = f"{email}/process/{data['process_id']}/metadata/{data['tableName']}.json"
                    metadata_blob = bucket.blob(metadata_path)
                    
                    if not metadata_blob.exists():
                        print(f"[DEBUG] Metadata not found: {metadata_path}")
                        return jsonify({"error": f"Metadata for table {data['tableName']} not found"}), 404
                        
                    metadata = json.loads(metadata_blob.download_as_string())
                    print(f"[DEBUG] Metadata from storage loaded successfully")
                    
                    # Transform metadata before using it
                    transformed_metadata = transform_metadata(metadata)
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None
                    )
                    
                    if not result.get("success"):
                        print(f"[DEBUG] run_chain error: {result.get('error')}")
                        return jsonify({"error": result.get("error")}), 400
                        
                except Exception as e:
                    print(f"[DEBUG] Metadata processing error: {str(e)}")
                    return jsonify({"error": f"Error processing metadata: {str(e)}"}), 500

                if not result or not result.get("parameters"):
                    return jsonify({"error": "Failed to generate operation parameters"}), 500

                # Update parameters with output table name if provided
                if output_table_name:
                    result["parameters"]["outputTableName"] = output_table_name
                elif not result["parameters"].get("outputTableName"):
                    result["parameters"]["outputTableName"] = f"{data['tableName']}_modified"

                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.REPLACE_RENAME_REORDER.value,
                    operation_subtype="replace_rename_reorder",
                    payload=result["parameters"]
                )
                db.session.add(df_operation)
                db.session.commit()
                print(f"[DEBUG] Created DataFrame Operation: {df_operation.id}")

                operation_result = process_dataframe_operations(
                    email=email,
                    process_id=data["process_id"],
                    source_df=dataframe,  # Pass DataFrame object directly
                    operations=result["parameters"].get("operations", []),
                    output_table_name=output_table_name,
                    existing_df=None
                )
                print(f"[DEBUG] Replace/Rename/Reorder operation result: {operation_result}")

                if operation_result.get('success'):
                    df_operation.set_success()
                    
                    # Store AI response with tokens and confidence
                    ai_request.response = {
                        'ai_response': result,
                        'parameters': result.get("parameters"),
                        'metadata_used': result.get("metadata_used"),
                        'domain': result.get("domain"),
                        'tokens_used': result.get("domain", {}).get("tokens_used"),
                        'confidence': result.get("domain", {}).get("confidence")
                    }
                    ai_request.status = 'success'
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    
                    # Fetch fresh metadata from storage after operation
                    try:
                        output_name = result["parameters"]["outputTableName"]
                        metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_name}.json"
                        metadata_blob = bucket.blob(metadata_path)
                        if metadata_blob.exists():
                            updated_metadata = json.loads(metadata_blob.download_as_string())
                        else:
                            updated_metadata = None
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                        updated_metadata = None
                    
                    return jsonify({
                        "success": True,
                        "operation_details": {
                            "id": df_operation.id,
                            "sourceTable": result["parameters"].get("tableName"),
                            "dataframeId": df_operation.dataframe_id,
                            "message": operation_result.get('message'),
                            "newTableName": result["parameters"].get("outputTableName"),
                            "aiRequestId": ai_request.id,
                            "retryCount": ai_request.retry_count,
                            "maxRetries": ai_request.max_retries,
                            "tokens_used": result.get("domain", {}).get("tokens_used"),
                            "confidence": result.get("domain", {}).get("confidence")
                        },
                        "parameters": result["parameters"],
                        "metadata_used": result.get("metadata_used"),
                        "domain": result.get("domain"),
                        "updated_metadata": updated_metadata
                    })
                else:
                    df_operation.set_error(operation_result.get('error'))
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = operation_result.get('error')
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": operation_result.get('error'),
                        "operationId": df_operation.id,
                        "aiRequestId": ai_request.id
                    }), 400

            except Exception as e:
                print(f"[DEBUG] Error in replace_rename_reorder processing: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                
                # Rollback the session before attempting to set error
                db.session.rollback()
                
                try:
                    # Store error in AI request
                    ai_request.status = 'error'
                    ai_request.error_message = str(e)
                    ai_request.response_time = datetime.now(timezone.utc)
                    # Skip processing duration calculation to avoid timezone issues
                    db.session.commit()
                except Exception as inner_e:
                    print(f"[DEBUG] Error setting error status: {str(inner_e)}")
                    db.session.rollback()
                    
                return jsonify({"error": str(e)}), 500

        elif data["operation_type"] == "format":
            try:
                print("[DEBUG] Processing format operation")

                # Fetch the source DataFrame
                dataframe = DataFrame.query.filter_by(
                    name=data["tableName"],
                    process_id=data["process_id"]
                ).first()
                if not dataframe:
                    print(f"[DEBUG] Table not found")
                    return jsonify({
                        "error": f"DataFrame {data['tableName']} not found in process {data['process_id']}"
                    }), 404

                # Load metadata JSON from storage bucket
                metadata_path = f"{email}/process/{data['process_id']}/metadata/{data['tableName']}.json"
                metadata_blob = bucket.blob(metadata_path)
                if not metadata_blob.exists():
                    return jsonify({
                        "error": f"Metadata for table {data['tableName']} not found"
                    }), 404

                metadata = json.loads(metadata_blob.download_as_string())

                # Transform metadata as needed
                transformed_metadata = transform_metadata(metadata)

                # Call AI chain to get formatting parameters
                result = run_chain(
                    user_input=data["query"],
                    operation_type="format",
                    table_name=data["tableName"],
                    process_id=data["process_id"],
                    dataframe_metadata=transformed_metadata,
                    table2_metadata=None
                )

                if not result or not result.get("parameters"):
                    return jsonify({"error": "Failed to generate format parameters"}), 500

                # Determine output table name
                output_table_name = data.get("output_table_name") or result["parameters"].get("outputTableName") or f"{data['tableName']}_formatted"
                result["parameters"]["outputTableName"] = output_table_name

                # Create a DataFrameOperation record for audit trail
                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type="format",
                    operation_subtype="format",
                    payload=result["parameters"]
                )
                db.session.add(df_operation)
                db.session.commit()

                # === Call your backend function to perform the actual formatting ===
                # This function you must implement similar to your other operations
                operation_result = process_format_table(
                    email=email,
                    process_id=data["process_id"],
                    source_df=dataframe,
                    format_params=result["parameters"],
                    output_table_name=output_table_name
                )

                print(f"[DEBUG] Format operation result: {operation_result}")

                if operation_result.get('success'):
                    df_operation.set_success()
                    db.session.commit()

                    # Update AIRequest with success response info
                    ai_request.response = {
                        'ai_response': result,
                        'parameters': result.get("parameters"),
                        'metadata_used': result.get("metadata_used"),
                        'domain': result.get("domain"),
                        'tokens_used': result.get("domain", {}).get("tokens_used"),
                        'confidence': result.get("domain", {}).get("confidence")
                    }
                    ai_request.status = 'success'
                    ai_request.response_time = datetime.now(timezone.utc)
                    db.session.commit()

                    # Load updated metadata of the new formatted table
                    try:
                        updated_metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_table_name}.json"
                        updated_metadata_blob = bucket.blob(updated_metadata_path)
                        if updated_metadata_blob.exists():
                            updated_metadata = json.loads(updated_metadata_blob.download_as_string())
                        else:
                            updated_metadata = None
                    except Exception as e:
                        print(f"[DEBUG] Could not fetch updated metadata: {str(e)}")
                        updated_metadata = None

                    # Return success response with details
                    return jsonify({
                        "success": True,
                        "operation_details": {
                            "id": df_operation.id,
                            "sourceTable": data["tableName"],
                            "dataframeId": df_operation.dataframe_id,
                            "message": operation_result.get("message", "Format operation completed successfully"),
                            "newTableName": output_table_name,
                            "aiRequestId": ai_request.id,
                            "retryCount": ai_request.retry_count,
                            "maxRetries": ai_request.max_retries,
                            "tokens_used": result.get("domain", {}).get("tokens_used"),
                            "confidence": result.get("domain", {}).get("confidence")
                        },
                        "parameters": result["parameters"],
                        "metadata_used": result.get("metadata_used"),
                        "domain": result.get("domain"),
                        "updated_metadata": updated_metadata
                    })

                else:
                    df_operation.set_error(operation_result.get('error', 'Unknown error during format processing'))
                    ai_request.status = 'error'
                    ai_request.error_message = operation_result.get('error', 'Unknown error during format processing')
                    ai_request.response_time = datetime.now(timezone.utc)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": operation_result.get('error', 'Format operation failed'),
                        "operationId": df_operation.id,
                        "aiRequestId": ai_request.id
                    }), 400

            except Exception as e:
                print(f"[DEBUG] Exception during format operation: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                db.session.rollback()

                try:
                    ai_request.status = 'error'
                    ai_request.error_message = str(e)
                    ai_request.response_time = datetime.now(timezone.utc)
                    db.session.commit()
                except Exception as inner_e:
                    print(f"[DEBUG] Failed to update AIRequest error status: {str(inner_e)}")
                    db.session.rollback()

                # Return friendly AI-like error reply (you can customize this)
                return jsonify({
                    "success": False,
                    "reply": (
                        "Sorry, I encountered an error while processing your format request. "
                        "Please check your input and try again."
                    ),
                    "error_details": str(e)
                }), 500

@nlp_bp.route("/regex", methods=["POST"])
def generate_regex_pattern():
    """Generate regex pattern from natural language description"""
    try:
        print("[DEBUG] Starting generate_regex_pattern endpoint")
        data = request.json
        email = request.headers.get("X-User-Email")
        print(f"[DEBUG] Received request - Email: {email}, Data: {json.dumps(data, indent=2)}")

        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        if not data.get("description"):
            return jsonify({"error": "Natural language description is required."}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Create AI request for tracking
        ai_request = AIRequest(
            process_id=data.get("process_id"),
            user_id=user.id,
            email=email,
            operation_type="regex",
            query=data.get("description")
        )
        db.session.add(ai_request)
        db.session.commit()
        print(f"[DEBUG] Created new AI request {ai_request.id}")

        try:
            # Call run_chain with regex-specific prompt
            result = run_chain(
                user_input=data["description"],
                operation_type="regex",
                table_name=None,  # Not needed for regex
                process_id=data.get("process_id"),
                dataframe_metadata={
                    "columns": [],
                    "summary": {
                        "nullCounts": {},
                        "uniqueCounts": {}
                    },
                    "columnTypes": {}
                },
                table2_metadata=None
            )

            if not result.get("success"):
                print(f"[DEBUG] run_chain error: {result.get('error')}")
                ai_request.status = 'error'
                ai_request.error_message = result.get("error")
                ai_request.response_time = datetime.now(timezone.utc)
                db.session.commit()
                return jsonify({"error": result.get("error")}), 400

            # Store AI response with tokens and confidence
            ai_request.response = {
                'ai_response': result,
                'pattern': result.get("parameters", {}).get("pattern"),
                'metadata_used': result.get("metadata_used"),
                'domain': result.get("domain"),
                'tokens_used': result.get("domain", {}).get("tokens_used"),
                'confidence': result.get("domain", {}).get("confidence")
            }
            ai_request.status = 'success'
            ai_request.response_time = datetime.now(timezone.utc)
            db.session.commit()

            return jsonify({
                "success": True,
                "pattern": result.get("parameters", {}).get("pattern"),
                "explanation": result.get("parameters", {}).get("explanation"),
                "aiRequestId": ai_request.id,
                "tokens_used": result.get("domain", {}).get("tokens_used"),
                "confidence": result.get("domain", {}).get("confidence")
            })

        except Exception as e:
            print(f"[DEBUG] Error in regex generation: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # Rollback the session before attempting to set error
            db.session.rollback()
            
            try:
                # Store error in AI request
                ai_request.status = 'error'
                ai_request.error_message = str(e)
                ai_request.response_time = datetime.now(timezone.utc)
                db.session.commit()
            except Exception as inner_e:
                print(f"[DEBUG] Error setting error status: {str(inner_e)}")
                db.session.rollback()
                
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500 