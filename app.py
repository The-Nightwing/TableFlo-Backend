import asyncio
from flask import Blueprint, request, jsonify
from flask import Blueprint, request, jsonify
from models import DataFrame, UserProcess, DataFrameOperation, OperationType, db, FormattingStep, AIRequest, User
from llm_chain import run_chain, get_openai_api_key
from llm_chain import PROMPT_INCOMPLETE_MESSAGE
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
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
import copy

COLUMN_NOT_FOUND_MESSAGE = "The specified column name(s) could not be found."


def _build_column_lookup(columns: set):
    """Create a case-insensitive lookup for column names.

    Returns a dict mapping lower(column) -> original column.
    """
    lookup = {}
    for c in columns or set():
        if isinstance(c, str):
            lookup[c.lower()] = c
    return lookup


def _check_column_ci(column_name: str, available_lookup: dict, missing: set):
    """Case-insensitive column check.

    We treat column names as case-insensitive because users often type them differently
    than the source schema, and LLMs may change casing.
    """
    if not column_name or not isinstance(column_name, str):
        return
    if column_name.lower() not in available_lookup:
        missing.add(column_name)

nlp_bp = Blueprint('nlp', __name__, url_prefix='/api/nlp')


class IntentValidation(BaseModel):
    match: bool = Field(description="Whether the query matches the expected operation")
    predicted_operation: str = Field(description="Predicted operation type by the AI")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    explanation: str = Field(description="Short explanation why the model predicted this")


def ai_validate_intent(user_query: str, expected_operation: str, metadata: dict) -> dict:
    """Ask the LLM to validate whether the user's query intent matches the expected operation.

    Returns a dict: { success: bool, match: bool, predicted_operation: str, confidence: float, explanation: str }
    On failure returns { success: False, error: str }
    """
    try:
        parser = PydanticOutputParser(pydantic_object=IntentValidation)
        format_instructions = parser.get_format_instructions()

        # Build a compact column summary for context
        cols = metadata.get('columns') or []
        ctypes = metadata.get('columnTypes') or {}
        column_info = "\n".join([f"- {c} ({ctypes.get(c, 'unknown')})" for c in cols])

        prompt = PromptTemplate(
            template=(
                "You are a precise classifier. Given a user's natural-language query and the expected operation type,\n"
                "decide whether the query's intent matches the expected operation. Respond ONLY in the JSON schema described.\n\n"
                "Schema: {format_instructions}\n\n"
                "Context - Table columns:\n{column_info}\n\n"
                "User query: {user_query}\n"
                "Expected operation: {expected_operation}\n\n"
                "Choose predicted_operation from: add_column, sort, filter, merge, reconcile, group_pivot, formatting, file_operations, unknown.\n"
                "If you are uncertain, set match=false and give a low confidence. Be concise in explanation."
            ),
            input_variables=["user_query", "expected_operation", "column_info"],
            partial_variables={"format_instructions": format_instructions}
        )

        api_key = get_openai_api_key()
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=api_key)
        chain = prompt | llm | parser

        res = chain.invoke({
            "user_query": user_query,
            "expected_operation": expected_operation,
            "column_info": column_info
        })

        # Normalize parser output: LangChain's PydanticOutputParser may return a pydantic model
        # or an object with an `args` attribute. Handle both and also dicts.
        def _get_field(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            if hasattr(obj, key):
                return getattr(obj, key)
            # pydantic models expose model_dump for Python-side dict
            if hasattr(obj, 'model_dump'):
                try:
                    return obj.model_dump().get(key, default)
                except Exception:
                    return default
            return default

        parsed_obj = None
        if hasattr(res, 'args'):
            parsed_obj = res.args
        else:
            parsed_obj = res

        match_val = _get_field(parsed_obj, 'match', False)
        predicted_operation = _get_field(parsed_obj, 'predicted_operation') or _get_field(parsed_obj, 'predictedOperation')
        confidence = _get_field(parsed_obj, 'confidence', 0.0)
        explanation = _get_field(parsed_obj, 'explanation', '')

        return {
            "success": True,
            "match": bool(match_val),
            "predicted_operation": predicted_operation,
            "confidence": float(confidence) if confidence is not None else 0.0,
            "explanation": explanation
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def validate_intent_or_abort(user_query: str, expected_operation: str, metadata: dict, min_confidence: float = 0.6):
    """Single helper to run AI intent validation and return a Flask response tuple on failure.

    Returns None on success, or a (response_body, status_code) tuple to be returned by the caller.
    """
    validation = ai_validate_intent(user_query, expected_operation, metadata)
    if not validation.get("success"):
        print(f"[DEBUG] AI intent validation failed: {validation.get('error')}")
        return ({"error": "AI validation failed", "details": validation.get("error")}, 500)
    if not validation.get("match") and validation.get("confidence", 0) > min_confidence:
        return ({"error": f"Query and operation type mismatch. Query: {user_query}", "ai": validation}, 409)
    return None

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

def normalize_operation_type(operation_type: str) -> str:
    if not operation_type:
        return ""
    return operation_type.replace("_", "-").lower()


def deep_merge_parameters(base: dict, updates: dict) -> dict:
    if base is None and updates is None:
        return {}
    if base is None:
        return copy.deepcopy(updates)
    if updates is None:
        return copy.deepcopy(base)

    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if value is None:
            continue
        # Lists should not be deep-merged. Most operation payloads contain lists where
        # positional/semantic merging is ambiguous (e.g., operations[], replacements[],
        # sort_config[], filter_config[]). When the user submits a follow-up correction,
        # the LLM often outputs a partial/updated list. Deep-merging lists causes
        # "pollution" from previous outputs (like treating filler words as values).
        #
        # So: if the new value is a list, treat it as authoritative and overwrite.
        if isinstance(value, list):
            merged[key] = copy.deepcopy(value)
            continue

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_parameters(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def is_followup_revision(original_request_id: str, previous_query: str, current_query: str) -> bool:
    """Decide whether to treat this request as a follow-up revision.

    Design goal: do NOT hardcode linguistic markers. If the client supplies an
    `aiRequestId`, they are explicitly chaining this request to a previous one.
    In that case, parameter *merging* is risky for most operations (payloads contain
    lists where semantic alignment is ambiguous), so we treat the new query as a
    revision and re-derive parameters based on conversation context.

    This keeps the behavior generic and lets the model interpret the correction.
    """
    if not original_request_id:
        return False
    if not previous_query or not current_query:
        return False

    prev = previous_query.strip()
    curr = current_query.strip()
    if not prev or not curr:
        return False

    # If the text changed, assume it's a revision. If it didn't change, it's a retry.
    return prev != curr


def merge_with_previous_parameters(current_operation: str, result: dict, previous_operation: str, previous_parameters: dict) -> dict:
    if not result or not previous_parameters:
        return result

    if normalize_operation_type(current_operation) != normalize_operation_type(previous_operation):
        return result

    current_params = result.get("parameters")
    if not current_params:
        result["parameters"] = copy.deepcopy(previous_parameters)
        return result

    result["parameters"] = deep_merge_parameters(previous_parameters, current_params)
    return result


def _check_column(column_name: str, available_columns: set, missing: set):
    if not column_name:
        return
    if isinstance(column_name, str) and column_name not in available_columns:
        missing.add(column_name)


def find_missing_columns_for_add_column(params: dict, available_columns: set):
    if not params:
        return set()

    available_set = available_columns or set()
    available_lookup = _build_column_lookup(available_set)
    missing = set()

    op_type = params.get("operationType")
    source_column = params.get("sourceColumn")
    _check_column_ci(source_column, available_lookup, missing)

    if op_type == "calculate":
        for step in params.get("operations", []) or []:
            # calculate schema: each step references column1 and (column2 | fixed_value)
            _check_column_ci(step.get("column1"), available_lookup, missing)
            col2 = step.get("column2")
            if isinstance(col2, str):
                _check_column_ci(col2, available_lookup, missing)
    elif op_type == "concatenate":
        # NOTE: The LLM may output either:
        # 1) the intended concatenate schema (list of ConcatStep objects)
        # 2) a calculate-style schema (column1/column2/operator) even when the user asks to concatenate.
        # We validate both so missing columns are always surfaced.
        for step in params.get("operations", []) or []:
            if not isinstance(step, dict):
                continue

            # Case 1: proper concatenate step
            if "column" in step:
                _check_column_ci(step.get("column"), available_lookup, missing)
                continue

            # Case 2: calculate-like step
            if "column1" in step:
                _check_column_ci(step.get("column1"), available_lookup, missing)
            col2 = step.get("column2")
            if isinstance(col2, str):
                _check_column_ci(col2, available_lookup, missing)
    elif op_type == "conditional":
        for condition in params.get("conditions", []) or []:
            _check_column_ci(condition.get("column"), available_lookup, missing)

    return missing


def find_missing_columns_for_sort_filter(params: dict, available_columns: set):
    missing = set()
    if not params:
        return missing

    available_set = available_columns or set()
    for cfg in params.get("sort_config", []) or []:
        _check_column(cfg.get("column"), available_set, missing)
    for cfg in params.get("filter_config", []) or []:
        _check_column(cfg.get("column"), available_set, missing)
    return missing


def find_missing_columns_for_group_pivot(params: dict, available_columns: set):
    missing = set()
    if not params:
        return missing

    available_set = available_columns or set()
    for col in params.get("rowIndex", []) or []:
        _check_column(col, available_set, missing)
    _check_column(params.get("columnIndex"), available_set, missing)
    for pivot in params.get("pivotValues", []) or []:
        _check_column(pivot.get("column"), available_set, missing)
    return missing


def find_missing_columns_for_merge(params: dict, columns_table1: set, columns_table2: set):
    missing = set()
    if not params:
        return missing

    available1 = columns_table1 or set()
    available2 = columns_table2 or set()
    key_pairs = params.get("keyPairs", []) or []
    for pair in key_pairs:
        _check_column(pair.get("left"), available1, missing)
        _check_column(pair.get("right"), available2, missing)
    return missing


def find_missing_columns_for_reconcile(params: dict, columns_table1: set, columns_table2: set):
    missing = set()
    if not params:
        return missing

    available1 = columns_table1 or set()
    available2 = columns_table2 or set()

    for key in params.get("keys", []) or []:
        _check_column(key.get("left"), available1, missing)
        _check_column(key.get("right"), available2, missing)

    for value in params.get("values", []) or []:
        _check_column(value.get("left"), available1, missing)
        _check_column(value.get("right"), available2, missing)

    cross_reference = params.get("crossReference", {}) or {}
    for ref in cross_reference.get("left", []) or []:
        _check_column(ref, available1, missing)
    for ref in cross_reference.get("right", []) or []:
        _check_column(ref, available2, missing)
    return missing


def find_missing_columns_for_replace(params: dict, available_columns: set):
    missing = set()
    if not params:
        return missing

    available_set = available_columns or set()
    for operation in params.get("operations", []) or []:
        op_type = operation.get("type")
        if op_type == "rename_columns":
            for col in (operation.get("mapping", {}) or {}).keys():
                _check_column(col, available_set, missing)
        elif op_type == "reorder_columns":
            for col in operation.get("order", []) or []:
                _check_column(col, available_set, missing)
        elif op_type == "replace_values":
            for replacement in operation.get("replacements", []) or []:
                _check_column(replacement.get("column"), available_set, missing)
    return missing


def find_missing_columns_for_format(params: dict, available_columns: set):
    missing = set()
    if not params:
        return missing

    available_set = available_columns or set()
    for config in params.get("formattingConfigs", []) or []:
        location = (config.get("location") or {}).get("range")
        if location:
            for token in location.split(","):
                col_name = token.strip()
                if not col_name:
                    continue
                _check_column(col_name, available_set, missing)
    return missing


def collect_missing_columns(operation_type: str, params: dict, metadata: dict, metadata2: dict = None):
    normalized = normalize_operation_type(operation_type)
    columns_table1 = set((metadata or {}).get("columns", []) or [])
    columns_table2 = set((metadata2 or {}).get("columns", []) or [])

    if normalized == "add-column":
        return find_missing_columns_for_add_column(params, columns_table1)
    if normalized == "sort-filter":
        return find_missing_columns_for_sort_filter(params, columns_table1)
    if normalized == "group-pivot":
        return find_missing_columns_for_group_pivot(params, columns_table1)
    if normalized == "merge-files":
        return find_missing_columns_for_merge(params, columns_table1, columns_table2)
    if normalized == "reconcile":
        return find_missing_columns_for_reconcile(params, columns_table1, columns_table2)
    if normalized == "replace-rename-reorder":
        return find_missing_columns_for_replace(params, columns_table1)
    if normalized == "format":
        return find_missing_columns_for_format(params, columns_table1)
    return set()


def validate_operation_columns_or_abort(operation_type: str, params: dict, metadata: dict, metadata2: dict = None):
    missing = collect_missing_columns(operation_type, params, metadata, metadata2)
    if missing:
        print(f"[DEBUG] Missing columns for {operation_type}: {sorted(missing)}")
        return jsonify({"error": COLUMN_NOT_FOUND_MESSAGE}), 409
    return None


def validate_chain_result_or_abort(result: dict):
    """Normalize run_chain failures into API-friendly responses.

    Returns None when ok, otherwise a (json, status).
    """
    if not isinstance(result, dict):
        return (jsonify({"error": "Error processing request"}), 500)
    if result.get("success") is True:
        return None

    # Standardize incomplete prompt
    err = result.get("error")
    if err == PROMPT_INCOMPLETE_MESSAGE:
        return (jsonify({"error": PROMPT_INCOMPLETE_MESSAGE}), 409)

    # Default: treat as prompt/LLM failure
    status = 400
    return (jsonify({"error": err or "Error processing request"}), status)

@nlp_bp.route("/process", methods=["POST"])
def process_natural_language():
    """Process natural language query with DataFrame context"""
    # Initialize variables at the top level
    ai_request = None
    result = None
    previous_request_query = None
    previous_request_context = None
    previous_operation_type = None
    previous_request_parameters = None
    is_followup = False
    
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
                    previous_request_query = ai_request.query
                    if ai_request.response:
                        previous_request_context = {
                            'query': ai_request.query,
                            'operation_type': ai_request.operation_type,
                            'response': ai_request.response
                        }
                        previous_operation_type = ai_request.operation_type
                        previous_request_parameters = (ai_request.response or {}).get('parameters')
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
            
            # Fetch conversation history for context (last 5 successful AI requests in this process)
            conversation_history = []
            try:
                previous_requests = db.session.query(AIRequest).filter(
                    AIRequest.process_id == data.get("process_id"),
                    AIRequest.status == 'success',
                    AIRequest.id != ai_request.id  # Exclude current request
                ).order_by(AIRequest.request_time.desc()).limit(5).all()
                
                conversation_history = [{
                    'query': req.query,
                    'operation_type': req.operation_type,
                    'response': req.response
                } for req in reversed(previous_requests)]  # Reverse to get chronological order
                
                if previous_request_context:
                    conversation_history.append(previous_request_context)
                print(f"[DEBUG] Loaded {len(conversation_history)} previous conversations for context")
            except Exception as hist_err:
                print(f"[DEBUG] Failed to load conversation history: {str(hist_err)}")
                conversation_history = []
                
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

        user_query = data.get("query", "")
        if ai_request:
            ai_request.query = user_query
            ai_request.operation_type = data.get('operation_type')
            ai_request.table_name = data.get('tableName')
            ai_request.second_table_name = data.get('table2Name')
            try:
                db.session.flush()
            except Exception as update_err:
                print(f"[DEBUG] Failed to persist updated AI request context: {update_err}")
                db.session.rollback()

        validation_query = user_query
        if original_request_id and previous_request_query:
            validation_query = f"{previous_request_query}\nFollow-up: {user_query}"
            is_followup = is_followup_revision(original_request_id, previous_request_query, user_query)

        # Validate required fields based on operation type
        if data.get("operation_type") in ["merge-files", "merge_files"]:
            required_fields = ["query", "operation_type", "process_id", "table1Name", "table2Name", "output_table_name"]
        elif data.get("operation_type") == "reconcile":
            required_fields = ["query", "operation_type", "process_id", "sourceTableNames", "output_table_name"]
        else:
            required_fields = ["query", "operation_type", "tableName", "process_id", "output_table_name"]

        if not all(k in data for k in required_fields):
            print(f"[DEBUG] Missing fields. Received: {list(data.keys())}")
            return jsonify({
                "error": f"Missing required fields. Required: {required_fields}"
            }), 400

        # Get optional output table name
        output_table_name = data.get("output_table_name")
        print(f"[DEBUG] Output table name: {output_table_name}")

                # Check if output table name already exists
        existing_df = DataFrame.query.filter_by(
            process_id=data.get('process_id'),
            name=output_table_name
        ).first()
        if existing_df:
            if existing_df.is_temporary == False:
                return jsonify({"error": f"Table with name {output_table_name} already exists."}), 409

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

                # Validate intent once using centralized helper
                err = validate_intent_or_abort(validation_query, operation_type, transformed_metadata)
                if err:
                    # Return a concise, actionable error including the user's query and the requested operation
                    return jsonify({
                        "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                    }), 409

                # Call run_chain for merge operation
                result = run_chain(
                    user_input=data["query"],
                    operation_type=operation_type,
                    table_name=data["table1Name"],
                    process_id=data["process_id"],
                    dataframe_metadata=transformed_metadata,
                    table2_metadata=transformed_metadata2,
                    conversation_history=conversation_history
                )
                chain_err = validate_chain_result_or_abort(result)
                if chain_err:
                    return chain_err
                if not is_followup:
                    result = merge_with_previous_parameters(
                        operation_type,
                        result,
                        previous_operation_type,
                        previous_request_parameters
                    )
                column_validation_error = validate_operation_columns_or_abort(
                    operation_type,
                    result.get("parameters"),
                    transformed_metadata,
                    transformed_metadata2
                )
                if column_validation_error:
                    return column_validation_error
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

            # Validate intent once using centralized helper
            err = validate_intent_or_abort(validation_query, "reconcile", transformed_metadata)
            if err:
                # Return a concise, actionable error including the user's query and the requested operation
                return jsonify({
                    "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                }), 409

            # Call run_chain with transformed metadata
            result = run_chain(
                user_input=data["query"],
                operation_type=data["operation_type"],
                table_name=source_table_names[0],
                process_id=data["process_id"],
                dataframe_metadata=transformed_metadata,
                table2_metadata=transformed_metadata2,
                conversation_history=conversation_history
            )
            chain_err = validate_chain_result_or_abort(result)
            if chain_err:
                return chain_err
            if not is_followup:
                result = merge_with_previous_parameters(
                    data["operation_type"],
                    result,
                    previous_operation_type,
                    previous_request_parameters
                )
            column_validation_error = validate_operation_columns_or_abort(
                data["operation_type"],
                result.get("parameters"),
                transformed_metadata,
                transformed_metadata2
            )
            if column_validation_error:
                return column_validation_error
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
                            print(f"[DEBUG] Missing left key column '{key.get('left')}' in first table columns: {actual_columns1}")
                            return jsonify({
                                "error": COLUMN_NOT_FOUND_MESSAGE
                            }), 409
                        if key.get('right') not in actual_columns2:
                            print(f"[DEBUG] Missing right key column '{key.get('right')}' in second table columns: {actual_columns2}")
                            return jsonify({
                                "error": COLUMN_NOT_FOUND_MESSAGE
                            }), 409

                # Validate values
                for value in result["parameters"].get("values", []):
                    if isinstance(value, dict):
                        if value.get('left') not in actual_columns1:
                            print(f"[DEBUG] Missing left value column '{value.get('left')}' in first table columns: {actual_columns1}")
                            return jsonify({
                                "error": COLUMN_NOT_FOUND_MESSAGE
                            }), 409
                        if value.get('right') not in actual_columns2:
                            print(f"[DEBUG] Missing right value column '{value.get('right')}' in second table columns: {actual_columns2}")
                            return jsonify({
                                "error": COLUMN_NOT_FOUND_MESSAGE
                            }), 409

                # Validate cross-reference columns
                cross_reference = result["parameters"].get("crossReference", {})
                if isinstance(cross_reference, dict):
                    for ref in cross_reference.get("left", []):
                        if ref not in actual_columns1:
                            print(f"[DEBUG] Missing cross-reference column '{ref}' in first table columns: {actual_columns1}")
                            return jsonify({
                                "error": COLUMN_NOT_FOUND_MESSAGE
                            }), 409
                    for ref in cross_reference.get("right", []):
                        if ref not in actual_columns2:
                            print(f"[DEBUG] Missing cross-reference column '{ref}' in second table columns: {actual_columns2}")
                            return jsonify({
                                "error": COLUMN_NOT_FOUND_MESSAGE
                            }), 409

            # Create DataFrameOperation record
            operation_message = f"Reconciling tables {request_source_table_names} with output '{output_table_name}'"
            df_operation = DataFrameOperation(
                process_id=data["process_id"],
                dataframe_id=source_dfs[0].id,  # Using first dataframe's ID
                operation_type=OperationType.RECONCILE_FILES.value,
                operation_subtype="reconcile",
                payload=result["parameters"],
                message=operation_message,
                title="Reconcile Tables"
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
                    detailed_error = f"Key column '{key['left']}' not found in first table. Available columns: {actual_columns1}"
                    print(f"[DEBUG] {detailed_error}")
                    df_operation.set_error(detailed_error)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": COLUMN_NOT_FOUND_MESSAGE,
                        "operationId": df_operation.id
                    }), 400
                if key['right'] not in actual_columns2:
                    detailed_error = f"Key column '{key['right']}' not found in second table. Available columns: {actual_columns2}"
                    print(f"[DEBUG] {detailed_error}")
                    df_operation.set_error(detailed_error)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": COLUMN_NOT_FOUND_MESSAGE,
                        "operationId": df_operation.id
                    }), 400

            for value in values:
                if value['left'] not in actual_columns1:
                    detailed_error = f"Value column '{value['left']}' not found in first table. Available columns: {actual_columns1}"
                    print(f"[DEBUG] {detailed_error}")
                    df_operation.set_error(detailed_error)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": COLUMN_NOT_FOUND_MESSAGE,
                        "operationId": df_operation.id
                    }), 400
                if value['right'] not in actual_columns2:
                    detailed_error = f"Value column '{value['right']}' not found in second table. Available columns: {actual_columns2}"
                    print(f"[DEBUG] {detailed_error}")
                    df_operation.set_error(detailed_error)
                    db.session.commit()
                    return jsonify({
                        "success": False,
                        "error": COLUMN_NOT_FOUND_MESSAGE,
                        "operationId": df_operation.id
                    }), 400

            # Add debug logging for reconciliation parameters
            print(f"[DEBUG] Reconciliation parameters:")
            print(f"[DEBUG] Keys: {json.dumps(keys, indent=2)}")
            print(f"[DEBUG] Values: {json.dumps(values, indent=2)}")
            print(f"[DEBUG] Settings: {json.dumps(result['parameters'].get('settings', {}), indent=2)}")
            print(f"[DEBUG] Cross reference: {json.dumps(result['parameters'].get('crossReference', {}), indent=2)}")

            output_table_name1 = result["parameters"].get("outputTableName1") or f"{output_table_name}_left"
            output_table_name2 = result["parameters"].get("outputTableName2") or f"{output_table_name}_right"
            operation_result = process_dataframe_reconciliation(
                email=email,
                process_id=data["process_id"],
                source_dfs=source_dfs,
                keys=keys,
                values=values,
                settings=result["parameters"].get('settings', {}),
                cross_reference=result["parameters"].get('crossReference', {}),
                output_table_name1=output_table_name1,
                output_table_name2=output_table_name2,
                existing_df_1=None,
                existing_df_2=None
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
                    metadata_path1 = f"{email}/process/{data['process_id']}/metadata/{output_table_name1}.json"
                    metadata_blob1 = bucket.blob(metadata_path1)
                    if metadata_blob1.exists():
                        updated_metadata1 = json.loads(metadata_blob1.download_as_string())
                    else:
                        updated_metadata1 = None
                    metadata_path2 = f"{email}/process/{data['process_id']}/metadata/{output_table_name2}.json"
                    metadata_blob2 = bucket.blob(metadata_path2)
                    if metadata_blob2.exists():
                        updated_metadata2 = json.loads(metadata_blob2.download_as_string())
                    else:
                        updated_metadata2 = None
                except Exception as e:
                    print(f"[DEBUG] Warning: Could not load updated metadata: {str(e)}")
                    updated_metadata1 = None
                    updated_metadata2 = None
                
                return jsonify({
                    "success": True,
                    "operation_details": {
                        "id": df_operation.id,
                        "sourceTables": [df.name for df in source_dfs],
                        "dataframeId": df_operation.dataframe_id,
                        "message": operation_result.get('message'),
                        "newTableName1": output_table_name1,
                        "newTableName2": output_table_name2,
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
                    "updated_metadata": {
                        "table1": updated_metadata1,
                        "table2": updated_metadata2
                    }
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
                    err = validate_intent_or_abort(validation_query, data["operation_type"], transformed_metadata)
                    if err:
                        # Return a concise, actionable error including the user's query and the requested operation
                        return jsonify({
                            "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                        }), 409

                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None,
                        conversation_history=conversation_history
                    )
                    chain_err = validate_chain_result_or_abort(result)
                    if chain_err:
                        return chain_err
                    if not is_followup:
                        result = merge_with_previous_parameters(
                            data["operation_type"],
                            result,
                            previous_operation_type,
                            previous_request_parameters
                        )
                    column_validation_error = validate_operation_columns_or_abort(
                        data["operation_type"],
                        result.get("parameters"),
                        transformed_metadata
                    )
                    if column_validation_error:
                        return column_validation_error
                    print(f"[DEBUG] Add column result: {result}")
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
                    result["parameters"]["output_table_name"] = f"{data['tableName']}_columnadded"
                
                # Fix operations format if needed
                if result["parameters"].get("operationType") == "calculate":
                    operations = result["parameters"].get("operations", [])
                    for op in operations:
                        # If column2 is a number, move it to fixed_value
                        if "column2" in op and isinstance(op["column2"], (int, float)):
                            op["fixed_value"] = op["column2"]
                            op["column2"] = None

                operation_message = f"Adding column '{result['parameters'].get('newColumnName', '')}' to table '{result['parameters'].get('tableName', '')}' with operation '{result['parameters'].get('operationType', '')}'"
                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.ADD_COLUMN.value,
                    operation_subtype=result["parameters"].get("operationType"),
                    payload=result["parameters"],
                    message=operation_message,
                    title="Add Column"
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
                    operation_params=result["parameters"],
                    output_table_name=result["parameters"].get("output_table_name")
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
                        metadata_path = f"{email}/process/{data['process_id']}/metadata/{output_table_name}.json"
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
                    err = validate_intent_or_abort(validation_query, data["operation_type"], transformed_metadata)
                    if err:
                        # Return a concise, actionable error including the user's query and the requested operation
                        return jsonify({
                            "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                        }), 409
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None,
                        conversation_history=conversation_history
                    )
                    # NOTE: For replace/rename/reorder, follow-up messages are commonly corrections
                    # (e.g., "actually" / "instead" / "sorry, it should be..."). Deep-merging
                    # list-heavy parameters (operations/replacements) across attempts leads to
                    # invalid combined outputs (e.g., replacing "sorry" -> "england").
                    # We rely on conversation_history + the prompt instructions to produce the
                    # *final* operations instead of attempting to merge.
                    column_validation_error = validate_operation_columns_or_abort(
                        data["operation_type"],
                        result.get("parameters"),
                        transformed_metadata
                    )
                    if column_validation_error:
                        return column_validation_error

                    if data["operation_type_new"] == "sort":
                        print(f"[DEBUG] Sort operation result: {result}")
                        if len(result.get("parameters").get("sort_config")) == 0:
                            return jsonify({"error": f"Query and operation type mismatch. Query : {data['query']} and Operation : {data['operation_type_new']}"}), 409
                    elif data["operation_type_new"] == "filter":
                        print(f"[DEBUG] Filter operation result: {result}")
                        if len(result.get("parameters").get("filter_config")) == 0:
                            return jsonify({"error": f"Query and operation type mismatch. Query : {data['query']} and Operation : {data['operation_type_new']}"}), 409                    
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

                # Generate message based on operation types (same format as non-AI path)
                message_parts = []
                sort_config = result["parameters"].get("sort_config", [])
                filter_config = result["parameters"].get("filter_config", [])
                table_name = data['tableName']
                
                if sort_config:
                    sort_columns = [f"'{config['column']}'" for config in sort_config]
                    message_parts.append(f"Sort table '{table_name}' on column(s) {', '.join(sort_columns)}")
                if filter_config:
                    filter_columns = [f"'{config['column']}'" for config in filter_config]
                    message_parts.append(f"Filter table '{table_name}' on column(s) {', '.join(filter_columns)}")
                
                operation_message = " and ".join(message_parts) if message_parts else f"Sort/filter on table '{table_name}'"
                
                # Determine title based on what operations are being performed
                if sort_config and filter_config:
                    operation_title = "Sort and Filter"
                elif sort_config:
                    operation_title = "Sort"
                elif filter_config:
                    operation_title = "Filter"
                else:
                    operation_title = "Sort/Filter"
                
                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.SORT_FILTER.value,
                    operation_subtype="sort_filter",
                    payload=result["parameters"],
                    message=operation_message,
                    title=operation_title
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
                    err = validate_intent_or_abort(validation_query, data["operation_type"], transformed_metadata)
                    if err:
                        # Return a concise, actionable error including the user's query and the requested operation
                        return jsonify({
                            "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                        }), 409
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None,
                        conversation_history=conversation_history
                    )
                    if not is_followup:
                        result = merge_with_previous_parameters(
                            data["operation_type"],
                            result,
                            previous_operation_type,
                            previous_request_parameters
                        )
                    column_validation_error = validate_operation_columns_or_abort(
                        data["operation_type"],
                        result.get("parameters"),
                        transformed_metadata
                    )
                    if column_validation_error:
                        return column_validation_error
                    
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

                operation_message = f"Group/pivot on table '{result['parameters'].get('tableName', '')}' to output '{output_table_name}'"
                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.GROUP_PIVOT.value,
                    operation_subtype="group_pivot",
                    payload=result["parameters"],
                    message=operation_message
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
                        
                        # Single centralized validation before generating merge parameters
                        err = validate_intent_or_abort(validation_query, data["operation_type"], transformed_metadata1)
                        if err:
                            # Return a concise, actionable error including the user's query and the requested operation
                            return jsonify({
                                "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                            }), 409

                        # Call run_chain to get merge parameters
                        result = run_chain(
                            user_input=data["query"],
                            operation_type="merge-files",
                            table_name=data["table1Name"],
                            process_id=data["process_id"],
                            dataframe_metadata=transformed_metadata1,
                            table2_metadata=transformed_metadata2
                        )
                        if not is_followup:
                            result = merge_with_previous_parameters(
                                data["operation_type"],
                                result,
                                previous_operation_type,
                                previous_request_parameters
                            )
                        column_validation_error = validate_operation_columns_or_abort(
                            data["operation_type"],
                            result.get("parameters"),
                            transformed_metadata1,
                            transformed_metadata2
                        )
                        if column_validation_error:
                            return column_validation_error
                        
                        if not result or not result.get("parameters"):
                            return jsonify({"error": "Failed to generate merge parameters"}), 500
                            
                    except Exception as e:
                        print(f"[DEBUG] Error getting metadata: {str(e)}")
                        return jsonify({"error": f"Error processing metadata: {str(e)}"}), 500

                # Use table names from request body
                table1_name = data["table1Name"]
                table2_name = data["table2Name"]
                output_table_name = data["output_table_name"]

                merge_method = result["parameters"].get("mergeMethod", "left")
                merge_type = result["parameters"].get("mergeType", "horizontal")
                if merge_type == "horizontal":
                    message = f"Merge tables {table1_name} and {table2_name} horizontally based on the {merge_method} method"
                elif merge_type == "vertical":
                    message = f"Merge tables {table1_name} and {table2_name} vertically"


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
                    payload=result["parameters"],
                    message=message
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
                    
                    err = validate_intent_or_abort(validation_query, data["operation_type"], transformed_metadata)
                    if err:
                        # Return a concise, actionable error including the user's query and the requested operation
                        return jsonify({
                            "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                        }), 409
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None,
                        conversation_history=conversation_history
                    )
                    if not is_followup:
                        result = merge_with_previous_parameters(
                            data["operation_type"],
                            result,
                            previous_operation_type,
                            previous_request_parameters
                        )
                    column_validation_error = validate_operation_columns_or_abort(
                        data["operation_type"],
                        result.get("parameters"),
                        transformed_metadata
                    )
                    if column_validation_error:
                        return column_validation_error
                    
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

                operation_message = f"Replace/rename/reorder on table '{result['parameters'].get('tableName', '')}' to output '{output_table_name}'"
                df_operation = DataFrameOperation(
                    process_id=data["process_id"],
                    dataframe_id=dataframe.id,
                    operation_type=OperationType.REPLACE_RENAME_REORDER.value,
                    operation_subtype="replace_rename_reorder",
                    payload=result["parameters"],
                    message=operation_message
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
                    
                    err = validate_intent_or_abort(validation_query, data["operation_type"], transformed_metadata)
                    if err:
                        # Return a concise, actionable error including the user's query and the requested operation
                        return jsonify({
                            "error": f"Query and operation type mismatch. Query: {data['query']}, operationType: {data['operation_type']}",
                        }), 409
                    
                    # Call run_chain with transformed metadata
                    result = run_chain(
                        user_input=data["query"],
                        operation_type=data["operation_type"],
                        table_name=data["tableName"],
                        process_id=data["process_id"],
                        dataframe_metadata=transformed_metadata,
                        table2_metadata=None,
                        conversation_history=conversation_history
                    )
                    if not is_followup:
                        result = merge_with_previous_parameters(
                            data["operation_type"],
                            result,
                            previous_operation_type,
                            previous_request_parameters
                        )
                    column_validation_error = validate_operation_columns_or_abort(
                        data["operation_type"],
                        result.get("parameters"),
                        transformed_metadata
                    )
                    if column_validation_error:
                        return column_validation_error
                    
                    if not result.get("success"):
                        print(f"[DEBUG] run_chain error: {result.get('error')}")
                        return jsonify({"error": result.get("error")}), 400
                        
                except Exception as e:
                    print(f"[DEBUG] Metadata processing error: {str(e)}")
                    return jsonify({"error": f"Error processing metadata: {str(e)}"}), 500

                if not result or not result.get("parameters"):
                    return jsonify({"error": "Failed to generate format parameters"}), 500

                # Update parameters with output table name if provided
                if output_table_name:
                    result["parameters"]["outputTableName"] = output_table_name
                elif not result["parameters"].get("outputTableName"):
                    result["parameters"]["outputTableName"] = f"{data['tableName']}_formatted"

                # Just return the AI-selected parameters without processing
                return jsonify({
                    "success": True,
                    "operation_details": {
                        "sourceTable": result["parameters"].get("tableName"),
                        "dataframeId": dataframe.id,
                        "message": "Format parameters selected by AI",
                        "newTableName": result["parameters"].get("outputTableName"),
                        "aiRequestId": ai_request.id
                    },
                    "parameters": result["parameters"],
                    "metadata_used": result.get("metadata_used"),
                    "domain": result.get("domain")
                })

            except Exception as e:
                print(f"[DEBUG] Error in format processing: {str(e)}")
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

        # For other operation types, return the original result
        print("[DEBUG] Returning successful response")
        return jsonify(result)

    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # Rollback the session before attempting to set error
        db.session.rollback()
        
        try:
            # Store error in AI request if it exists
            if ai_request:
                ai_request.status = 'error'
                ai_request.error_message = str(e)
                ai_request.response_time = datetime.now(timezone.utc)
                # Skip processing duration calculation to avoid timezone issues
                db.session.commit()
        except Exception as inner_e:
            print(f"[DEBUG] Error setting error status: {str(inner_e)}")
            db.session.rollback()
            
        return jsonify({"error": str(e)}), 500

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
            result = asyncio.run(run_chain(
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
            ))

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