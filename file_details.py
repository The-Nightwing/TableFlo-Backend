from flask import Blueprint, request, jsonify
from firebase_config import get_storage_bucket
import json
import os
from models import File, UserProcess, User

file_details_bp = Blueprint('file_details', __name__, url_prefix='/api/')

@file_details_bp.route('/get-file-details/', methods=['GET'])
def get_file_details():
    """
    Retrieve metadata (sheets, columns, and data types) of a file from Firebase Storage.
    """
    file_id = request.args.get('fileId')
    file_name = request.args.get('fileName')  # Keep for backward compatibility

    if not file_id and not file_name:
        return jsonify({'error': 'Either fileId or fileName is required'}), 400

    bucket = get_storage_bucket()
    user_email = request.headers.get('X-User-Email')
    if not user_email:
        return jsonify({'error': 'Email header is required'}), 400

    # Look up file in database
    file = None
    if file_id:
        file = File.query.filter_by(id=file_id, email=user_email).first()
    else:
        file = File.query.filter_by(file_name=file_name, email=user_email).first()

    if not file:
        return jsonify({'error': 'File not found'}), 404

    # Get metadata using file ID
    metadata_path = f"{user_email}/metadata/{file.id}.json"
    print(f"Metadata path: {metadata_path}")
    blob = bucket.blob(metadata_path)

    if not blob.exists():
        # Try legacy path as fallback
        legacy_path = f"{user_email}/metadata/{file.file_name}.json"
        blob = bucket.blob(legacy_path)
        if not blob.exists():
            return jsonify({'error': f'Metadata not found'}), 404

    try:
        metadata_content = blob.download_as_text()
        metadata = json.loads(metadata_content)
        print(f"Metadata content: {metadata}")

        if 'error' in metadata:
            error_message = metadata['error']
            if "File is not a zip file" in error_message:
                return jsonify({
                    'error': 'This appears to be an older Excel file format (.xls). Please convert it to .xlsx format and try again.'
                }), 400
            return jsonify({'error': error_message}), 400

        response = {
            'id': file.id,  # Include ID in response
            'fileName': file.file_name,
            'fileType': file.file_type,
            'uploadDate': file.upload_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if file.file_type == 'Excel':
            sheets_data = metadata.get('sheets', {})
            response['sheets'] = {}
            for sheet_name, sheet_data in sheets_data.items():
                response['sheets'][sheet_name] = {
                    'columns': sheet_data.get('columns', []),
                    'columnTypes': sheet_data.get('columnTypes', {}),
                    'rowCount': sheet_data.get('rowCount', 0)
                }
        elif file.file_type == 'CSV':
            response['sheets'] = {
                'Sheet1': {
                    'columns': metadata.get('columns', []),
                    'columnTypes': metadata.get('columnTypes', {}),
                    'rowCount': metadata.get('rowCount', 0)
                }
            }

        return jsonify(response)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return jsonify({'error': 'Failed to parse metadata JSON'}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Failed to retrieve metadata: {str(e)}'}), 500

@file_details_bp.route('/get-table-details/', methods=['GET'])
def get_table_details():
    """
    Retrieve metadata of a table from a process folder in Firebase Storage.
    """
    try:
        process_id = request.args.get('processId')
        table_name = request.args.get('tableName')
        user_email = request.headers.get('X-User-Email')

        # Validate required parameters
        if not all([process_id, table_name]):
            return jsonify({'error': 'Process ID and table name are required'}), 400
        if not user_email:
            return jsonify({'error': 'Email header is required'}), 400

        # First get the user by email
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get the process using user_id
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({'error': 'Process not found or access denied'}), 404

        # Construct the metadata path using process ID
        metadata_path = f"{user_email}/process/{process.id}/metadata/{table_name}.json"
        print(f"Metadata path: {metadata_path}")

        # Get the metadata from Firebase
        bucket = get_storage_bucket()
        blob = bucket.blob(metadata_path)

        if not blob.exists():
            return jsonify({'error': f'Metadata for table "{table_name}" not found'}), 404

        try:
            # Download and parse the metadata JSON
            metadata_content = blob.download_as_text()
            metadata = json.loads(metadata_content)
            print(f"Metadata content: {metadata}")

            # Handle the case where metadata contains an error
            if 'error' in metadata:
                return jsonify({'error': metadata['error']}), 400

            # Prepare the response
            response = {
                'processId': process_id,
                'tableName': metadata.get('tableName'),
                'description': metadata.get('description', ''),
                'sourceFile': metadata.get('sourceFile'),
                'sourceSheet': metadata.get('sourceSheet'),
                'createdAt': metadata.get('createdAt'),
                'rowCount': metadata.get('rowCount', 0),
                'columnCount': metadata.get('columnCount', 0),
                'columns': metadata.get('columns', []),
            }

            return jsonify(response)

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            return jsonify({'error': 'Failed to parse metadata JSON'}), 500

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Failed to retrieve table metadata: {str(e)}'}), 500
