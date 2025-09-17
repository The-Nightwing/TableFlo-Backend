from flask import Blueprint, request, jsonify
from firebase_config import get_storage_bucket
from firebase_admin import auth
from models import File, db
from models import User
from concurrent.futures import ThreadPoolExecutor, as_completed

firebase_files_bp = Blueprint('firebase_files', __name__, url_prefix='/api/list-files/')

@firebase_files_bp.route('/', methods=['GET'])
def list_files():
    email = request.headers.get('X-User-Email')
    if not email:
        return jsonify({'error': 'Email header is required'}), 400

    try:
        user = auth.get_user_by_email(email)
        user_uid = user.uid
    except Exception as e:
        return jsonify({'error': 'Failed to retrieve user', 'details': str(e)}), 500

    try:
        db_files = File.query.filter_by(email=email).all()
        if not db_files:
            return jsonify({'files': [], 'totalCount': 0, 'message': 'No files found for this user'})

        bucket = get_storage_bucket()

        # Fetch existing blobs just once
        existing_blobs = {blob.name for blob in bucket.list_blobs(prefix=f'{email}/uploaded_files/')}

        def generate_file_info(db_file):
            blob_path = f'{email}/uploaded_files/{db_file.file_name}'
            if blob_path not in existing_blobs:
                return None
            blob = bucket.blob(blob_path)
            try:
                download_url = blob.generate_signed_url(expiration=604800, version='v4')
            except Exception:
                return None
            return {
                'id': db_file.id,
                'userId': db_file.user_id,
                'fileName': db_file.file_name,
                'fileType': db_file.file_type,
                'uploadTime': db_file.upload_time.strftime("%Y-%m-%d %H:%M:%S") if db_file.upload_time else None,
                'downloadUrl': download_url,
            }

        file_list = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_file_info, f) for f in db_files]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    file_list.append(result)

        return jsonify({
            'files': file_list,
            'totalCount': len(file_list),
            'message': 'Files retrieved successfully'
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to list files',
            'details': str(e),
            'files': [],
            'totalCount': 0
        }), 500


@firebase_files_bp.route('/processed/', methods=['GET'])
def list_processed_files():
    """
    Endpoint to list all processed files in a Firebase Storage folder based on the user's email from the headers.
    """
    email = request.headers.get('X-User-Email')  # Retrieve email from request headers
    if not email:
        return jsonify({'error': 'Email header is required'}), 400

    # Retrieve user UID using email
    try:
        user = auth.get_user_by_email(email)
        user_uid = user.uid
    except auth.UserNotFoundError:
        return jsonify({'error': 'User not found with the provided email'}), 404
    except Exception as e:
        return jsonify({'error': 'Failed to retrieve user', 'details': str(e)}), 500

    # Define folder path based on user email
    folder_path = f'{email}/process/'

    try:
        bucket = get_storage_bucket()
        blobs = bucket.list_blobs(prefix=folder_path)

        file_list = []
        for blob in blobs:
            # Skip the folder itself if listed
            if blob.name == folder_path or blob.name.endswith('/'):
                continue

            # Generate signed URL for downloading the file
            try:
                download_url = blob.generate_signed_url(expiration=3600, version='v4')
            except Exception as e:
                return jsonify({'error': 'Failed to generate signed URL', 'details': str(e)}), 500

            file_list.append({
                'fileName': blob.name.split('/')[-1],  # Extract file name
                "fileType": blob.content_type,
                "uploadTime": blob.time_created.isoformat() if blob.time_created else None,
                'downloadUrl': download_url
            })

        # Return sorted files in descending order of upload time
        sorted_files = sorted(file_list, key=lambda x: x.get('uploadTime', ''), reverse=True)

        return jsonify({'email': email, 'files': sorted_files})
    except Exception as e:
        return jsonify({'error': 'Failed to list files', 'details': str(e)}), 500

@firebase_files_bp.route('/<file_id>/delete', methods=['POST'])
def delete_file(file_id):
    """
    Endpoint to delete a file from Firebase Storage and Files table using file ID.
    """
    try:
        email = request.headers.get('X-User-Email')
        if not email:
            return jsonify({'error': 'Email header is required'}), 400

        # Get user and verify
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get File record
        file = File.query.filter_by(id=file_id, user_id=user.id).first()
        if not file:
            return jsonify({'error': 'File not found or access denied'}), 404

        # Store file details for response
        file_info = {
            "id": file.id,
            "fileName": file.file_name,
            "fileType": file.file_type,
            "uploadDate": file.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fileUuid": file.file_uuid,
            "userId": file.user_id
        }

        try:
            # Delete file from Firebase Storage
            bucket = get_storage_bucket()
            
            # Construct paths for all possible file locations
            possible_paths = [
                f"{email}/uploaded_files/{file.file_name}",
                f"{email}/processed_files/{file.file_name}",
                f"{email}/metadata/{file.file_name}.json",
                f"{email}/previews/{file.file_name}_preview.csv"
            ]

            # Try to delete each possible file
            for path in possible_paths:
                blob = bucket.blob(path)
                if blob.exists():
                    blob.delete()

        except Exception as storage_error:
            print(f"Warning: Error cleaning up storage for file {file_id}: {str(storage_error)}")

        # Delete File record
        db.session.delete(file)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'File deleted successfully',
            'deletedFile': file_info
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in delete_file: {str(e)}")
        return jsonify({'error': str(e)}), 500
