import os
import pandas as pd
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from config import Config

# Setup for UPLOAD and MERGED folders
UPLOAD_FOLDER = Config.UPLOAD_FOLDER
MERGED_FOLDER = Config.MERGED_FOLDER

# Create Blueprint for file merging
merge_files = Blueprint('merge_files', __name__, url_prefix='/api/merge-files')

@merge_files.route('/merge-files', methods=['POST'])
def merge_files_endpoint():
    try:
        if 'files' not in request.files:
            return jsonify({'message': 'No files provided'}), 400

        files = request.files.getlist('files')
        dfs = []

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Handling file type
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return jsonify({'message': f'Unsupported file type: {filename}'}), 400

            dfs.append(df)

        # Merge the DataFrames
        merged_df = pd.concat(dfs, ignore_index=True)

        # Saving merged file
        merged_filename = 'merged_output.xlsx'
        merged_path = os.path.join(MERGED_FOLDER, merged_filename)
        merged_df.to_excel(merged_path, index=False)

        return jsonify({
            'message': 'Files merged successfully',
            'downloadUrl': f'/api/merge-files/download/{merged_filename}'
        }), 200
    except Exception as e:
        return jsonify({'message': f'Error merging files: {e}'}), 500

@merge_files.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        file_path = os.path.join(MERGED_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'message': f'Error downloading file: {e}'}), 500
