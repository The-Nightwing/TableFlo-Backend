from flask import Flask, jsonify, request, send_from_directory, Response
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from routes import register_user, login, forgot_password, reset_password, validate_otp_bp, validate_forgot_password_otp_bp
from Services.merge_files import merge_files
from setup import create_app
from werkzeug.middleware.proxy_fix import ProxyFix
from firebase_config import get_firestore_client, get_storage_bucket
from file_upload import file_processing_bp
from werkzeug.middleware.proxy_fix import ProxyFix
from models import User, File
from firebase_files import firebase_files_bp
from file_details import file_details_bp
from edit_file import edit_file_bp
from file_preview import preview_bp
from merge_files import merge_files_bp
from group_pivot import group_pivot_bp
from sort_filter import sort_filter_bp
from add_column import add_column_bp
import pymysql
import os
from formatting import formatting_bp
from visualization import visualization_bp
from file_operations import file_operations
from process import process_bp
from run_process import run_process_bp
from app import nlp_bp


# Initialize application
application = create_app()
# Set up CORS configuration globally
cors_origin = [
    "https://main.d3td5jj1bvp0sj.amplifyapp.com",
]


CORS(application, resources={r"/*": {"origins": "*"}})

application.wsgi_app = ProxyFix(application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

@application.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = jsonify({"message": "Preflight request handled"})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust origin as needed
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, PATCH')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-User-Email')
        #response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

# # Initialize DB (Make sure this is before routes, so DB is available)
# Register Blueprints
application.register_blueprint(register_user)
application.register_blueprint(login)
application.register_blueprint(forgot_password)
application.register_blueprint(reset_password)
application.register_blueprint(validate_forgot_password_otp_bp)
application.register_blueprint(validate_otp_bp)
application.register_blueprint(file_processing_bp)  # This will make the URL for uploading files '/api/upload'
application.register_blueprint(firebase_files_bp)
application.register_blueprint(file_details_bp)
application.register_blueprint(edit_file_bp)
application.register_blueprint(preview_bp)
application.register_blueprint(merge_files_bp)
application.register_blueprint(group_pivot_bp)
application.register_blueprint(sort_filter_bp)
application.register_blueprint(add_column_bp)
application.register_blueprint(formatting_bp)
application.register_blueprint(visualization_bp)
application.register_blueprint(file_operations)
application.register_blueprint(process_bp)
application.register_blueprint(run_process_bp)
application.register_blueprint(nlp_bp)
@application.route('/api/user/', methods=['GET'])
def get_user_details():
    email = request.headers.get('X-User-Email')
    if not email:
            return jsonify({"error": "Email is required in headers"}), 400

    # Query the database for the user with the provided email
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Fetch the upload history for the user
    uploaded_files = File.query.filter_by(email=email).all()
    file_history_data = [file.to_dict() for file in uploaded_files]


    # Return the user data along with the upload history
    user_data = {
         "userId" : user.id,
        "name": user.name,
        "email": user.email,
        "company": user.company_name,
        "uploadHistory": file_history_data
    }

    return jsonify(user_data)


@application.route('/.well-known/pki-validation/4673D506424724DC3E3FD3218174EB47.txt')
def serve_auth_file():
    auth_text = "72478CA2DF37B3E1C1C4E4886E8FF5A44929D1837B9020FA5791A039F450840D\ncomodoca.com\nc7264e9628072c4"
    return Response(auth_text, mimetype='text/plain')

if __name__ == '__main__':
    application.run(debug=True, port = 5001)
