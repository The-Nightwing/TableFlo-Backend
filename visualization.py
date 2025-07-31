from flask import Blueprint, request, jsonify
import pandas as pd
from io import BytesIO
from firebase_config import get_storage_bucket
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timezone
from urllib.parse import quote
from flask import current_app
from models import db, Visualization, User, UserProcess

visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/visualization/')

def load_data_from_firebase(bucket, email, file_name, sheet_name=None):
    """Load data from Firebase Storage."""
    file_path = f"{email}/uploaded_files/{file_name}"
    blob = bucket.blob(file_path)
    
    if not blob.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Download as bytes instead of text for Excel files
    content = blob.download_as_bytes()
    
    # Check file extension
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        # For Excel files
        df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
    else:
        # For CSV files
        df = pd.read_csv(BytesIO(content))
    
    return df

@visualization_bp.route('/preview/', methods=['POST'])
def preview_visualization():
    try:
        data = request.json
        email = request.headers.get('X-User-Email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        file_name = data.get('fileName')
        sheet_name = data.get('sheet')
        visualization_config = data.get('visualizationConfig')

        if not file_name or not visualization_config:
            return jsonify({"error": "Missing required parameters"}), 400

        # Load data from Firebase
        bucket = get_storage_bucket()
        df = load_data_from_firebase(bucket, email, file_name, sheet_name)

        # Create a temporary session for preview
        preview_session_id = f"preview_{email}_{int(time.time())}"
        
        # Store the data and config in Firebase for Streamlit to access
        session_data = {
            "email": email,
            "fileName": file_name,
            "sheetName": sheet_name,
            "visualizationConfig": visualization_config,
            "timestamp": datetime.now().isoformat(),
            "mode": "preview"  # Indicate this is a preview session
        }
        
        # Save session data to Firebase
        config_blob = bucket.blob(f"streamlit_sessions/{preview_session_id}/config.json")
        config_blob.upload_from_string(
            json.dumps(session_data),
            content_type='application/json'
        )

        # Save DataFrame to Firebase as CSV for Streamlit to access
        df_blob = bucket.blob(f"streamlit_sessions/{preview_session_id}/data.csv")
        df_blob.upload_from_string(
            df.to_csv(index=False),
            content_type='text/csv'
        )

        # Construct Streamlit URL for preview
        streamlit_base_url = current_app.config.get('STREAMLIT_URL', 'https://app-visualisation-smeratockwq4n2nhchmgge.streamlit.app')
        streamlit_url = (
            f"{streamlit_base_url}?"
            f"session_id={preview_session_id}&"
            f"mode=preview&"
            f"email={quote(email)}"
        )

        return jsonify({
            "success": True,
            "streamlit_url": streamlit_url,
            "session_id": preview_session_id,
            "columns": df.columns.tolist()  # Return columns for frontend reference
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@visualization_bp.route('/save/', methods=['POST'])
def save_visualization():
    try:
        data = request.json
        email = request.headers.get('X-User-Email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Save visualization config to Firebase
        bucket = get_storage_bucket()
        vis_config_path = f"{email}/visualizations/{data['fileName']}"
        vis_config_blob = bucket.blob(vis_config_path)
        vis_config_blob.upload_from_string(
            json.dumps(data['visualizationConfig']),
            content_type='application/json'
        )

        return jsonify({
            "success": True,
            "message": "Visualization saved successfully"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@visualization_bp.route('/load/', methods=['GET'])
def load_visualization():
    try:
        email = request.headers.get('X-User-Email')
        file_name = request.args.get('fileName')
        
        if not email or not file_name:
            return jsonify({"error": "Email and fileName are required"}), 400

        # Load visualization config from Firebase
        bucket = get_storage_bucket()
        vis_config_path = f"{email}/visualizations/{file_name}"
        vis_config_blob = bucket.blob(vis_config_path)
        
        if not vis_config_blob.exists():
            return jsonify({"error": "Visualization not found"}), 404

        config = json.loads(vis_config_blob.download_as_string())
        
        return jsonify({
            "success": True,
            "data": config
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@visualization_bp.route('/process/create/', methods=['POST'])
def create_process_visualization():
    """Endpoint to create a visualization entry for a process."""
    try:
        data = request.json
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected a JSON object."}), 400

        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        required_fields = ['processId', 'configuration']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        process_id = data.get('processId')
        configurations = data.get('configuration')
        description = data.get('description', '').strip() or None  # Handle empty description

        # Validate configuration format
        if not isinstance(configurations, (list, dict)):
            return jsonify({"error": "Configuration must be either a JSON object or an array of JSON objects"}), 400

        # Convert single configuration to list for uniform processing
        if isinstance(configurations, dict):
            configurations = [configurations]

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        try:
            # Create new visualization entry
            visualization = Visualization(
                process_id=process_id,
                configuration=configurations,
                message=description  # Save description as message (can be None)
            )
            
            db.session.add(visualization)
            db.session.commit()

            return jsonify({
                "success": True,
                "message": "Visualization created successfully",
                "visualization": {
                    **visualization.to_dict(),
                    'description': visualization.message  # Include message as description in response
                }
            })

        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": f"Failed to create visualization: {str(e)}"
            }), 500

    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@visualization_bp.route('/process/<visualization_id>/remove', methods=['DELETE'])
def delete_process_visualization(visualization_id):
    """Endpoint to delete a visualization entry."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Get user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get visualization and verify ownership through process
        visualization = Visualization.query.join(
            UserProcess, Visualization.process_id == UserProcess.id
        ).filter(
            Visualization.id == visualization_id,
            UserProcess.user_id == user.id
        ).first()

        if not visualization:
            return jsonify({"error": "Visualization not found or access denied"}), 404

        try:
            # Delete the visualization from database
            db.session.delete(visualization)
            db.session.commit()

            return jsonify({
                "success": True,
                "message": "Visualization deleted successfully",
                "deletedId": visualization_id
            })

        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": f"Failed to delete visualization: {str(e)}"
            }), 500

    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@visualization_bp.route('/process/<process_id>/visualizations', methods=['GET'])
def get_process_visualizations(process_id):
    """Get all visualizations for a process."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Get user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get all visualizations for the process
        visualizations = Visualization.query.filter_by(process_id=process_id).all()
        
        # Convert to dictionary format
        visualizations_data = [{
            **vis.to_dict(),
            'description': vis.message,  # Include message as description in response
            'title': "Visualization"  # Add title field
        } for vis in visualizations]

        return jsonify({
            "success": True,
            "process": {
                "id": process.id,
                "name": process.process_name
            },
            "visualizations": visualizations_data,
            "count": len(visualizations_data)
        })

    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@visualization_bp.route('/process/visualization/<visualization_id>', methods=['PUT'])
def update_process_visualization(visualization_id):
    """Endpoint to update a visualization configuration."""
    try:
        data = request.json
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected a JSON object."}), 400

        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Validate required parameters
        if 'configuration' not in data:
            return jsonify({"error": "Missing required field: configuration"}), 400

        # Get user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get visualization and verify ownership through process
        visualization = Visualization.query.join(
            UserProcess, Visualization.process_id == UserProcess.id
        ).filter(
            Visualization.id == visualization_id,
            UserProcess.user_id == user.id
        ).first()

        if not visualization:
            return jsonify({"error": "Visualization not found or access denied"}), 404

        try:
            # Update configuration and description
            visualization.configuration = data['configuration']
            description = data.get('description', '').strip() or None  # Handle empty description
            visualization.message = description  # Update message with description (can be None)
            visualization.updated_at = datetime.now(timezone.utc)
            
            db.session.add(visualization)
            db.session.commit()

            return jsonify({
                "success": True,
                "message": "Visualization configuration updated successfully",
                "visualization": {
                    **visualization.to_dict(),
                    'description': visualization.message  # Include message as description in response
                }
            })

        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": f"Failed to update visualization: {str(e)}"
            }), 500

    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@visualization_bp.route('/process/<visualization_id>', methods=['GET'])
def get_process_visualization(visualization_id):
    """Endpoint to get a visualization configuration."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required in the headers."}), 400

        # Get user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get visualization and verify ownership through process
        visualization = Visualization.query.join(
            UserProcess, Visualization.process_id == UserProcess.id
        ).filter(
            Visualization.id == visualization_id,
            UserProcess.user_id == user.id
        ).first()

        if not visualization:
            return jsonify({"error": "Visualization not found or access denied"}), 404

        return jsonify({
            "success": True,
            "message": "Visualization retrieved successfully",
            "visualization": {
                **visualization.to_dict(),
                'description': visualization.message  # Include message as description in response
            }
        })

    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500 