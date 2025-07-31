from flask import Blueprint, request, jsonify
from models import User, OTP, TemporaryUser, BlacklistedToken
from auth import generate_token, decode_token
from services import send_otp_email
from Services.merge_files import merge_files_endpoint, download_file
from services import generate_otp, generate_short_uuid
from setup import create_app, db  # Import the create_app function and db instance
import datetime 
from firebase_admin import auth

# Define Blueprints for different functionalities

# Register User Route
register_user = Blueprint('register_user', __name__, url_prefix='/api/register/')
@register_user.route('/', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        company_name = data.get('company_name')
        password = data.get('password')

        # Check if the email is already used or pending OTP validation
        if User.query.filter_by(email=email).first() or TemporaryUser.query.filter_by(email=email).first():
            return jsonify({'message': 'Email is already in use or pending OTP validation'}), 400

        # Generate and send OTP
        otp = generate_otp()
        send_otp_email(email, otp)

        # Save details in TemporaryUser table
        temp_user = TemporaryUser(
            name=name,
            company_name=company_name,
            email=email,
            password=password  # Hash the password before saving in production
        )
        db.session.add(temp_user)

        # Save OTP entry with email reference
        otp_entry = OTP(
            email=email,
            otp=otp,
            valid_duration_minutes=100000
        )
        db.session.add(otp_entry)

        db.session.commit()

        return jsonify({'message': 'OTP sent to email. Please validate to complete registration.'}), 201
    except Exception as e:
        print(f"Error occurred during registration: {e}")
        return jsonify({'message': 'An error occurred while processing the request'}), 500

# Login Route
login = Blueprint('login', __name__, url_prefix='/api/login/')
@login.route('/', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"message": "Missing email or password"}), 400

        user = User.query.filter_by(email=email).first()

        if not user:
            return jsonify({"message": "Invalid credentials"}), 401

        # Check password (you should hash this comparison in production)
        if user.password != password:
            return jsonify({"message": "Invalid credentials"}), 401

        # Pass necessary attributes to token generation function
        token = generate_token(user.id, user.email)

        return jsonify({
            "message": "Login successful",
            "token": token,
            "email": user.email,
            "userId": user.id,
            "name": user.name,
            "companyName": user.company_name,
            "id": user.firebase_uid  # Add Firebase UID to response
        }), 200
    except Exception as e:
        print(f"Error occurred during login: {e}")
        return jsonify({"message": f"An error occurred: {e}"}), 500

# Forgot Password Route
forgot_password = Blueprint('forgot_password', __name__, url_prefix='/api/forgot-password/')

@forgot_password.route('/', methods=['POST'])
def forgot_password_user():
    try:
        data = request.get_json()
        email = data.get('email')

        # Check if the email exists in the User table
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'message': 'Email not found'}), 404

        # Generate and send OTP
        otp = generate_otp()
        send_otp_email(email, otp)

        # Save OTP entry
        otp_entry = OTP(
            email=email,
            otp=otp,
            valid_duration_minutes=100000  # Expiry time for the OTP
        )
        db.session.add(otp_entry)
        db.session.commit()

        return jsonify({'message': 'OTP sent to email for password reset.'}), 200
    except Exception as e:
        print(f"Error occurred during forgot-password process: {e}")
        return jsonify({'message': 'An error occurred while processing the request'}), 500

# Reset Password Route
reset_password = Blueprint('reset_password', __name__, url_prefix='/api/reset-password/')

@reset_password.route('/', methods=['POST'])
def reset_password_user():
    try:
        data = request.get_json()
        email = data.get('email')
        otp = data.get('otp')
        new_password = data.get('newPassword')

        # Find the OTP entry
        otp_entry = OTP.query.filter_by(email=email, otp=otp).first()
        if not otp_entry:
            return jsonify({'message': 'Invalid OTP'}), 400

        # Check if OTP is expired
        if datetime.datetime.now() > otp_entry.valid_till:
            return jsonify({'message': 'OTP has expired'}), 400

        # Update the user's password
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'message': 'User not found'}), 404

        user.password = new_password  # Hash the password before saving in production
        db.session.delete(otp_entry)  # Remove OTP after successful password reset
        db.session.commit()

        return jsonify({'message': 'Password reset successfully'}), 200
    except Exception as e:
        print(f"Error occurred during password reset: {e}")
        return jsonify({'message': 'An error occurred while processing the request'}), 500

validate_forgot_password_otp_bp = Blueprint('validate_forgot_password_otp', __name__, url_prefix='/api/validate-forgot-password-otp/')

@validate_forgot_password_otp_bp.route('/', methods=['POST'])
def validate_forgot_password_otp():
    try:
        data = request.get_json()
        email = data.get('email')
        otp = data.get('otp')

        # Find the OTP entry
        otp_entry = OTP.query.filter_by(email=email, otp=otp).first()
        if not otp_entry:
            return jsonify({'message': 'Invalid OTP'}), 400

        # Check if OTP is expired
        if datetime.datetime.now() > otp_entry.valid_till:
            return jsonify({'message': 'OTP has expired'}), 400

        return jsonify({'message': 'OTP validated successfully. You can now reset your password.'}), 200
    except Exception as e:
        print(f"Error occurred during OTP validation: {e}")
        return jsonify({'message': 'An error occurred while processing the request'}), 500

# Blueprint for OTP validation during registration flow
validate_otp_bp = Blueprint('validate_otp', __name__, url_prefix='/api/validate-otp/')

@validate_otp_bp.route('/', methods=['POST'])
def validate_otp():
    try:
        data = request.get_json()
        email = data.get('email')
        otp = data.get('otp')

        # Find the temporary user
        temp_user = TemporaryUser.query.filter_by(email=email).first()
        if not temp_user:
            return jsonify({'message': 'Temporary user not found'}), 404

        # Find the OTP entry
        otp_entry = OTP.query.filter_by(email=email, otp=otp).first()
        if not otp_entry:
            return jsonify({'message': 'Invalid OTP'}), 400

        # Check if OTP is expired
        if datetime.datetime.now() > otp_entry.valid_till:
            return jsonify({'message': 'OTP has expired'}), 400

        try:
            firebase_user = auth.create_user(
                email=temp_user.email,
                email_verified=True,
                password=temp_user.password,
                display_name=temp_user.name,
                disabled=False
            )
            print('Successfully created new user in Firebase: {0}'.format(firebase_user.uid))
        except Exception as e:
            print(f"Error creating user in Firebase: {e}")
            return jsonify({'message': 'Failed to register user in Firebase.'}), 500

        # Move data from TemporaryUser to User
        new_user = User(
            name=temp_user.name,
            company_name=temp_user.company_name,
            email=temp_user.email,
            password=temp_user.password , # Already hashed
            firebase_uid=firebase_user.uid  # Store Firebase UID for reference
        )
        db.session.add(new_user)

        # Cleanup temporary data and OTP
        db.session.delete(temp_user)
        db.session.delete(otp_entry)

        db.session.commit()

        return jsonify({'message': 'Registration completed successfully.'}), 200
    except Exception as e:
        print(f"Error occurred during OTP validation: {e}")
        return jsonify({'message': 'An error occurred while processing the request'}), 500


@login.route('/logout/', methods=['POST'])
def logout():
    """
    Logs out the user by blacklisting the provided token.
    """
    try:
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        print(auth_header)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Token is missing or invalid'}), 401

        token = auth_header.split(' ')[1]  # Extract the token from the header
        print(f"Extracted Token: {token}, Type: {type(token)}")

        # Validate the token
        decoded_token = decode_token(token)  # Replace with your token validation logic
        if not decoded_token:
            return jsonify({'message': 'Invalid or expired token'}), 401

        # Check if the token is already blacklisted
        existing_blacklisted_token = BlacklistedToken.query.filter_by(token=token).first()
        if existing_blacklisted_token:
            return jsonify({'message': 'Token is already blacklisted'}), 400

        # Blacklist the token
        blacklisted_token = BlacklistedToken(token=token)
        db.session.add(blacklisted_token)
        db.session.commit()

        return jsonify({"message": "Logout successful"}), 200
    except Exception as e:
        print(f"Error during logout: {e}")
        return jsonify({'message': 'An error occurred while processing the request'}), 500