import jwt
import datetime
import jwt
from datetime import datetime, timedelta
from config import Config
from flask import current_app

def generate_token(user_id, user_email):
    try:
        payload = {
            'sub': user_id,  # Subject (user's ID)
            'email': user_email,
            'exp': datetime.now() + timedelta(days=1),  # Expiry time
        }
        # Generate JWT token (you might use a different method depending on your needs)
        token = jwt.encode(payload, Config.SECRET_KEY, algorithm='HS256')
        return token
    except Exception as e:
        print(f"Error generating token: {e}")
        return None

def decode_token(token):
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
