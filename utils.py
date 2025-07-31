from backend.application import db
from models import User

def add_user(username, email, password):
    """Add a new user to the database."""
    user = User(username=username, email=email, password=password)
    db.session.add(user)
    db.session.commit()
    return user.to_dict()

def get_all_users():
    """Retrieve all users from the database."""
    users = User.query.all()
    return [user.to_dict() for user in users]

def get_user_by_id(user_id):
    """Retrieve a user by their ID."""
    user = User.query.get(user_id)
    return user.to_dict() if user else None
