import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pymysql

# Initialize the app and db
db = SQLAlchemy()

def create_app():
    application = Flask(__name__)

    # Enable CORS
    CORS(application, resources={r'/*': {'origins': '*'}}, supports_credentials=True)

    # Setup AWS RDS connection
    #rds_host = "fileapp.cb04g40cq6ae.ap-south-1.rds.amazonaws.com"  # RDS endpoint
    rds_host = "tableflowdb.c142wimke6yn.ap-south-1.rds.amazonaws.com"
    #rds_host = "localhost"
    rds_port = 3306  # RDS MySQL port
    db_user = "admin"  # Database username
    #db_pass = "PratikLonari"  # Database password
    db_pass = "tableflowDB"
    #db_pass = "Jepru7astur47"
    #db_pass = "ptk#"
    #db_name = "filesystem"  # Database name
    db_name = "tableflowdb1"
    # Configure SQLAlchemy with an explicit connection string
    application.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{db_user}:{db_pass}@{rds_host}:{rds_port}/{db_name}"
    application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking

    # Initialize the db with the app
    db.init_app(application)

    # Test database connection and create tables within the app context
    with application.app_context():
        try:
            # Test database connection
            with db.engine.connect() as connection:
                print("Database connection to AWS RDS successful!")

            # Create tables if they don't exist
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            raise

    return application
