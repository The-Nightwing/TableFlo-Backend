from datetime import datetime, timedelta, timezone
from setup import db  # Import db from the initialized app module
import uuid
from enum import Enum
import json
import pandas as pd

# Helper function to generate UUIDs
def generate_uuid():
    return str(uuid.uuid4())

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)  # UUID as primary key
    name = db.Column(db.String(80), nullable=False)
    company_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(128), nullable=False)
    firebase_uid = db.Column(db.String(128), nullable=False)

    processes = db.relationship('UserProcess', backref='user', lazy=True, cascade="all, delete-orphan")

    def __init__(self, name, company_name, email, password, firebase_uid):
        self.name = name
        self.company_name = company_name
        self.email = email
        self.password = password
        self.firebase_uid = firebase_uid

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'company_name': self.company_name,
            'email': self.email,
            'firebase_uid': self.firebase_uid
        }
        
class OTP(db.Model):
    __tablename__ = 'otps'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    otp = db.Column(db.String(6), nullable=False)
    generated_at = db.Column(db.DateTime, default=datetime.now(), nullable=False)
    valid_till = db.Column(db.DateTime, nullable=False)

    # Store email instead of user ID
    email = db.Column(db.String(120), nullable=False)

    def __init__(self, email, otp, valid_duration_minutes=100000):
        self.email = email
        self.otp = otp
        self.generated_at = datetime.now()
        self.valid_till = self.generated_at + timedelta(minutes=valid_duration_minutes)

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'otp': self.otp,
            'generated_at': self.generated_at.isoformat(),
            'valid_till': self.valid_till.isoformat()
        }
        
        
class TemporaryUser(db.Model):
    __tablename__ = 'temporary_users'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)  # UUID as primary key
    name = db.Column(db.String(80), nullable=False)
    company_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(128), nullable=False)  # Store hashed password

    def __init__(self, name, company_name, email, password):
        self.name = name
        self.company_name = company_name
        self.email = email
        self.password = password

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'company_name': self.company_name,
            'email': self.email,
            'password': self.password
        }
        
class File(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)  # Change to UUID
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)  # Optional link to user
    email = db.Column(db.String(120), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.now())
    file_uuid = db.Column(db.String(255), unique=True, nullable=False)  # Keep for backward compatibility

    # Relationship
    user = db.relationship('User', backref='files', lazy=True)

    def __init__(self, email, file_name, file_size, file_type, file_uuid=None, user_id=None):
        self.id = generate_uuid()
        self.email = email
        self.file_name = file_name
        self.file_size = file_size
        self.file_type = file_type
        self.user_id = user_id
        self.file_uuid = file_uuid or f"{email}_{file_name}"  # Maintain backward compatibility

    def to_dict(self):
        return {
            'id': self.id,  # Add ID to response
            'userId': self.user_id,
            'fileName': self.file_name,
            'fileType': self.file_type,
            'uploadDate': self.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            'fileUuid': self.file_uuid  # Include for backward compatibility
        }

class BlacklistedToken(db.Model):
    __tablename__ = 'blacklisted_tokens'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    token = db.Column(db.String(512), nullable=False, unique=True)  # Reduce length if necessary
    blacklisted_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        db.Index('idx_token_prefix', 'token', mysql_length=255),  # Index first 255 characters
    )

    def __init__(self, token):
        self.token = token

    def to_dict(self):
        return {
            'id': self.id,
            'token': self.token,
            'blacklisted_at': self.blacklisted_at.isoformat()
        }

class ProcessFileKey(db.Model):
    """Table to store file key requirements for a process."""
    __tablename__ = 'process_file_keys'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    key_name = db.Column(db.String(255), nullable=False)  # e.g., "source_data", "reference_file"
    required_structure = db.Column(db.JSON, nullable=False)  # Store required columns and their types
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserProcess(db.Model):
    """
    Table to store user-defined processes.
    """
    __tablename__ = 'user_processes'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    process_name = db.Column(db.String(255), nullable=False)
    file_mappings = db.Column(db.JSON, nullable=True)  # Store file key mappings
    file_metadata = db.Column(db.JSON, nullable=True)  # Store file metadata
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())
    is_active = db.Column(db.Boolean, default=True)
    is_original = db.Column(db.Boolean, nullable=True)  # Track if this is an original process
    original_process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=True)  # ID of the process this was copied from
    is_draft = db.Column(db.Boolean, default=False)  # New field to track draft status

    operations = db.relationship(
        'ProcessOperation',
        backref='process',
        lazy=True,
        cascade="all, delete-orphan",
        order_by="ProcessOperation.sequence"  # Default ordering by sequence
    )

    file_keys = db.relationship(
        'ProcessFileKey',
        backref='process',
        lazy=True,
        cascade="all, delete-orphan"
    )

    # Add self-referential relationship to track original process
    original_process = db.relationship(
        'UserProcess',
        remote_side=[id],
        backref='copied_processes',
        lazy=True
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'process_name': self.process_name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active,
            'is_original': self.is_original,
            'original_process_id': self.original_process_id,
            'is_draft': self.is_draft
        }

    def reorder_operations(self, operation_order):
        """
        Reorder operations based on the provided order
        operation_order: list of tuples [(operation_id, new_sequence), ...]
        """
        for op_id, new_sequence in operation_order:
            operation = ProcessOperation.query.get(op_id)
            if operation and operation.process_id == self.id:
                operation.sequence = new_sequence
        db.session.commit()

class ProcessOperation(db.Model):
    """
    Table to store individual operations within a process.
    """
    __tablename__ = 'process_operations'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)  # UUID primary key
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)  # Link to UserProcess
    operation_name = db.Column(db.String(255), nullable=False)  # Type of operation (e.g., merge, replace)
    title = db.Column(db.String(255), nullable=True)  # Title of the operation
    description = db.Column(db.Text, nullable=True)  # Description of the operation
    parameters = db.Column(db.JSON, nullable=False)  # Store parameters as JSON
    sequence = db.Column(db.Float, nullable=False)  # Changed to Float for easier reordering
    is_active = db.Column(db.Boolean, default=True)  # Add status flag
    created_at = db.Column(db.DateTime, default=datetime.now())  # Add created_at
    updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())  # Track updates
    dataframe_operation_id = db.Column(db.String(36), db.ForeignKey('dataframe_operations.id'), nullable=True)

    # Add relationship
    dataframe_operation = db.relationship('DataFrameOperation', backref='process_operations', lazy=True)

    __table_args__ = (
        db.Index('idx_process_sequence', 'process_id', 'sequence'),  # Add index for faster reordering queries
    )

    def to_dict(self):
        return {
            'id': self.id,
            'process_id': self.process_id,
            'operation_name': self.operation_name,
            'title': self.title,
            'description': self.description,
            'parameters': self.parameters,
            'sequence': self.sequence,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'dataframe_operation_id': self.dataframe_operation_id
        }

class DataFrame(db.Model):
    """
    Table to store DataFrame metadata and tracking information.
    """
    __tablename__ = 'dataframes'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)  # Optional link to user
    name = db.Column(db.String(255), nullable=False)  # Name/identifier of the DataFrame
    email = db.Column(db.String(120), nullable=False)  # Owner's email
    row_count = db.Column(db.Integer, nullable=False)
    column_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_metadata = db.Column(db.JSON, nullable=True)  # Store column types, statistics, etc.
    storage_path = db.Column(db.String(512), nullable=False)  # Path in Firebase Storage
    is_active = db.Column(db.Boolean, default=True)
    is_originally_uploaded = db.Column(db.Boolean, default=False, nullable=True)  # Track if this was an uploaded file

    # Relationships
    process = db.relationship('UserProcess', backref='dataframes', lazy=True)
    user = db.relationship('User', backref='dataframes', lazy=True)

    def __init__(self, process_id, name, email, row_count, column_count, storage_path, user_id=None, data_metadata=None, is_originally_uploaded=False):
        self.id = generate_uuid()
        self.process_id = process_id
        self.user_id = user_id
        self.name = name
        self.email = email
        self.row_count = row_count
        self.column_count = column_count
        self.storage_path = storage_path
        self.data_metadata = data_metadata or {}
        self.is_originally_uploaded = is_originally_uploaded

    def to_dict(self):
        return {
            'id': self.id,
            'processId': self.process_id,
            'userId': self.user_id,
            'name': self.name,
            'email': self.email,
            'rowCount': self.row_count,
            'columnCount': self.column_count,
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.data_metadata,
            'storagePath': self.storage_path,
            'isActive': self.is_active,
            'isOriginallyUploaded': self.is_originally_uploaded
        }

    @staticmethod
    def create_from_pandas(df, process_id, name, email, storage_path, user_id=None, is_originally_uploaded=False, metadata=None):
        """
        Create a DataFrame record from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame
            process_id: ID of the process
            name: Name for the DataFrame
            email: Owner's email
            storage_path: Path where the DataFrame is stored
            user_id: Optional ID of the user
            is_originally_uploaded: Boolean indicating if this was an uploaded file
            metadata: Optional dictionary containing additional metadata
        """
        def get_column_type(series):
            """Determine the type of a pandas Series."""
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if not isinstance(series, pd.Series):
                series = pd.Series(series)

            dtype_str = str(series.dtype)
            if dtype_str.startswith('int'):
                return 'integer'
            elif dtype_str.startswith('float'):
                return 'float'
            elif dtype_str.startswith('datetime'):
                return 'datetime'
            elif dtype_str.startswith('bool'):
                return 'boolean'
            else:
                try:
                    pd.to_datetime(series.dropna().iloc[0])
                    return 'datetime'
                except (ValueError, IndexError):
                    return 'string'

        # Convert pandas Series to basic Python types
        null_counts = {k: int(v) for k, v in df.isnull().sum().items()}
        unique_counts = {k: int(v) for k, v in df.nunique().items()}

        data_metadata = {
            'columns': list(df.columns),
            'columnTypes': {col: get_column_type(df[col]) for col in df.columns},
            'summary': {
                'nullCounts': null_counts,
                'uniqueCounts': unique_counts
            }
        }

        # Merge additional metadata if provided
        if metadata:
            # Convert any pandas objects in metadata to basic Python types
            def convert_to_basic_types(obj):
                if isinstance(obj, (pd.Series, pd.DataFrame)):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_to_basic_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_basic_types(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)

            converted_metadata = convert_to_basic_types(metadata)
            data_metadata.update(converted_metadata)

        return DataFrame(
            process_id=process_id,
            name=name,
            email=email,
            row_count=len(df),
            column_count=len(df.columns),
            storage_path=storage_path,
            user_id=user_id,
            data_metadata=data_metadata,
            is_originally_uploaded=is_originally_uploaded
        )

class OperationType(str, Enum):
    ADD_COLUMN = "add_column"
    MERGE_FILES = "merge_files"
    GROUP_PIVOT = "group_pivot"
    SORT_FILTER = "sort_filter"
    APPLY_FORMATTING = "apply_formatting"
    REPLACE_RENAME_REORDER = "replace_rename_reorder"
    RECONCILE_FILES = "reconcile_files"

class AddColumnSubType(str, Enum):
    APPLY_CALCULATION = "apply_calculation"
    APPLY_CONDITIONAL = "apply_conditional"
    APPLY_PATTERN = "apply_pattern"
    APPLY_CONCAT = "apply_concat"

class MergeSubType(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

class OperationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    IN_PROGRESS = "in_progress"

class DataFrameOperation(db.Model):
    """Model for tracking operations performed on DataFrames."""
    __tablename__ = 'dataframe_operations'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    dataframe_id = db.Column(db.String(36), db.ForeignKey('dataframes.id'), nullable=False)
    
    # Operation type and subtype
    operation_type = db.Column(db.String(50), nullable=False)
    operation_subtype = db.Column(db.String(50), nullable=True)
    
    # Operation details stored as JSON
    payload = db.Column(db.JSON, nullable=False, default=dict)
    
    # Status tracking
    status = db.Column(db.String(20), nullable=False, default=OperationStatus.IN_PROGRESS.value)
    error_message = db.Column(db.Text, nullable=True)
    message = db.Column(db.Text, nullable=True)  # New nullable message field
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())
    
    # Relationships
    process = db.relationship('UserProcess', backref='dataframe_operations', lazy=True)
    dataframe = db.relationship('DataFrame', backref='operations', lazy=True)

    def __init__(self, process_id, dataframe_id, operation_type, operation_subtype=None, payload=None, message=None):
        self.id = generate_uuid()
        self.process_id = process_id
        self.dataframe_id = dataframe_id
        self.operation_type = operation_type
        self.operation_subtype = operation_subtype
        # Ensure payload is a dict
        self.payload = payload if isinstance(payload, dict) else {}
        self.status = OperationStatus.IN_PROGRESS.value
        self.message = message

    def set_success(self):
        """Mark operation as successful"""
        self.status = OperationStatus.SUCCESS.value
        self.error_message = None

    def set_error(self, error_message):
        """Mark operation as failed with error message"""
        self.status = OperationStatus.ERROR.value
        if isinstance(error_message, dict):
            self.error_message = json.dumps(error_message)
        else:
            self.error_message = str(error_message)

    def validate_operation(self):
        """Validate operation type and subtype"""
        if self.operation_type not in [e.value for e in OperationType]:
            raise ValueError(f"Invalid operation type: {self.operation_type}")
            
        if self.operation_type == OperationType.ADD_COLUMN.value:
            if self.operation_subtype not in [e.value for e in AddColumnSubType]:
                raise ValueError(f"Invalid subtype for ADD_COLUMN: {self.operation_subtype}")
        elif self.operation_type == OperationType.MERGE_FILES.value:
            if self.operation_subtype not in [e.value for e in MergeSubType]:
                raise ValueError(f"Invalid subtype for MERGE_FILES: {self.operation_subtype}")

    def to_dict(self):
        result = {
            'id': self.id,
            'processId': self.process_id,
            'dataframeId': self.dataframe_id,
            'operationType': self.operation_type,
            'operationSubtype': self.operation_subtype,
            'status': self.status,
            'message': self.message,
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None
        }

        # Add payload directly
        result['payload'] = self.payload

        try:
            if self.error_message:
                result['errorMessage'] = json.loads(self.error_message) if isinstance(self.error_message, str) else self.error_message
        except:
            result['errorMessage'] = self.error_message

        return result

class FormattingStep(db.Model):
    """Model for tracking formatting configurations applied to DataFrames within a process."""
    __tablename__ = 'formatting_steps'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    source_dataframe_id = db.Column(db.String(36), db.ForeignKey('dataframes.id'), nullable=False)    
    configuration = db.Column(db.JSON, nullable=False)
    storage_location = db.Column(db.String(255), nullable=True)  # Added storage_location field
    message = db.Column(db.Text, nullable=True)  # Changed to Text type
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    include_in_process = db.Column(db.Boolean, nullable=True, default=False)  # Updated with default=False
    
    # Relationships
    process = db.relationship('UserProcess', backref=db.backref('formatting_steps', lazy=True))
    source_dataframe = db.relationship('DataFrame', foreign_keys=[source_dataframe_id])

    def to_dict(self):
        """Convert the formatting step to a dictionary."""
        return {
            'id': self.id,
            'processId': self.process_id,
            'sourceDataframeId': self.source_dataframe_id,
            'configuration': self.configuration,
            'storageLocation': self.storage_location,  # Added to dictionary output
            'message': self.message,  # Added message to dictionary output
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None,
            'includeInProcess': self.include_in_process
        }

    @classmethod
    def create_from_request(cls, process_id, source_df_id, request_data):
        """Create a formatting step from the request data."""
        return cls(
            process_id=process_id,
            source_dataframe_id=source_df_id,
            configuration=request_data,
            storage_location=request_data.get('storageLocation'),  # Added storage_location handling
            message=request_data.get('message')  # Added message handling
        )

class DataFrameBatchOperation(db.Model):
    """Model for storing batch operations on multiple DataFrames"""
    __tablename__ = 'dataframe_batch_operations'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='in_progress')  # in_progress, success, error, partial_success
    error_message = db.Column(db.Text)
    message = db.Column(db.Text, nullable=True)  # New nullable message field
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Store the complete payload of the request
    payload = db.Column(db.JSON, nullable=False)
    
    # Store the list of DataFrame IDs involved
    dataframe_ids = db.Column(db.JSON, nullable=False)  # List of DataFrame IDs
    
    # Store the list of individual operation IDs
    operation_ids = db.Column(db.JSON, nullable=False)  # List of DataFrameOperation IDs
    
    # Statistics
    total_dataframes = db.Column(db.Integer, nullable=False)
    successful_dataframes = db.Column(db.Integer, default=0)
    
    # Relationships
    process = db.relationship('UserProcess', backref='batch_operations')

    def set_success(self):
        """Mark the operation as successful"""
        self.status = 'success'
        self.updated_at = datetime.now(timezone.utc)

    def set_error(self, error_message):
        """Mark the operation as failed"""
        self.status = 'error'
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)

    def set_partial_success(self, error_message):
        """Mark the operation as partially successful"""
        self.status = 'partial_success'
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)

    def increment_success_count(self):
        """Increment the successful dataframes count"""
        self.successful_dataframes += 1
        if self.successful_dataframes == self.total_dataframes:
            self.set_success()
        elif self.successful_dataframes > 0:
            self.set_partial_success("Some dataframes failed to process")

class ProcessRun(db.Model):
    """
    Table to store Process Run information and track dataframe mappings.
    """
    __tablename__ = 'process_runs'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    original_process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)  # Added field
    run_number = db.Column(db.Integer, nullable=False)
    dataframe_mappings = db.Column(db.JSON, nullable=True)  # Store mappings between original and run dataframes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    process = db.relationship('UserProcess', backref='runs', lazy=True, foreign_keys=[process_id])
    original_process = db.relationship('UserProcess', lazy=True, foreign_keys=[original_process_id])  # Added relationship

    def __init__(self, process_id, original_process_id, run_number, dataframe_mappings=None):  # Updated constructor
        self.id = generate_uuid()
        self.process_id = process_id
        self.original_process_id = original_process_id  # Added field initialization
        self.run_number = run_number
        self.dataframe_mappings = dataframe_mappings or {
            "original_dataframes": [],  # List of original dataframe names
            "run_dataframes": {}        # Mapping of original names to new dataframe details
        }

    def to_dict(self):
        return {
            'id': self.id,
            'processId': self.process_id,
            'originalProcessId': self.original_process_id,  # Added field to dict
            'runNumber': self.run_number,
            'dataframeMappings': self.dataframe_mappings,
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None
        }

    def update_dataframe_mapping(self, original_name, run_dataframe):
        """
        Update the mapping for a specific dataframe.
        
        Args:
            original_name: Name of the original dataframe
            run_dataframe: DataFrame instance for the run
        """
        if original_name not in self.dataframe_mappings['original_dataframes']:
            self.dataframe_mappings['original_dataframes'].append(original_name)
        
        self.dataframe_mappings['run_dataframes'][original_name] = {
            'id': run_dataframe.id,
            'name': run_dataframe.name
        }

class Visualization(db.Model):
    """Model for storing visualization configurations within a process."""
    __tablename__ = 'visualizations'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), db.ForeignKey('user_processes.id'), nullable=False)
    configuration = db.Column(db.JSON, nullable=False)
    message = db.Column(db.Text, nullable=True)  # Added nullable message field
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    def __init__(self, process_id, configuration, message=None):
        self.process_id = process_id
        self.configuration = configuration
        self.message = message

    def to_dict(self):
        """Convert the visualization to a dictionary."""
        return {
            'id': self.id,
            'processId': self.process_id,
            'configuration': self.configuration,
            'message': self.message,  # Added message to dictionary output
            'createdAt': self.created_at.isoformat() if self.created_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None
        }

class AIRequest(db.Model):
    """Model for tracking AI requests and responses."""
    __tablename__ = 'ai_requests'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    process_id = db.Column(db.String(36), nullable=True)
    user_id = db.Column(db.String(36), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    
    # Request details
    operation_type = db.Column(db.String(50), nullable=True)
    query = db.Column(db.Text, nullable=True)
    table_name = db.Column(db.String(255), nullable=True)
    second_table_name = db.Column(db.String(255), nullable=True)
    
    # AI Response details - store everything in a single JSON field
    response = db.Column(db.JSON, nullable=True)
    
    # Status tracking
    status = db.Column(db.String(20), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # Timing information
    request_time = db.Column(db.DateTime(timezone=True), nullable=True)
    response_time = db.Column(db.DateTime(timezone=True), nullable=True)
    processing_duration = db.Column(db.Float, nullable=True)

    # Retry tracking
    original_request_id = db.Column(db.String(36), nullable=True)
    retry_count = db.Column(db.Integer, default=0)
    max_retries = db.Column(db.Integer, default=5)
    last_retry_time = db.Column(db.DateTime(timezone=True), nullable=True)

    def __init__(self, process_id=None, user_id=None, email=None, operation_type=None, query=None, table_name=None, second_table_name=None, original_request_id=None, custom_id=None):
        # Set custom ID if provided, otherwise generate UUID
        self.id = custom_id if custom_id else generate_uuid()
        self.process_id = process_id
        self.user_id = user_id
        self.email = email
        self.operation_type = operation_type
        self.query = query
        self.table_name = table_name
        self.second_table_name = second_table_name
        self.request_time = datetime.now(timezone.utc)
        self.status = 'in_progress'
        self.original_request_id = original_request_id
        self.retry_count = 0
        self.max_retries = 5

    def to_dict(self):
        """Convert the AI request to a dictionary."""
        return {
            'id': self.id,
            'processId': self.process_id,
            'userId': self.user_id,
            'email': self.email,
            'operationType': self.operation_type,
            'query': self.query,
            'tableName': self.table_name,
            'secondTableName': self.second_table_name,
            'status': self.status,
            'requestTime': self.request_time.isoformat() if self.request_time else None,
            'responseTime': self.response_time.isoformat() if self.response_time else None,
            'processingDuration': self.processing_duration,
            'originalRequestId': self.original_request_id,
            'retryCount': self.retry_count,
            'maxRetries': self.max_retries,
            'lastRetryTime': self.last_retry_time.isoformat() if self.last_retry_time else None,
            'response': self.response,
            'errorMessage': self.error_message
        }

