import os
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict, ValidationInfo
from langchain_community.chat_models import ChatOpenAI
from typing import Optional, List, Dict, Any, Union
import datetime
from typing_extensions import Annotated

# Get OpenAI API key from environment variable or config
def get_openai_api_key():
    """Get OpenAI API key from environment or app config."""
    return "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"
    #return os.getenv('OPENAI_API_KEY') or current_app.config.get('OPENAI_API_KEY')

# Argument models for each operation type
class PatternOperation(BaseModel):
    tableName: str
    newColumnName: str
    operationType: str = "pattern"
    sourceColumn: str
    pattern: str

class CalculationStep(BaseModel):
    column1: str
    operator: str
    column2: Optional[str] = None
    fixed_value: Optional[float] = None

class CalculationOperation(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "column1": "Total",
                "operator": "*",
                "column2": None,
                "fixed_value": 10.0
            }
        }
    )
    
    column1: Optional[str] = None
    operator: str
    column2: Optional[str] = None  # Will always be included in output
    fixed_value: Optional[float] = None

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v: str) -> str:
        if v not in ['+', '-', '*', '/']:
            raise ValueError('Operator must be one of: +, -, *, /')
        return v

    @model_validator(mode='after')
    def validate_value_sources(self) -> 'CalculationOperation':
        if self.fixed_value is not None:
            self.column2 = None  # Explicitly set column2 to None when using fixed_value
        elif self.column2 is not None:
            self.fixed_value = None  # Explicitly set fixed_value to None when using column2
        else:
            raise ValueError('Must specify either column2 or fixed_value')
        return self

    def model_dump(self, **kwargs):
        # Ensure column2 is always included in the output
        data = super().model_dump(**kwargs)
        data['column2'] = None if self.fixed_value is not None else self.column2
        return data

class ConcatStep(BaseModel):
    column: str
    type: str  # "Left", "Right", or "Full text"
    chars: Optional[int] = None

class ConcatOperation(BaseModel):
    tableName: str
    newColumnName: str
    operationType: str = "concatenate"
    operations: List[ConcatStep]

class Condition(BaseModel):
    operator: str
    reference_value: str
    conditional_value: str

class ConditionalOperation(BaseModel):
    tableName: str
    newColumnName: str
    operationType: str = "conditional"
    sourceColumn: str
    conditions: List[Condition]
    residualValue: str

class AddColumnArgs(BaseModel):
    tableName: str = Field(description="Name of the source table")
    newColumnName: str = Field(description="Name of the new column to be created")
    operationType: str = Field(description="Type of operation: pattern, calculate, concatenate, conditional")
    sourceColumn: Optional[str] = Field(None, description="Source column for pattern or conditional operations")
    pattern: Optional[str] = Field(None, description="Regex pattern for pattern operation")
    operations: Optional[List[CalculationOperation]] = Field(None, description="Steps for calculation or concatenation")
    conditions: Optional[List[Condition]] = Field(None, description="Conditions for conditional operation")
    residualValue: Optional[str] = Field(None, description="Default value for conditional operation")

    @model_validator(mode='after')
    def validate_operations(self) -> 'AddColumnArgs':
        if self.operationType == 'calculate':
            if not self.operations or len(self.operations) < 1:
                raise ValueError('Must provide at least one operation for calculation type')
            if not self.operations[0].column1:
                raise ValueError('First operation must have a column1')
            for op in self.operations[1:]:
                if op.column1 is not None:
                    raise ValueError('Subsequent operations must have column1 as null')
                
            # Ensure all operations with fixed_value have column2 as null
            for op in self.operations:
                if op.fixed_value is not None:
                    op.column2 = None
                    
        return self

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if self.operationType == 'calculate' and self.operations:
            # Ensure column2 is explicitly included in each operation
            for op in data['operations']:
                if op.get('fixed_value') is not None:
                    op['column2'] = None
        return data

class KeyPair(BaseModel):
    left: str = Field(description="Column name from first table")
    right: str = Field(description="Column name from second table")

class MergeFilesArgs(BaseModel):
    table1Name: str = Field(description="Name of first table")
    table2Name: str = Field(description="Name of second table")
    mergeType: str = Field(description="Type of merge: horizontal or vertical")
    mergeMethod: Optional[str] = Field(None, description="Method for horizontal merge: inner, outer, left, right")
    keyPairs: Optional[List[KeyPair]] = Field(None, description="Key pairs for joining tables")
    showCountSummary: Optional[bool] = Field(False, description="Show count summary of merge")
    outputTableName: str = Field(description="Name of the output table")

    @model_validator(mode='after')
    def validate_merge_config(self) -> 'MergeFilesArgs':
        if self.mergeType == "horizontal":
            if not self.mergeMethod:
                raise ValueError("mergeMethod is required for horizontal merge")
            if self.mergeMethod not in ["inner", "outer", "left", "right"]:
                raise ValueError("mergeMethod must be one of: inner, outer, left, right")
            if not self.keyPairs:
                raise ValueError("keyPairs are required for horizontal merge")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "table1Name": "2007-2",
                    "table2Name": "2007-1",
                    "mergeType": "vertical",
                    "showCountSummary": True,
                    "outputTableName": "vertical"
                },
                {
                    "table1Name": "table1",
                    "table2Name": "table2",
                    "mergeType": "horizontal",
                    "mergeMethod": "inner",
                    "keyPairs": [{"left": "id", "right": "id"}],
                    "showCountSummary": True,
                    "outputTableName": "merged_table"
                }
            ]
        }
    )

class PivotValue(BaseModel):
    column: str = Field(description="Column name to aggregate")
    aggregation: str = Field(description="Aggregation function to use")

    @field_validator('aggregation')
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        valid_aggs = {'mean', 'sum', 'count', 'min', 'max'}
        if v not in valid_aggs:
            raise ValueError(f'Aggregation must be one of: {", ".join(valid_aggs)}')
        return v

class GroupPivotArgs(BaseModel):
    tableName: str = Field(description="Name of the source table")
    rowIndex: List[str] = Field(description="Array of column names to use as row indices")
    columnIndex: Optional[str] = Field(None, description="Column to use as column index (can be null)")
    pivotValues: List[PivotValue] = Field(description="Array of values to aggregate")
    outputTableName: str = Field(description="Name for the output pivot table")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tableName": "DF",
                "rowIndex": ["Country"],
                "columnIndex": None,
                "pivotValues": [
                    {
                        "column": "S1: Demographic Pressures",
                        "aggregation": "mean"
                    },
                    {
                        "column": "S2: Refugees and IDPs",
                        "aggregation": "mean"
                    }
                ],
                "outputTableName": "social_indicators"
            }
        }
    )

class FilterConfig(BaseModel):
    column: str
    operator: str
    value: Union[str, int, float]

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v: str) -> str:
        valid_operators = {
            "equals", "not_equals", "contains", "not_contains",
            "greater_than", "less_than", "greater_equals", "less_equals",
            "starts_with", "ends_with"
        }
        if v not in valid_operators:
            raise ValueError(f'Operator must be one of: {", ".join(valid_operators)}')
        return v

class SortConfig(BaseModel):
    column: str
    order: str

    @field_validator('order')
    @classmethod
    def validate_order(cls, v: str) -> str:
        if v not in ['asc', 'desc']:
            raise ValueError('Order must be either "asc" or "desc"')
        return v

class SortFilterArgs(BaseModel):
    table_name: str = Field(description="Name of the source table")
    sort_config: List[SortConfig] = Field(description="Sorting configuration")
    filter_config: List[FilterConfig] = Field(description="Filtering configuration")
    output_table_name: Optional[str] = Field(None, description="Name of the output table")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "table_name": "example_table",
                "sort_config": [{"column": "date", "order": "desc"}],
                "filter_config": [{"column": "value", "operator": "greater_than", "value": 100}],
                "output_table_name": "filtered_table"
            }
        }
    )

class FunctionCall(BaseModel):
    function: str = Field(description="Operation type: add_column, merge_files, group_pivot, sort_filter, replace_rename_reorder, reconcile, format")
    args: dict

    @classmethod
    def validate_args(cls, values):
        function = values.get("function")
        args = values.get("args")

        if function == "reconcile":
            parsed_args = ReconcileArgs(**args).model_dump()
        elif function == "add_column":
            parsed_args = AddColumnArgs(**args).model_dump()
        elif function == "merge_files":
            parsed_args = MergeFilesArgs(**args).model_dump()
        elif function == "group_pivot":
            parsed_args = GroupPivotArgs(**args).model_dump()
        elif function == "sort_filter":
            parsed_args = SortFilterArgs(**args).model_dump()
        elif function == "replace_rename_reorder":
            parsed_args = ReplaceRenameReorderArgs(**args).model_dump()
        elif function == "format":
            parsed_args = FormattingArgs(**args).model_dump()
        elif function == "regex":
            parsed_args = RegexArgs(**args).model_dump()
        else:
            raise ValueError(f"Unsupported function type: {function}")

        return {"function": function, "args": parsed_args}

class DataFrameSummary(BaseModel):
    nullCounts: Dict[str, int] = Field(description="Count of null values in each column")
    uniqueCounts: Dict[str, int] = Field(description="Count of unique values in each column")

class DataFrameMetadata(BaseModel):
    """Model for DataFrame metadata"""
    columns: List[str] = Field(description="List of column names in the DataFrame")
    summary: DataFrameSummary = Field(description="Summary statistics of the DataFrame")
    columnTypes: Dict[str, str] = Field(description="Data types of each column")

# Add new domain model
class AIDomain(BaseModel):
    """Model for AI response domain information"""
    model: str = Field(description="AI model used for processing")
    version: str = Field(description="Model version")
    timestamp: str = Field(description="Processing timestamp")
    confidence: float = Field(description="Confidence score of the response", ge=0, le=1)
    processing_time: float = Field(description="Time taken to process the request in seconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used in the request")

# Add new models for replace/rename/reorder operations
class ReplaceValueConfig(BaseModel):
    column: str = Field(description="Column name where values will be replaced")
    oldValue: Any = Field(description="Value to be replaced")
    newValue: Any = Field(description="New value")
    matchCase: bool = Field(default=False, description="Whether to match case for string replacements")

class Operation(BaseModel):
    type: str = Field(description="Operation type: rename_columns, reorder_columns, or replace_values")
    mapping: Optional[Dict[str, str]] = Field(None, description="Mapping of old column names to new names")
    order: Optional[List[str]] = Field(None, description="New order of columns")
    replacements: Optional[List[ReplaceValueConfig]] = Field(None, description="Value replacement configurations")

    @model_validator(mode='after')
    def validate_operation_config(self) -> 'Operation':
        if self.type == "rename_columns":
            if not self.mapping:
                raise ValueError("mapping is required for rename_columns operation")
        elif self.type == "reorder_columns":
            if not self.order:
                raise ValueError("order is required for reorder_columns operation")
        elif self.type == "replace_values":
            if not self.replacements:
                raise ValueError("replacements are required for replace_values operation")
        else:
            raise ValueError("type must be one of: rename_columns, reorder_columns, replace_values")
        return self

class ReplaceRenameReorderArgs(BaseModel):
    tableName: str = Field(description="Source table name")
    outputTableName: str = Field(description="Output table name")
    operations: List[Operation] = Field(description="List of operations to perform")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tableName": "DF",
                "outputTableName": "fsi_2007_final",
                "operations": [
                    {
                        "type": "rename_columns",
                        "mapping": {
                            "C1: Security Apparatus": "Security",
                            "C2: Factionalized Elites": "Elites"
                        }
                    },
                    {
                        "type": "reorder_columns",
                        "order": ["Country", "Year", "Rank", "Total"]
                    },
                    {
                        "type": "replace_values",
                        "replacements": [
                            {
                                "column": "Country",
                                "oldValue": "USA",
                                "newValue": "United States",
                                "matchCase": True
                            }
                        ]
                    }
                ]
            }
        }
    )

# Add these new models after the existing models

class ReconcileKey(BaseModel):
    left: str = Field(description="Column name from first table")
    right: str = Field(description="Column name from second table")
    criteria: str = Field(default="exact", description="Matching criteria: 'exact' or 'fuzzy'")
    case_sensitive: str = Field(default="yes", description="Case sensitivity: 'yes' or 'no'")
    ignore_special: str = Field(default="no", description="Special characters handling: 'yes' or 'no'")
    fuzzy_ranking_basis: Optional[bool] = Field(None, description="Required when criteria is fuzzy and multiple fuzzy matches exist")

    @field_validator('criteria')
    @classmethod
    def validate_criteria(cls, v: str) -> str:
        if v not in ['exact', 'fuzzy']:
            raise ValueError("criteria must be either 'exact' or 'fuzzy'")
        return v

    @field_validator('case_sensitive', 'ignore_special')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        if v not in ['yes', 'no']:
            raise ValueError("must be either 'yes' or 'no'")
        return v

class ReconcileValue(BaseModel):
    left: str = Field(description="Column name from first table")
    right: str = Field(description="Column name from second table")
    threshold_type: Optional[str] = Field(default="amount", description="Type of threshold: 'percent', 'amount', or null")
    threshold_value: float = Field(default=0.0, description="Numeric threshold value")

    @field_validator('threshold_type')
    @classmethod
    def validate_threshold_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ['percent', 'amount']:
            raise ValueError("threshold_type must be 'percent', 'amount', or null")
        return v

    @field_validator('threshold_value')
    @classmethod
    def validate_threshold_value(cls, v: float, info: ValidationInfo) -> float:
        if v <= 0:
            raise ValueError("threshold_value must be positive")
        threshold_type = info.data.get('threshold_type')
        if threshold_type == 'percent' and v > 100:
            raise ValueError("percent threshold must be between 0 and 100")
        return v

class CrossReferenceSettings(BaseModel):
    method: str = Field(default="one-to-many", description="Matching method: 'one-to-one' or 'one-to-many'")
    duplicate: str = Field(default="first_occurrence", description="Duplicate handling: 'first_occurrence', 'last_occurrence', or 'all'")
    basis_column: Dict[str, str] = Field(description="Basis columns for matching")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ['one-to-one', 'one-to-many']:
            raise ValueError("method must be either 'one-to-one' or 'one-to-many'")
        return v

    @field_validator('duplicate')
    @classmethod
    def validate_duplicate(cls, v: str) -> str:
        if v not in ['first_occurrence', 'last_occurrence', 'all']:
            raise ValueError("duplicate must be 'first_occurrence', 'last_occurrence', or 'all'")
        return v

class CrossReference(BaseModel):
    settings: CrossReferenceSettings
    left: List[str] = Field(description="Columns to include from first table")
    right: List[str] = Field(description="Columns to include from second table")

class ReconcileArgs(BaseModel):
    sourceTableNames: List[str] = Field(description="Array of exactly 2 table names")
    outputTableName: str = Field(description="Output table name")
    keys: List[ReconcileKey] = Field(description="Array of matching key configurations")
    values: List[ReconcileValue] = Field(description="Array of value comparison configurations")
    crossReference: CrossReference = Field(description="Cross-reference configuration")

    @model_validator(mode='after')
    def validate_tables_and_keys(self) -> 'ReconcileArgs':
        if len(self.sourceTableNames) != 2:
            raise ValueError("sourceTableNames must contain exactly 2 table names")
        
        # Ensure at least one exact key match
        if not any(k.criteria == 'exact' for k in self.keys):
            raise ValueError("Must have at least one key with criteria: 'exact'")
        
        # Check fuzzy ranking basis
        fuzzy_keys = [k for k in self.keys if k.criteria == 'fuzzy']
        if fuzzy_keys:
            # Set fuzzy_ranking_basis=True for the first fuzzy key if not specified
            first_fuzzy_key_found = False
            for k in self.keys:
                if k.criteria == 'fuzzy':
                    if not first_fuzzy_key_found:
                        k.fuzzy_ranking_basis = True
                        first_fuzzy_key_found = True
                    else:
                        # Ensure others don't have ranking basis
                        k.fuzzy_ranking_basis = False
        
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sourceTableNames": ["DF", "high_risk_countries_2007"],
                "outputTableName": "reconciled_risk_analysis_2007",
                "keys": [
                    {
                        "left": "Country",
                        "right": "Country",
                        "criteria": "exact",
                        "case_sensitive": "no",
                        "ignore_special": "yes"
                    }
                ],
                "values": [
                    {
                        "left": "Total",
                        "right": "Total",
                        "threshold_type": "percent",
                        "threshold_value": 1
                    }
                ],
                "crossReference": {
                    "settings": {
                        "method": "one-to-one",
                        "duplicate": "first_occurrence",
                        "basis_column": {
                            "left": "Country",
                            "right": "Country"
                        }
                    },
                    "left": ["Country", "Total"],
                    "right": ["Country", "Total"]
                }
            }
        }
    )

# Add new models for formatting operations

class FormattingLocation(BaseModel):
    range: str = Field(description="Column letters (e.g., 'A,B' or 'A:D')")

class ConditionalFormatConfig(BaseModel):
    operator: str = Field(description="Comparison operator")
    value: Any = Field(description="Value to compare against")
    trueFormat: str = Field(description="Format to apply when condition is true (hex color code)")
    falseFormat: str = Field(description="Format to apply when condition is false (hex color code)")

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v: str) -> str:
        valid_operators = [
            "equals", "not equals", "greater than", "less than", 
            "greater than or equal to", "less than or equal to"
        ]
        if v not in valid_operators:
            raise ValueError(f"operator must be one of: {', '.join(valid_operators)}")
        return v

class FormatConfig(BaseModel):
    # Different format configurations based on type
    numberFormat: Optional[str] = Field(None, description="Number format type")
    fontColor: Optional[str] = Field(None, description="Font color hex code")
    fillColor: Optional[str] = Field(None, description="Fill color hex code")
    bold: Optional[bool] = Field(None, description="Whether to apply bold formatting")
    columnWidth: Optional[int] = Field(None, description="Width of the column")
    conditional: Optional[ConditionalFormatConfig] = Field(None, description="Conditional formatting configuration")

class FormattingOperation(BaseModel):
    type: str = Field(description="Type of formatting to apply")
    location: FormattingLocation = Field(description="Location to apply formatting")
    format: FormatConfig = Field(description="Formatting configuration")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = [
            "Cell Number Format", "Font Colour", "Fill Colour", 
            "Bold", "Column Width", "Conditional Formatting"
        ]
        if v not in valid_types:
            raise ValueError(f"type must be one of: {', '.join(valid_types)}")
        return v

class FormattingArgs(BaseModel):
    processId: str = Field(description="ID of the process")
    tableName: str = Field(description="Name of the source table to format")
    outputTableName: str = Field(description="Name for the formatted output table")
    formattingConfigs: List[FormattingOperation] = Field(description="Array of formatting configurations")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "processId": "4eceb325-1c43-4a6a-a1c9-6e688a11ca8b",
                "tableName": "T12",
                "outputTableName": "formatted",
                "formattingConfigs": [
                    {
                        "type": "Column Width",
                        "location": {
                            "range": "A,B,C,D"
                        },
                        "format": {
                            "columnWidth": 15
                        }
                    },
                    {
                        "type": "Conditional Formatting",
                        "location": {
                            "range": "B"
                        },
                        "format": {
                            "conditional": {
                                "operator": "greater than",
                                "value": 5,
                                "trueFormat": "#FF0000",
                                "falseFormat": "#00FF00"
                            }
                        }
                    }
                ]
            }
        }
    )

class RegexArgs(BaseModel):
    pattern: str = Field(description="The actual regex pattern")
    explanation: str = Field(description="A brief explanation of how the pattern works")

def run_chain(user_input: str, operation_type: str, table_name: str, process_id: str, dataframe_metadata: Dict[str, Any], table2_metadata: Optional[Dict[str, Any]] = None):
    """
    Process natural language input with context about the DataFrame(s)
    
    Args:
        user_input: Natural language query from user
        operation_type: Type of operation (add_column, merge_files, group_pivot, sort_filter)
        table_name: Name of the DataFrame/table
        process_id: ID of the process
        dataframe_metadata: Metadata about the first DataFrame
        table2_metadata: Metadata about the second DataFrame (for merge operations)
    """
    try:
        import time
        start_time = time.time()

        # Convert metadata to structured format
        df_meta = DataFrameMetadata(**dataframe_metadata)
        df2_meta = DataFrameMetadata(**table2_metadata) if table2_metadata else None
        
        # Prepare column information for first table
        numeric_columns = [
            col for col, dtype in df_meta.columnTypes.items() 
            if dtype in ['float64', 'int64']
        ]
        categorical_columns = [
            col for col, dtype in df_meta.columnTypes.items() 
            if dtype in ['object', 'datetime64[ns]']
        ]
        
        # Format column info with types for first table
        column_info = "\n".join([
            f"- {col} ({df_meta.columnTypes[col]})"
            for col in df_meta.columns
        ])

        # Prepare column information for second table if it exists
        if df2_meta:
            numeric_columns2 = [
                col for col, dtype in df2_meta.columnTypes.items() 
                if dtype in ['float64', 'int64']
            ]
            categorical_columns2 = [
                col for col, dtype in df2_meta.columnTypes.items() 
                if dtype in ['object', 'datetime64[ns]']
            ]
            column_info2 = "\n".join([
                f"- {col} ({df2_meta.columnTypes[col]})"
                for col in df2_meta.columns
            ])
        
        # Create operation-specific prompt template
        prompt_template = create_operation_prompt(operation_type, df_meta, df2_meta)
        
        # Create chain with appropriate parser
        chain = create_chain(operation_type, prompt_template)
        
        # Add context to user input
        context_input = {
            "user_input": user_input,
            "tableName": table_name,
            "column_info": column_info,
            "column_count": len(df_meta.columns),
            "numeric_columns": ", ".join(numeric_columns),
            "categorical_columns": ", ".join(categorical_columns),
            "process_id": process_id
        }

        # Add second table information for merge operations or reconcile operations
        if (operation_type == "merge-files" or operation_type == "reconcile") and df2_meta:
            context_input.update({
                "table2_column_info": column_info2,
                "table2_column_count": len(df2_meta.columns),
                "table2_numeric_columns": ", ".join(numeric_columns2),
                "table2_categorical_columns": ", ".join(categorical_columns2)
            })
        elif (operation_type == "merge-files" or operation_type == "reconcile") and not df2_meta:
            raise ValueError("Second table metadata is required for merge and reconcile operations")
        
        structured_output = chain.invoke(context_input)

        # Normalize structured output from chain: it may be a Pydantic model, have an `args` attribute,
        # or be a dict depending on LangChain version. Extract parameters accordingly.
        def _extract_parameters(obj):
            if obj is None:
                return {}
            if hasattr(obj, 'args'):
                return obj.args
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, 'model_dump'):
                try:
                    return obj.model_dump()
                except Exception:
                    return {}
            # Fallback: try to convert to dict if possible
            try:
                return dict(obj)
            except Exception:
                return {}

        parsed_params = _extract_parameters(structured_output)
        
        # Calculate processing time
        processing_time = time.time() - start_time

        # Create domain information
        domain_info = AIDomain(
            model="gpt-4.1-mini",
            version="1.0",
            timestamp=datetime.datetime.now().isoformat(),
            confidence=0.95,
            processing_time=processing_time,
            tokens_used=None
        )

        return {
            "success": True,
            "operation": operation_type,
            "tableName": table_name,
            "process_id": process_id,
            "table_name": table_name,
            "parameters": parsed_params,
            "metadata_used": {
                "table1": {
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns
                },
                "table2": {
                    "numeric_columns": numeric_columns2,
                    "categorical_columns": categorical_columns2
                } if df2_meta else None
            },
            "domain": domain_info.model_dump()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing request: {str(e)}"
        }

def create_operation_prompt(operation_type: str, df_meta: DataFrameMetadata, df2_meta: Optional[DataFrameMetadata] = None) -> PromptTemplate:
    """Create operation-specific prompt template"""
    
    parser = PydanticOutputParser(pydantic_object=FunctionCall)
    format_instructions = parser.get_format_instructions()
    
    # Define base variables that are common to all operations
    all_variables = [
        "user_input", 
        "tableName", 
        "column_info",
        "column_count",
        "numeric_columns",
        "categorical_columns",
        "format_instructions"
    ]

    if operation_type == "merge-files":
        # Add merge-specific variables
        all_variables.extend([
            "table2_column_info",
            "table2_column_count",
            "table2_numeric_columns",
            "table2_categorical_columns"
        ])
        template = """You are a data processing assistant. Create merge operations based on the available tables.

First Table columns and their types:
{column_info}

First Table summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

Second Table columns and their types:
{table2_column_info}

Second Table summary:
- Total columns: {table2_column_count}
- Numeric columns: {table2_numeric_columns}
- Categorical columns: {table2_categorical_columns}

User request: {user_input}

{format_instructions}

Generate a structured merge-files operation that specifies:
1. mergeType: Must be one of:
   - "horizontal" (join tables side by side using key columns)
   - "vertical" (stack tables on top of each other)

2. For horizontal merges (mergeType = "horizontal"):
   - mergeMethod: Must be one of: "inner", "outer", "left", "right"
   - keyPairs: Array of column pairs to join on, each containing:
     * left: Column name from first table
     * right: Column name from second table

3. For vertical merges (mergeType = "vertical"):
   - No additional parameters needed
   - Tables should have matching columns

4. Optional parameters:
   - showCountSummary: boolean (true/false) to show merge statistics

Example horizontal merge:
{{
    "mergeType": "horizontal",
    "mergeMethod": "inner",
    "keyPairs": [
        {{
            "left": "Country",
            "right": "Country"
        }}
    ],
    "showCountSummary": true
}}

Example vertical merge:
{{
    "mergeType": "vertical",
    "showCountSummary": true
}}

Note:
- For horizontal merges, key columns should have matching data types
- For vertical merges, tables should have similar structure
- Common columns between tables: Compare the column lists above
"""

    elif operation_type == "replace_rename_reorder":
        template = """You are a data processing assistant. Create operations to modify the DataFrame structure and values.

Available columns and their types:
{column_info}

Data summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

DataFrame ID: {tableName}
User request: {user_input}

{format_instructions}

Your response must be structured as a function call with:
- function: "replace_rename_reorder"
- args: containing the operation parameters

Example complete response:
{{
    "function": "replace_rename_reorder",
    "args": {{
        "tableName": "{tableName}",
        "outputTableName": "{tableName}_modified",
        "operations": [
            {{
                "type": "rename_columns",
                "mapping": {{
                    "Old Name": "New Name"
                }}
            }},
            {{
                "type": "reorder_columns",
                "order": ["Column1", "Column2"]
            }},
            {{
                "type": "replace_values",
                "replacements": [
                    {{
                        "column": "Column1",
                        "oldValue": "Old",
                        "newValue": "New",
                        "matchCase": true
                    }}
                ]
            }}
        ]
    }}
}}

The operations array can include any combination of:

1. Rename Columns Operation:
   - type: "rename_columns"
   - mapping: Dictionary of old names to new names

2. Reorder Columns Operation:
   - type: "reorder_columns"
   - order: Array of column names in desired order

3. Replace Values Operation:
   - type: "replace_values"
   - replacements: Array of replacement configurations
   - Each replacement needs:
     * column: Column name
     * oldValue: Value to replace
     * newValue: New value
     * matchCase: boolean (for string replacements)

Note:
- Available columns: {column_info}
- For rename operations, old names must exist in the current columns
- For reorder operations, must include all existing columns
- For replace operations:
  * Numeric columns: {numeric_columns}
  * Text columns: {categorical_columns}
  * Match data types when replacing values
"""

    elif operation_type == "sort_filter":
        template = """You are a data processing assistant. Create filter and sort operations based on the available columns.

Available columns and their types:
{column_info}

Data summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

DataFrame ID: {tableName}
User request: {user_input}

{format_instructions}

Generate a structured sort_filter operation that specifies:
1. table_name: Name of the source table
2. filter_config: Array of filter conditions, each with:
   - column: Column name to filter on
   - operator: Must be one of these exact strings:
     * "equals"
     * "not_equals"
     * "contains"
     * "not_contains"
     * "greater_than"
     * "less_than"
     * "greater_equals"
     * "less_equals"
     * "starts_with"
     * "ends_with"
   - value: Value to compare against
3. sort_config: Array of sort conditions, each with:
   - column: Column name to sort by
   - order: "asc" or "desc"
4. output_table_name: Name for the resulting filtered/sorted table

Example response:
{{
    "table_name": "{tableName}",
    "filter_config": [
        {{
            "column": "Total",
            "operator": "greater_than",
            "value": 90
        }}
    ],
    "sort_config": [
        {{
            "column": "Year",
            "order": "desc"
        }}
    ],
    "output_table_name": "{tableName}_filtered"
}}

Note: 
- Numeric comparisons (greater_than, less_than, etc.) can only be done on: {numeric_columns}
- Text operations (contains, starts_with, etc.) can be applied to: {categorical_columns}
- Use exact operator strings as shown above
- Each filter condition must use the exact operator names listed
"""

    elif operation_type == "group_pivot":
        template = """You are a data processing assistant. Create grouping and pivot operations based on the available columns.

Available columns and their types:
{column_info}

Data summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

DataFrame ID: {tableName}
User request: {user_input}

{format_instructions}

Generate a structured group_pivot operation that specifies:
1. tableName: Name of the source table
2. rowIndex: Array of column names to use as row indices (typically categorical columns)
3. columnIndex: Column to use as column index (can be null if not needed)
4. pivotValues: Array of values to aggregate, each containing:
   - column: Column name to aggregate (must be numeric)
   - aggregation: One of: "mean", "sum", "count", "min", "max"
5. outputTableName: Name for the output pivot table

Example response:
{{
    "tableName": "{tableName}",
    "rowIndex": ["Country"],
    "columnIndex": null,
    "pivotValues": [
        {{
            "column": "Total",
            "aggregation": "mean"
        }},
        {{
            "column": "E1: Economy",
            "aggregation": "mean"
        }}
    ],
    "outputTableName": "{tableName}_pivot"
}}

Note:
- Row indices should be chosen from categorical columns: {categorical_columns}
- Values to aggregate should be chosen from numeric columns: {numeric_columns}
- Column index (if used) should be a categorical column
- Valid aggregation functions are: mean, sum, count, min, max
"""

    elif operation_type == "add_column":
        template = """You are a data processing assistant. Create a new column based on the available columns.

Available columns and their types:
{column_info}

Data summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

DataFrame ID: {tableName}
User request: {user_input}

{format_instructions}

Generate a structured add_column operation that specifies one of these formats:

1. Pattern Operation:
   - Use for extracting patterns from text
   - Required fields: tableName, newColumnName, operationType="pattern", sourceColumn, pattern

2. Calculation Operation:
   - Use for arithmetic operations
   - Required fields: tableName, newColumnName, operationType="calculate", operations list
   - Each operation needs: column1, operator (+|-|*|/), column2 or fixed_value
   - For multiple operations, use this exact structure:
   {{
     "operations": [
       {{
         "column1": "Total",
         "column2": "E1: Economy",
         "operator": "+"
       }},
       {{
         "column1": null,
         "column2": "E2: Economic Inequality",
         "operator": "+"
       }},
       {{
         "column1": null,
         "column2": null,
         "operator": "/",
         "fixed_value": 3
       }}
     ]
   }}
   - First operation must have column1 specified
   - Subsequent operations should have column1 as null to use previous result
   - Use fixed_value when you need to use a constant number
   - Each operation will be applied in sequence

3. Concatenation Operation:
   - Use for combining text
   - Required fields: tableName, newColumnName, operationType="concatenate", operations list
   - Each operation needs:
     * column: name of the column to use
     * type: "Full text" (use all characters), "Left" (use first N chars), or "Right" (use last N chars)
     * chars: number of characters to use (omit or set to null for Full text to use all characters)
   Example concatenation:
   {{
     "tableName": "example_table",
     "newColumnName": "combined_name",
     "operationType": "concatenate",
     "operations": [
       {{
         "column": "first_name",
         "type": "Full text",
         "chars": null
       }},
       {{
         "column": "last_name",
         "type": "Left",
         "chars": 1
       }}
     ]
   }}

4. Conditional Operation:
   - Use for if-then logic
   - Required fields: tableName, newColumnName, operationType="conditional", sourceColumn, conditions, residualValue
   - Each condition needs: operator, reference_value, conditional_value
   - Valid operators (use exact words):
     * "equals"
     * "does not equal"
     * "greater than"
     * "greater than or equal to"
     * "less than"
     * "less than or equal to"
     * "begins with"
     * "does not begin with"
     * "ends with"
     * "does not end with"
     * "contains"
     * "does not contain"
   Example conditional:
   {{
     "tableName": "example_table",
     "newColumnName": "status",
     "operationType": "conditional",
     "sourceColumn": "score",
     "conditions": [
       {{
         "operator": "greater than",
         "reference_value": "90",
         "conditional_value": "Excellent"
       }},
       {{
         "operator": "greater than or equal to",
         "reference_value": "70",
         "conditional_value": "Good"
       }}
     ],
     "residualValue": "Needs Improvement"
   }}

Note: 
- Numeric calculations can only be performed on: {numeric_columns}
- Text operations can be performed on: {categorical_columns}
- For concatenation, if no chars value is specified, Full text type will be used by default
- For conditional operations, always use the word form of operators (e.g., "greater than" not ">")
"""

    elif operation_type == "reconcile":
        template = """You are a data processing assistant. Create reconciliation operations for comparing and matching data between two tables.

First Table columns and their types:
{column_info}

First Table summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

Second Table columns and their types:
{table2_column_info}

Second Table summary:
- Total columns: {table2_column_count}
- Numeric columns: {table2_numeric_columns}
- Categorical columns: {table2_categorical_columns}

User request: {user_input}

{format_instructions}

Generate a structured reconcile operation that specifies:

1. Keys for matching records:
   - User will specify one column for each table 'left and 'right' as keys for the matching.
   - For each text column, user will mention a 'criteria' from the following:
     * Exact matching (default for datetime, float, or int columns)
     * Non-exact (fuzzy) matching
   - At least one key must use exact matching     
   - Key settings (defaults):
     * criteria: "exact" by default (unless explicitly mentioned as fuzzy / non-exact / approximate matching)
     * case_sensitive: "yes" by default (unless explicitly mentioned as case insensitive)
     * ignore_special: "no" by default (unless explicitly mentioned to ignore special characters)
   - For fuzzy matches:
     * If multiple fuzzy keys exist, the first fuzzy key will be marked as fuzzy_ranking_basis: true; unless the user has specified another key as the ranking basis for fuzzy matching
     * Use fuzzy matching only if mentioned

2. Values for comparison:
   - The user will specify one column from each table 'left' and 'right' (excluding key columns) as a pair, whose values need to be compared. The user can mention multiple such pairs.
   - For numeric columns (int/float):
     * Threshold settings (defaults):
       - threshold_type: "amount" by default, unless the user has mentioned "percent" as the threshold type
       - threshold_value: 0 by default, unless the user has mentioned a value. A default value of 0 means that we are setting a threshold only if explicitly mentioned
   - For non-numeric columns:
     * No threshold settings needed

3. Settings (required):
   - method: One of (default: "many-to-many"):
     * "one-to-one"
     * "one-to-many"
     * "many-to-one"
     * "many-to-many"
   - For "one-to-one" method only:
     * duplicate: Required field with one of:
       - "first_occurrence" (use the first matching record)
       - "immediately_before" (use the record immediately before the reference record)
       - "immediately_after" (use the record immediately after the reference record)
       - "closest" (use the record closest to the reference record)
     * basis_table: Required field specifying which table to use as base:
       - left table or right table (matching sourceTableNames order)
     * basis_column: one column from each table left and right

4. Cross-reference (optional, disabled by default):
   - Can be specified for both tables or any one table
   - For "one" method table (i.e. both tables in case of "one-to-one", left table in case of "one-to-many" and right table in case of "many-to-one"):
     * User can mention any columns except value columns
   - For "many" method table (i.e. both tables in case of "many-to-many", right table in case of "one-to-many" and left table in case of "many-to-one"):
     * Can only select from key columns
   - If include custom_reference metinoed  "__Custom__" as the column name for either or both tables as specified by the user

Example response:
{
    "function": "reconcile",
    "args": {
        "sourceTableNames": ["Table 1", "Table 2"],
    "outputTableName": "reconciled_risk_analysis_2007",
    "keys": [
        {
            "left": "Country",
            "right": "Country",
            "criteria": "exact",
            "case_sensitive": "no",
            "ignore_special": "yes"
            "fuzzy_ranking":"yes"
        },
        {
            "left": "Year",
            "right": "Year",
            "criteria": "exact",
            "case_sensitive": "no",
            "ignore_special": "yes"
        },
        {
            "left": "Rank",
            "right": "Rank",
            "criteria": "fuzzy",
            "case_sensitive": "no",
            "ignore_special": "yes",
            "fuzzy_ranking_basis": true
        }
    ],
    "values": [
        {
            "left": "Total",
            "right": "Total",
            "threshold_type": "percent",
            "threshold_value": 1
        },
        {
            "left": "C1: Security Apparatus",
            "right": "C1: Security Apparatus",
            "threshold_type": "amount",
            "threshold_value": 2
        },
        {
            "left": "E1: Economy",
            "right": "E1: Economy",
            "threshold_type": "percent",
            "threshold_value": 2
        },
        {
            "left": "P3: Human Rights",
            "right": "P3: Human Rights",
            "threshold_type": "percent",
            "threshold_value": 2
        }
    ],
    "settings": {
            "method": "one-to-one",
            "duplicate": "first_occurrence",
            "basis_table": "Table 1"
            "basis_column": {
                "left": "Country",
                "right": "Country"
            }
        },
    "crossReference": {
        
        "left": [
            "Country",
            "Year",
            "Rank"
        ],
        "right": [
            "Country",
            "Year",
            "Rank"
            "__Custom__"
        ]
    }
}

"""

    elif operation_type == "format":
        template = """You are a data formatting assistant. Create formatting operations for a DataFrame to make it visually appealing.

Available columns and their types:
{column_info}

Data summary:
- Total columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

DataFrame ID: {tableName}
User request: {user_input}

{format_instructions}

Your response must be structured as a function call with:
- function: "format"
- args: containing the formatting parameters

IMPORTANT: When specifying column ranges, use Excel column letters (A, B, C, D, etc.) not actual column names.
For example, use 'A,B,C' or 'A:C', NOT the actual column names like 'E1: Economy'.

Example complete response:
{{
    "function": "format",
    "args": {{
        "processId": "{process_id}",
        "tableName": "{tableName}",
        "outputTableName": "{tableName}_formatted",
        "formattingConfigs": [
            {{
                "type": "Column Width",
                "location": {{
                    "range": "A,B,C,D"
                }},
                "format": {{
                    "columnWidth": 15
                }}
            }},
            {{
                "type": "Conditional Formatting",
                "location": {{
                    "range": "B"
                }},
                "format": {{
                    "conditional": {{
                        "operator": "greater than",
                        "value": 5,
                        "trueFormat": "#FF0000",
                        "falseFormat": "#00FF00"
                    }}
                }}
            }}
        ]
    }}
}}

Valid formatting types:
1. "Cell Number Format" - Formats numbers as:
   - format: {{ "numberFormat": "number" | "percentage" | "date" }}

2. "Font Colour" - Changes text color:
   - format: {{ "fontColor": "#hexcode" }}

3. "Fill Colour" - Changes cell background:
   - format: {{ "fillColor": "#hexcode" }}

4. "Bold" - Makes text bold:
   - format: {{ "bold": true }}

5. "Column Width" - Sets column width:
   - format: {{ "columnWidth": number }}

6. "Conditional Formatting" - Applies colors based on conditions:
   - format: {{
       "conditional": {{
         "operator": "greater than" | "less than" | "equals" | "not equals" | "greater than or equal to" | "less than or equal to",
         "value": any,
         "trueFormat": "#hexcode",
         "falseFormat": "#hexcode"
       }}
     }}

Note:
- Column ranges MUST use Excel column letters (A, B, C...) NOT actual column names
- For example, use 'A,B,C' or 'A:C', NOT column names like 'E1: Economy'
- First column is A, second column is B, and so on
- Color values should be valid hex codes (e.g., #FF0000 for red)
- Both trueFormat and falseFormat MUST be provided as valid hex color codes (e.g., "#FF0000")
- Never use null or None values for color formats
- If no specific color is desired, use "#FFFFFF" (white) instead of null
- Column ranges can be comma-separated (A,B) or colon-separated ranges (A:D)
- Available columns: {column_info}
"""

    elif operation_type == "regex":
        template = """You are a regex pattern generation assistant. Create regex patterns based on natural language descriptions.

User request: {user_input}

{format_instructions}

Generate a structured regex pattern that includes:
1. pattern: The actual regex pattern
2. explanation: A brief explanation of how the pattern works

Example response:
{{
    "function": "regex",
    "args": {{
        "pattern": "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}$",
        "explanation": "Matches email addresses with alphanumeric characters, dots, underscores, and hyphens before @, followed by domain name and TLD"
    }}
}}

Note:
- Use standard regex syntax
- Include proper escaping for special characters
- Provide clear explanations of the pattern
- Consider edge cases in the pattern
- Use appropriate anchors (^, $) when needed
- Use appropriate quantifiers (*, +, ?, {{n,m}})
- Use appropriate character classes ([], [^], \\d, \\w, etc.)
- Use appropriate groups and capturing when needed
"""

    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")

    return PromptTemplate(
        template=template,
        input_variables=all_variables,
        partial_variables={"format_instructions": format_instructions}
    )

def create_chain(operation_type: str, prompt: PromptTemplate):
    """Create LangChain with appropriate parser based on operation type"""
    
    parser = PydanticOutputParser(pydantic_object=FunctionCall)
    
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not found in environment or config")

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        openai_api_key=api_key
    )

    return prompt | llm | parser 