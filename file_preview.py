from flask import Blueprint, request, jsonify
from firebase_config import get_storage_bucket
import pandas as pd
import base64
from io import BytesIO
import csv
import traceback
import os
# Create a Blueprint
preview_bp = Blueprint('preview', __name__, url_prefix='/api/preview/')
@preview_bp.route('/generate/', methods=['POST'])
def generate_preview():
    """
    Generate a quick preview of the processed file using pre-saved preview data in Firebase Storage.
    """
    data = request.json
    file_name = data.get('fileName')
    selected_sheets = data.get('selectedSheets', {})
    email = request.headers.get('X-User-Email')

    if not file_name or not selected_sheets:
        return jsonify({"error": "File name and selected sheets/columns are required."}), 400

    file_extension = os.path.splitext(file_name)[-1].lower()
    try:
        bucket = get_storage_bucket()
        preview_data = {}
        column_types = {}

        def handle_duplicate_columns(df):
            """
            Append suffix to duplicate column names to ensure uniqueness.
            """
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [
                    f"{dup}.{i+1}" if i > 0 else dup for i in range(sum(cols == dup))
                ]
            df.columns = cols
        
            return df

        # Use saved preview files for Excel
        if file_extension in ['.xls', '.xlsx']:
            for sheet_name, columns in selected_sheets.items():
                preview_blob_path = f"{email}/previews/{file_name}/{sheet_name}_preview.csv"
                preview_blob = bucket.blob(preview_blob_path)

                if not preview_blob.exists():
                    return jsonify({"error": f"Preview data for sheet '{sheet_name}' does not exist."}), 404

                # Read preview file from Firebase
                preview_content = preview_blob.download_as_text()
                df = pd.read_csv(BytesIO(preview_content.encode('utf-8')))

                # Handle duplicate columns
                df = handle_duplicate_columns(df)
                df = df.fillna("-")

                # Filter columns based on the selected ones
                valid_columns = [col for col in columns if col in df.columns]
                if not valid_columns:
                    valid_columns = df.columns.tolist()  # Default to all columns if none are valid
                filtered_df = df[valid_columns]

                # Store preview data and column types
                preview_data[sheet_name] = filtered_df.reset_index(drop=True).to_dict(orient='records')
                column_types[sheet_name] = {col: str(df[col].dtypes) for col in valid_columns}

        # Use saved preview for CSV
        if file_name.endswith('.csv') and 'CSV' in selected_sheets:
            csv_preview_path = f"{email}/previews/{file_name}/preview.csv"
            print(csv_preview_path)
            csv_preview_blob = bucket.blob(csv_preview_path)

            if not csv_preview_blob.exists():
                return jsonify({"error": "Preview data for the CSV file does not exist."}), 404

            csv_content = csv_preview_blob.download_as_text()
            df = pd.read_csv(BytesIO(csv_content.encode('utf-8')))

            # Handle duplicate columns
            df = handle_duplicate_columns(df)

            selected_columns = selected_sheets.get('columns', [])
            valid_columns = [col for col in selected_columns if col in df.columns]
            if not valid_columns:
                valid_columns = df.columns.tolist()  # Default to all columns if none are valid
            filtered_df = df[valid_columns]

            preview_data['CSV'] = filtered_df.reset_index(drop=True).to_dict(orient='records')
            column_types['CSV'] = {col: str(df[col].dtypes) for col in valid_columns}

        return jsonify({
            "success": True,
            "previewData": preview_data,
            "columnTypes": column_types
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@preview_bp.route('/generate/addColumn/', methods=['POST'])
def generate_preview_with_column():
    """
    Generate a preview of the file, allowing operations to create a new column.
    """
    data = request.json
    file_name = data.get('fileName')
    selected_sheets = data.get('selectedSheets', {})
    operations = data.get('operations', [])
    new_column_name = data.get('newColumnName', 'new_column')
    email = request.headers.get('X-User-Email')

    if not file_name or not operations:
        return jsonify({"error": "File name and operations are required."}), 400

    try:
        # Fetch the file from Firebase Storage
        bucket = get_storage_bucket()
        blob = bucket.blob(f"{email}/uploaded_files/{file_name}")
        if not blob.exists():
            return jsonify({"error": "The specified file does not exist."}), 404

        file_content = blob.download_as_bytes()
        file_type = file_name.split('.')[-1].lower()

        preview_data = {}
        if file_type in ['xls', 'xlsx']:
            excel_file = pd.ExcelFile(BytesIO(file_content))
            for sheet_name, columns in selected_sheets.items():
                if sheet_name in excel_file.sheet_names:
                    df = excel_file.parse(sheet_name, nrows=25)  # Read only 25 rows
                    valid_columns = [col for col in columns if col in df.columns]
                    if operations:
                        df[new_column_name] = eval_operations(df, operations)
                    filtered_df = df[valid_columns + [new_column_name]]
                    filtered_df = filtered_df.where(pd.notnull(filtered_df), None)
                    preview_data[sheet_name] = filtered_df.to_dict(orient='records')

        elif file_type == 'csv':
            df = pd.read_csv(BytesIO(file_content), nrows=25)  # Read only 25 rows
            selected_columns = selected_sheets.get('columns', [])
            valid_columns = [col for col in selected_columns if col in df.columns]
            if operations:
                df[new_column_name] = eval_operations(df, operations)
            filtered_df = df[valid_columns + [new_column_name]]
            filtered_df = filtered_df.where(pd.notnull(filtered_df), None)
            preview_data['CSV'] = filtered_df.to_dict(orient='records')

        else:
            return jsonify({"error": "Unsupported file format."}), 400

        return jsonify({
            "success": True,
            "previewData": preview_data,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def eval_operations(df, operations):
    """
    Evaluate operations to generate a new column.
    """
    result = None
    for op in operations:
        left_operand = df[op['leftOperand']] if op['leftOperand'] in df.columns else None
        right_operand = (
            df[op['rightOperand']]
            if op['rightOperand'] in df.columns
            else float(op['fixedValue'])
            if op['rightOperand'] == 'Fixed Value'
            else None
        )
        operator = op['operator']

        if left_operand is not None and right_operand is not None:
            temp_result = eval(f"left_operand {operator} right_operand")
        else:
            raise ValueError(f"Invalid operands in operation: {op}")

        result = temp_result if result is None else result + temp_result

    return result
