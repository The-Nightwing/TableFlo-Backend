from flask import Blueprint, request, jsonify
from models import User, UserProcess, ProcessOperation, ProcessFileKey, DataFrame, db, DataFrameOperation, DataFrameBatchOperation, ProcessRun, OperationType, MergeSubType, FormattingStep, Visualization
from firebase_config import get_storage_bucket
import json
from datetime import datetime
import pandas as pd
from io import BytesIO
import os
import traceback
from formatting import format_excel_file
from edit_file import edit_file
from sort_filter import process_sort_filter_data
from group_pivot import process_pivot_table
from add_column import process_add_column
from merge_files import process_merge_tables, process_reconciliation
from file_operations import process_file_operations, process_dataframe_operations
import copy
import time
from edit_file import process_columns_and_types
from formatting import process_dataframe_formatting
from merge_files import process_dataframe_reconciliation
run_process_bp = Blueprint('run_process', __name__, url_prefix='/api/run-process/')


@run_process_bp.route('/<process_id>/create-run', methods=['POST'])
def create_process_run(process_id):
    """Create a new run of an existing process with auto-incrementing name."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get original process and verify ownership
        original_process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not original_process:
            return jsonify({"error": "Original process not found or access denied"}), 404

        # Get original dataframe names
        original_dataframes = DataFrame.query.filter_by(
            process_id=process_id,
            is_originally_uploaded=True,
            is_active=True
        ).with_entities(DataFrame.name).all()
        
        dataframe_names = [df.name for df in original_dataframes]

        # Count existing runs of this process
        existing_runs = ProcessRun.query.filter_by(process_id=process_id).count()
        new_run_number = existing_runs + 1
        new_process_name = f"{original_process.process_name} - Run {new_run_number}"

        # Create new non-original process
        new_process = UserProcess(
            user_id=user.id,
            process_name=new_process_name,
            file_mappings=original_process.file_mappings,
            file_metadata=original_process.file_metadata,
            is_original=False,
            original_process_id=original_process.id
        )

        db.session.add(new_process)
        db.session.commit()

        # Create new process run
        process_run = ProcessRun(
            process_id=new_process.id,
            original_process_id=process_id,
            run_number=new_run_number,
            dataframe_mappings={
                "original_dataframes": dataframe_names,
                "run_dataframes": {}
            }
        )

        db.session.add(process_run)
        db.session.commit()

        # Create directory structure in Firebase Storage
        bucket = get_storage_bucket()
        base_path = f"{email}/process/{process_id}/runs/{process_run.id}"
        required_dirs = [
            f"{base_path}/dataframes/",
            f"{base_path}/metadata/",
            f"{base_path}/previews/"
        ]

        # Create empty placeholder files
        for dir_path in required_dirs:
            placeholder = bucket.blob(f"{dir_path}.placeholder")
            placeholder.upload_from_string('')

        return jsonify({
            "success": True,
            "message": f"Process run {new_run_number} created successfully",
            "run": {
                "id": process_run.id,
                "processId": process_run.process_id,
                "originalProcessId": process_run.original_process_id,
                "runNumber": process_run.run_number,
                "dataframeMappings": process_run.dataframe_mappings,
                "createdAt": process_run.created_at.isoformat(),
                "updatedAt": process_run.updated_at.isoformat()
            },
            "process": {
                "id": new_process.id,
                "name": new_process.process_name,
                "userId": user.id,
                "createdAt": new_process.created_at.isoformat(),
                "updatedAt": new_process.updated_at.isoformat(),
                "isOriginal": new_process.is_original,
                "originalProcessId": new_process.original_process_id
            },
            "originalDataframes": dataframe_names
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500


@run_process_bp.route('/<process_id>/validate-dataframes', methods=['POST'])
def validate_run_dataframes(process_id):
    """Validate that the provided dataframes match the structure of original process dataframes."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get original process
        original_process = UserProcess.query.get(process.original_process_id)
        if not original_process:
            return jsonify({"error": "Original process not found"}), 404

        data = request.json
        dataframe_mappings = data.get('dataframeMappings', {})
        
        if not dataframe_mappings:
            return jsonify({"error": "Dataframe mappings are required"}), 400

        # Get the ProcessRun object using process_id
        process_run = ProcessRun.query.filter_by(
            process_id=process_id
        ).order_by(ProcessRun.created_at.desc()).first()
        
        if not process_run:
            return jsonify({"error": "No run found for this process"}), 404

        # Get original dataframes with their metadata
        original_dataframes = DataFrame.query.filter_by(
            process_id=original_process.id,  # Use original process ID
            is_originally_uploaded=True,
            is_active=True
        ).all()

        # Create a mapping of original dataframe names to their metadata
        original_df_metadata = {df.name: df.data_metadata for df in original_dataframes}

        # Validate each provided dataframe against original
        validation_results = {}
        valid_mappings = {}  # Store valid mappings for updating ProcessRun

        for original_name, new_df_id in dataframe_mappings.items():
            try:
                # Check if original dataframe exists
                if original_name not in original_df_metadata:
                    validation_results[original_name] = {
                        "isValid": False,
                        "error": f"Original dataframe '{original_name}' not found"
                    }
                    continue

                # Get new dataframe
                new_df = DataFrame.query.get(new_df_id)
                if not new_df:
                    validation_results[original_name] = {
                        "isValid": False,
                        "error": f"New dataframe with ID '{new_df_id}' not found"
                    }
                    continue

                original_metadata = original_df_metadata[original_name]
                new_metadata = new_df.data_metadata

                # Compare columns
                original_columns = set(col['name'] for col in original_metadata.get('columns', []))
                new_columns = set(col['name'] for col in new_metadata.get('columns', []))

                missing_columns = original_columns - new_columns
                extra_columns = new_columns - original_columns

                if missing_columns or extra_columns:
                    validation_results[original_name] = {
                        "isValid": False,
                        "missingColumns": list(missing_columns),
                        "extraColumns": list(extra_columns),
                        "error": "Column mismatch detected"
                    }
                    continue

                # Compare column types
                original_types = {col['name']: col['type'] for col in original_metadata.get('columns', [])}
                new_types = {col['name']: col['type'] for col in new_metadata.get('columns', [])}
                type_mismatches = []

                for col in original_columns:
                    if original_types.get(col) != new_types.get(col):
                        type_mismatches.append({
                            "column": col,
                            "originalType": original_types.get(col),
                            "newType": new_types.get(col)
                        })

                if type_mismatches:
                    validation_results[original_name] = {
                        "isValid": False,
                        "typeMismatches": type_mismatches,
                        "error": "Column type mismatches detected"
                    }
                    continue

                # If we get here, validation passed
                validation_results[original_name] = {
                    "isValid": True,
                    "message": "Dataframe structure matches original",
                    "originalName": original_name,
                    "newName": new_df.name
                }

                # Store valid mapping
                valid_mappings[original_name] = {
                    "id": new_df.id,
                    "name": new_df.name,
                    "storagePath": new_df.storage_path
                }

            except Exception as e:
                validation_results[original_name] = {
                    "isValid": False,
                    "error": f"Validation error: {str(e)}"
                }

        # Check overall validation status
        is_valid = all(result["isValid"] for result in validation_results.values())

        if is_valid:
            # Update ProcessRun with validated mappings
            process_run.dataframe_mappings = {
                "original_dataframes": list(valid_mappings.keys()),
                "run_dataframes": valid_mappings
            }
            # Save changes to database
            db.session.add(process_run)
            db.session.commit()

        return jsonify({
            "success": True,
            "isValid": is_valid,
            "validations": validation_results,
            "message": "All dataframes valid and mappings updated" if is_valid else "Some dataframes have validation errors",
            "run": process_run.to_dict() if is_valid else None
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500


@run_process_bp.route('/<process_id>/execute', methods=['POST'])
def execute_process_run(process_id):
    """Execute process operations in sequence."""
    try:
        email = request.headers.get("X-User-Email")
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user and verify ownership
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get process and verify ownership
        process = UserProcess.query.filter_by(id=process_id, user_id=user.id).first()
        if not process:
            return jsonify({"error": "Process not found or access denied"}), 404

        # Get original process
        original_process = UserProcess.query.get(process.original_process_id)
        if not original_process:
            return jsonify({"error": "Original process not found"}), 404

        # Get the ProcessRun object using process_id
        process_run = ProcessRun.query.filter_by(
            process_id=process_id,
            original_process_id=original_process.id
        ).order_by(ProcessRun.created_at.desc()).first()
        
        if not process_run:
            return jsonify({"error": "No run found for this process"}), 404

        # Get dataframe mappings from the run
        dataframe_mappings = process_run.dataframe_mappings.get('run_dataframes', {})
        print("Initial mappings:", dataframe_mappings)

        # Get operations from original process in sequence
        operations = ProcessOperation.query.filter_by(
            process_id=original_process.id,
            is_active=True
        ).order_by(ProcessOperation.sequence).all()

        if not operations:
            return jsonify({"error": "No operations found for this process"}), 404

        # Initialize counters
        completed = 0
        errors = 0
        skipped = 0
        total_operations = len(operations)

        # Execute operations and collect results
        execution_results = []
        for op in operations:
            operation_result = {
                "sequence": float(op.sequence),
                "operation_name": op.operation_name,
                "parameters": op.parameters,
                "status": "pending"
            }

            try:
                result = None

                if op.operation_name == 'edit_file':
                    try:
                        batch_operation_id = op.parameters.get('batchOperationId')
                        if not batch_operation_id:
                            raise ValueError("Batch operation ID not found in parameters")

                        batch_op = DataFrameBatchOperation.query.get(batch_operation_id)
                        if not batch_op:
                            raise ValueError("Batch operation not found")

                        tables_results = []
                        success_count = 0
                        total_tables = len(batch_op.payload.get('tables', []))

                        for table_config in batch_op.payload.get('tables', []):
                            try:
                                original_table = table_config['tableName']
                                print(f"Processing table: {original_table}")
                                print(f"Available mappings: {dataframe_mappings}")

                                if original_table not in dataframe_mappings:
                                    raise ValueError(f"Table {original_table} not found in mappings")

                                mapped_table = DataFrame.query.get(dataframe_mappings[original_table]['id'])
                                if not mapped_table:
                                    raise ValueError(f"Mapped DataFrame not found for {original_table}")

                                config = table_config.copy()
                                config['tableName'] = mapped_table.name

                                result = process_columns_and_types(
                                    email=email,
                                    process_id=process_id,
                                    table_name=mapped_table.name,
                                    column_selections=config.get('columnSelections'),
                                    column_types=config.get('columnTypes'),
                                    datetime_formats=config.get('datetimeFormats')
                                )

                                if result.get('success'):
                                    success_count += 1

                                tables_results.append({
                                    "tableName": original_table,
                                    "success": result.get('success', False),
                                    "error": result.get('error'),
                                    "dataframeId": result.get('id'),
                                    #"metadata": result.get('metadata'),
                                    #"rowCount": result.get('rowCount', 0),
                                    #"columnCount": result.get('columnCount', 0)
                                })

                            except Exception as e:
                                print(f"Error processing table {original_table}: {str(e)}")
                                tables_results.append({
                                    "tableName": original_table,
                                    "success": False,
                                    "error": str(e)
                                })

                        operation_result.update({
                            "status": "completed" if success_count > 0 else "error",
                            # "batch_operation": {
                            #     "id": batch_op.id,
                            #     "tables_results": tables_results,
                            #     "success_count": success_count,
                            #     "total_count": total_tables
                            # }
                        })

                        if success_count > 0:
                            completed += 1
                        else:
                            errors += 1

                        # Update mappings with any new DataFrames created
                        if operation_result.get('status') == 'completed':
                            for table_result in operation_result['batch_operation']['tables_results']:
                                if table_result.get('success') and table_result.get('dataframeId'):
                                    dataframe_mappings[table_result['tableName']] = {
                                        'id': table_result['dataframeId'],
                                        'metadata': table_result.get('metadata', {})
                                    }

                    except Exception as e:
                        print(f"Error in edit_file operation: {str(e)}")
                        operation_result.update({
                            "status": "error",
                            "error": f"Edit file operation failed: {str(e)}"
                        })
                        errors += 1

                elif op.operation_name == 'add_column':
                    try:
                        df_operation = DataFrameOperation(
                            process_id=process_id,
                            dataframe_id=dataframe_mappings[op.parameters.get('tableName')]['id'],
                            operation_type=OperationType.ADD_COLUMN.value,
                            operation_subtype=op.parameters.get('operationType'),
                            payload=op.parameters
                        )
                        db.session.add(df_operation)
                        db.session.commit()

                        result = process_add_column(
                            email=email,
                            process_name=process.id,
                            table_name=op.parameters.get('tableName'),
                            new_column_name=op.parameters.get('newColumnName'),
                            operation_type=op.parameters.get('operationType'),
                            operation_params=op.parameters
                        )

                        if result.get('success'):
                            operation_result.update({
                                "status": "completed",
                                "operation_details": {
                                    "id": df_operation.id,
                                    "sourceTable": op.parameters.get('tableName'),
                                    "dataframeId": result.get('id'),
                                    "message": result.get('message')
                                }
                            })
                            df_operation.set_success()
                            completed += 1
                        else:
                            operation_result.update({
                                "status": "error",
                                "error": result.get('error'),
                                "operationId": df_operation.id
                            })
                            df_operation.set_error(result.get('error'))
                            errors += 1
                        
                    except Exception as e:
                        print(f"Error in add_column operation: {str(e)}")
                        operation_result.update({
                            "status": "error",
                            "error": f"Add column operation failed: {str(e)}"
                        })

                elif op.operation_name == 'merge_files':
                    try:
                        # Debug logging
                        print(f"Merging tables: {op.parameters.get('table1Name')} and {op.parameters.get('table2Name')}")
                        print(f"Available mappings: {dataframe_mappings}")
                        
                        table1_name = op.parameters.get('table1Name')
                        table2_name = op.parameters.get('table2Name')
                        
                        if table1_name not in dataframe_mappings or table2_name not in dataframe_mappings:
                            raise ValueError(f"Tables not found in mappings: {table1_name}, {table2_name}")
                        
                        mapped_df1 = DataFrame.query.get(dataframe_mappings[table1_name]['id'])
                        mapped_df2 = DataFrame.query.get(dataframe_mappings[table2_name]['id'])
                        
                        if not mapped_df1 or not mapped_df2:
                            raise ValueError("One or both mapped DataFrames not found")
                        
                        # Handle merge files with two table mappings
                        table1_name = op.parameters.get('table1Name')
                        table2_name = op.parameters.get('table2Name')
                        
                        # Check both tables in mappings
                        if table1_name not in dataframe_mappings or table2_name not in dataframe_mappings:
                            raise ValueError(f"One or both tables not found in mappings: {table1_name}, {table2_name}")
                        
                        mapped_df1 = DataFrame.query.get(dataframe_mappings[table1_name]['id'])
                        mapped_df2 = DataFrame.query.get(dataframe_mappings[table2_name]['id'])
                        
                        if not mapped_df1 or not mapped_df2:
                            raise ValueError("One or both mapped DataFrames not found")
                        
                        # Update parameters with mapped names
                        params = op.parameters.copy()
                        params['table1Name'] = mapped_df1.name
                        params['table2Name'] = mapped_df2.name

                        # Create operation record
                        df_operation = DataFrameOperation(
                            process_id=process_id,
                            dataframe_id=mapped_df1.id,
                            operation_type=OperationType.MERGE_FILES.value,
                            operation_subtype=params.get('mergeMethod', 'inner'),
                            payload=params
                        )
                        db.session.add(df_operation)
                        db.session.commit()

                        result = process_merge_tables(
                            email=email,
                            process_id=process_id,
                            table1=mapped_df1,
                            table2=mapped_df2,
                            merge_type=params.get('mergeType'),
                            merge_method=params.get('mergeMethod', 'inner'),
                            key_pairs=params.get('keyPairs', []),
                            show_count_summary=params.get('showCountSummary', False),
                            output_table_name=params.get('outputTableName', '').strip(),
                            existing_df=None
                        )

                        if result.get('success'):
                            # Add the new merged DataFrame to mappings
                            output_table_name = result.get('name')
                            dataframe_mappings[output_table_name] = {
                                'id': result['id'],
                                'metadata': result.get('metadata', {})
                            }
                            operation_result.update({
                                "status": "completed",
                                "operation_details": {
                                    "id": df_operation.id,
                                    "table1": mapped_df1.name,
                                    "table2": mapped_df2.name,
                                    "outputTableName": output_table_name,
                                    "dataframeId": result['id'],
                                    "message": result.get('message')
                                }
                            })
                            df_operation.set_success()
                            completed += 1
                        else:
                            operation_result.update({
                                "status": "error",
                                "error": result.get('error'),
                                "operationId": df_operation.id
                            })
                            df_operation.set_error(result.get('error'))
                            errors += 1

                    except Exception as e:
                        print(f"Error in merge_files operation: {str(e)}")
                        operation_result.update({
                            "status": "error",
                            "error": f"Merge operation failed: {str(e)}"
                        })

                elif op.operation_name in ['group_pivot', 'sort_filter', 'replace_rename_reorder']:
                    try:
                        # Common validation for all these operations
                        output_table_name = op.parameters.get('outputTableName', '').strip()
                        if not output_table_name:
                            raise ValueError("Output table name is required")

                        # Get source DataFrame
                        source_df = DataFrame.query.get(dataframe_mappings[op.parameters.get('tableName')]['id'])
                        if not source_df:
                            raise ValueError(f"Source DataFrame not found: {op.parameters.get('tableName')}")

                        # Create operation record with appropriate type
                        operation_type = {
                            'group_pivot': OperationType.GROUP_PIVOT.value,
                            'sort_filter': OperationType.SORT_FILTER.value,
                            'replace_rename_reorder': OperationType.REPLACE_RENAME_REORDER.value
                        }[op.operation_name]

                        df_operation = DataFrameOperation(
                            process_id=process_id,
                            dataframe_id=source_df.id,
                            operation_type=operation_type,
                            operation_subtype=op.operation_name,
                            payload=op.parameters
                        )
                        db.session.add(df_operation)
                        db.session.commit()

                        # Execute specific operation
                        if op.operation_name == 'group_pivot':
                            result = process_pivot_table(
                                email=email,
                                process_id=process_id,
                                source_table_name=op.parameters.get('tableName'),
                                row_index=op.parameters.get('rowIndex'),
                                column_index=op.parameters.get('columnIndex'),
                                pivot_values=op.parameters.get('pivotValues'),
                                output_table_name=output_table_name
                            )
                        elif op.operation_name == 'sort_filter':
                            result = process_sort_filter_data(
                                email=email,
                                process_id=process_id,
                                source_df=source_df,
                                sort_config=op.parameters.get('sortConfig', []),
                                filter_config=op.parameters.get('filterConfig', []),
                                output_table_name=output_table_name,
                                existing_df=None
                            )
                        else:  # replace_rename_reorder
                            result = process_dataframe_operations(
                                email=email,
                                process_id=process_id,
                                source_df=source_df,
                                operations=op.parameters.get('operations', []),
                                output_table_name=output_table_name,
                                existing_df=None
                            )

                        if result.get('success'):
                            # Add the new DataFrame to mappings
                            output_table_name = result.get('name')
                            dataframe_mappings[output_table_name] = {
                                'id': result['id'],
                                'metadata': result.get('metadata', {})
                            }
                            operation_result.update({
                                "status": "completed",
                                "operation_details": {
                                    "id": df_operation.id,
                                    "sourceTable": op.parameters.get('tableName'),
                                    "outputTableName": output_table_name,
                                    "dataframeId": result['id'],
                                    "rowCount": result.get('rowCount'),
                                    "columnCount": result.get('columnCount'),
                                    "message": result.get('message')
                                }
                            })
                            df_operation.set_success()
                            completed += 1
                        else:
                            operation_result.update({
                                "status": "error",
                                "error": result.get('error'),
                                "operationId": df_operation.id
                            })
                            df_operation.set_error(result.get('error'))
                            errors += 1

                    except Exception as e:
                        print(f"Error in {op.operation_name} operation: {str(e)}")
                        operation_result.update({
                            "status": "error",
                            "error": f"{op.operation_name} operation failed: {str(e)}"
                        })

                elif op.operation_name == 'reconcile_files':
                    try:
                        # Debug logging
                        print(f"Reconciling tables: {op.parameters.get('sourceTableNames')}")
                        print(f"Available mappings: {dataframe_mappings}")
                        
                        # Get source DataFrames
                        source_dfs = []
                        for table_name in op.parameters.get('sourceTableNames', []):
                            if table_name not in dataframe_mappings:
                                raise ValueError(f"Table not found in mappings: {table_name}")
                            
                            mapped_df = DataFrame.query.get(dataframe_mappings[table_name]['id'])
                            if not mapped_df:
                                raise ValueError(f"Mapped DataFrame not found for {table_name}")
                            
                            source_dfs.append(mapped_df)

                        print(f"Source DataFrames: {source_dfs}")   
                        if len(source_dfs) != 2:
                            raise ValueError("Exactly two source tables are required for reconciliation")

                        # Create operation record
                        df_operation = DataFrameOperation(
                            process_id=process_id,
                            dataframe_id=source_dfs[0].id,  # Use first source table as reference
                            operation_type=OperationType.RECONCILE_FILES.value,
                            operation_subtype="reconcile",
                            payload=op.parameters
                        )
                        db.session.add(df_operation)
                        db.session.commit()

                        # Process reconciliation
                        result = process_dataframe_reconciliation(
                            email=email,
                            process_id=process_id,
                            source_dfs=source_dfs,
                            keys=op.parameters.get('keys', []),
                            values=op.parameters.get('values', []),
                            settings=op.parameters.get('settings', {}),
                            cross_reference=op.parameters.get('crossReference', {}),
                            output_table_name=op.parameters.get('outputTableName', '').strip(),
                            existing_df=None
                        )

                        if result.get('success'):
                            # Add the reconciled DataFrame to mappings
                            output_table_name = op.parameters.get('outputTableName')
                            dataframe_mappings[output_table_name] = {
                                'id': result.get('id'),
                                'metadata': result.get('metadata', {})
                            }
                            operation_result.update({
                                "status": "completed",
                                "operation_details": {
                                    "id": df_operation.id,
                                    "sourceTables": [df.name for df in source_dfs],
                                    "outputTableName": output_table_name,
                                    "dataframeId": result.get('id'),
                                    "statistics": result.get('statistics'),
                                    "message": "Reconciliation completed successfully"
                                }
                            })
                            df_operation.set_success()
                            completed += 1
                        else:
                            operation_result.update({
                                "status": "error",
                                "error": result.get('error'),
                                "operationId": df_operation.id
                            })
                            df_operation.set_error(result.get('error'))
                            errors += 1

                    except Exception as e:
                        print(f"Error in reconcile operation: {str(e)}")
                        operation_result.update({
                            "status": "error",
                            "error": f"Reconciliation operation failed: {str(e)}"
                        })
                        errors += 1

                else:
                    operation_result.update({
                        "status": "skipped",
                        "message": f"Operation type '{op.operation_name}' execution not implemented yet"
                    })
                    skipped += 1

                # Update execution results
                execution_results.append(operation_result)

                # After each operation, update the process run with new mappings
                process_run.dataframe_mappings = {'run_dataframes': dataframe_mappings}
                db.session.commit()
                
                print(f"Updated mappings after operation {op.sequence}:", dataframe_mappings)

            except Exception as e:
                print(f"Unexpected error in operation execution: {str(e)}")
                operation_result.update({
                    "status": "error",
                    "error": f"Unexpected error: {str(e)}"
                })
                errors += 1
                execution_results.append(operation_result)

        # After all operations are complete, execute formatting steps
        formatting_results = []
        try:
            # Get formatting steps that are included in process
            formatting_steps = FormattingStep.query.filter_by(
                process_id=original_process.id,  # Use original process ID
                include_in_process=True
            ).all()

            if formatting_steps:
                for step in formatting_steps:
                    try:
                        # Get source DataFrame from mappings
                        source_df = DataFrame.query.get(step.source_dataframe_id)
                        if not source_df:
                            formatting_results.append({
                                "stepId": step.id,
                                "success": False,
                                "error": f"Source DataFrame not found: {step.source_dataframe_id}"
                            })
                            continue

                        # Extract output table name from configuration
                        output_table_name = step.configuration.get('outputTableName', f"formatted_{source_df.name}")
                        
                        # Execute formatting
                        result = process_dataframe_formatting(
                            email=email,
                            process_id=process_id,  # Use run process ID
                            source_df=source_df,
                            formatting_configs=step.configuration.get('formattingConfigs', []),
                            output_table_name=output_table_name
                        )

                        if result.get('success'):
                            formatting_results.append({
                                "stepId": step.id,
                                "success": True,
                                "outputTableName": output_table_name,
                                "downloadUrl": result.get('downloadUrl'),
                                "rowCount": result.get('rowCount'),
                                "columnCount": result.get('columnCount'),
                                "sourceTable": source_df.name
                            })
                        else:
                            formatting_results.append({
                                "stepId": step.id,
                                "success": False,
                                "error": result.get('error'),
                                "sourceTable": source_df.name
                            })

                    except Exception as e:
                        formatting_results.append({
                            "stepId": step.id,
                            "success": False,
                            "error": str(e),
                            "sourceTable": source_df.name if 'source_df' in locals() else None
                        })

        except Exception as e:
            print(f"Error executing formatting steps: {str(e)}")
            formatting_results.append({
                "success": False,
                "error": f"Error executing formatting steps: {str(e)}"
            })

        # Calculate formatting summary
        total_formatting_steps = len(formatting_steps) if 'formatting_steps' in locals() else 0
        successful_formatting_steps = sum(1 for result in formatting_results if result.get('success'))

        # Get visualizations for the process
        visualizations = Visualization.query.filter_by(
            process_id=original_process.id  # Use original process ID
        ).all()

        visualization_configs = []
        if visualizations:
            for vis in visualizations:
                visualization_configs.append({
                    "id": vis.id,
                    "configuration": vis.configuration,
                    "createdAt": vis.created_at.isoformat() if vis.created_at else None,
                    "updatedAt": vis.updated_at.isoformat() if vis.updated_at else None
                })
        # Get all dataframes associated with this process after execution
        dataframes = DataFrame.query.filter_by(process_id=process_id).all()
        bucket = get_storage_bucket()
        dataframe_list = [{
            "id": df.id,
            "name": df.name,
            "downloadUrl": bucket.blob(df.storage_path).generate_signed_url(expiration=604800, version='v4') if df.storage_path else None,
            #"rowCount": df.row_count,
            #"columnCount": df.column_count,
            #"isOriginallyUploaded": df.is_originally_uploaded,
            #"isActive": df.is_active,
            #"createdAt": df.created_at.isoformat() if df.created_at else None,
            #"updatedAt": df.updated_at.isoformat() if df.updated_at else None,
            #"metadata": df.data_metadata,
            #"storagePath": df.storage_path
        } for df in dataframes]

        # Return combined results including visualizations
        return jsonify({
            "success": True,
            "message": "Process operations, formatting, and visualizations retrieved",
            "dataframes": dataframe_list,
            "process": {
                "id": process.id,
                "name": process.process_name,
                "isOriginal": process.is_original,
                "originalProcessId": process.original_process_id
            },
            #"run": process_run.to_dict(),
            "execution_results": execution_results,
            #"final_dataframe_mappings": dataframe_mappings,
            "operations_summary": {
                "total_operations": total_operations,
                "completed": completed,
                "errors": errors,
                "skipped": skipped,
                "success_rate": f"{(completed/total_operations)*100:.2f}%" if total_operations > 0 else "0%"
            },
            "formatting": {
                "results": formatting_results,
                "summary": {
                    "totalSteps": total_formatting_steps,
                    "successfulSteps": successful_formatting_steps,
                    "failedSteps": total_formatting_steps - successful_formatting_steps,
                    "successRate": f"{(successful_formatting_steps/total_formatting_steps)*100:.2f}%" if total_formatting_steps > 0 else "0%"
                }
            },
            "visualizations": {
                "count": len(visualization_configs),
                "configurations": visualization_configs
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

    
