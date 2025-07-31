import streamlit as st
import pandas as pd
from io import BytesIO

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = "sheet_selection"
if 'file_data' not in st.session_state:
    st.session_state.file_data = {}
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = {}
if 'datatypes' not in st.session_state:
    st.session_state.datatypes = {}
if 'menu_option' not in st.session_state:
    st.session_state.menu_option = "Upload Files"
if 'merge_keys' not in st.session_state:
    st.session_state.merge_keys = []

# Helper functions
def validate_file(file_name):
    """Validate the file type as Excel or CSV."""
    if file_name.endswith(('.xlsx', '.xls')):
        return 'Excel'
    elif file_name.endswith('.csv'):
        return 'CSV'
    else:
        return None

def process_excel(file):
    """Read Excel file and return sheet names."""
    try:
        excel_file = pd.ExcelFile(file)
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return []

def process_csv(file):
    """Read CSV file and return a preview."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def infer_dtype(column):
    """Infer the dtype of a column."""
    if pd.api.types.is_integer_dtype(column):
        return "integer"
    elif pd.api.types.is_float_dtype(column):
        return "float"
    elif pd.api.types.is_datetime64_any_dtype(column):
        return "datetime"
    else:
        return "string"

st.title("File Processing App")

# Side menu options
menu_options = ["Upload Files", "Merge Files", "Group and Pivot", "Sort and Filter", "Add Column", "Transform Values", "Reconcile", "Apply Formatting", "Query and Visaulize"]
st.session_state.menu_option = st.sidebar.selectbox("Select Action", menu_options)

if st.session_state.menu_option == "Upload Files":
    if st.session_state.step == "sheet_selection":
        uploaded_files = st.file_uploader("Upload files (Excel or CSV only)", type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)

        if uploaded_files:
            st.write("### Uploaded Files")

            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_type = validate_file(file_name)

                if not file_type:
                    st.write(f"- {file_name}: **This file type is not permitted**", unsafe_allow_html=True)
                else:
                    if file_name not in st.session_state.file_data:
                        st.session_state.file_data[file_name] = {'type': file_type, 'content': uploaded_file}

                        if file_type == 'Excel':
                            sheet_names = process_excel(uploaded_file)
                            st.session_state.file_data[file_name]['sheet_names'] = sheet_names

                    details = st.session_state.file_data[file_name]

                    if file_type == 'Excel':
                        st.write(f"- {file_name} ({file_type}):")
                        selected_sheets = details.get('selected_sheets', [])
                        for sheet in details['sheet_names']:
                            if st.checkbox(f"Include sheet: {sheet} ({file_name})", key=f"sheet_{file_name}_{sheet}", value=sheet in selected_sheets):
                                if sheet not in selected_sheets:
                                    selected_sheets.append(sheet)
                            else:
                                if sheet in selected_sheets:
                                    selected_sheets.remove(sheet)
                        details['selected_sheets'] = selected_sheets

                    elif file_type == 'CSV':
                        st.write(f"- {file_name} ({file_type})")
                        details['selected_sheets'] = ['CSV']

            if st.button("Next"):
                st.session_state.step = "column_selection"
                st.experimental_rerun()

    elif st.session_state.step == "column_selection":
        st.write("### File Data Summary")

        for file_name, details in st.session_state.file_data.items():
            file_type = details['type']
            selected_sheets = details.get('selected_sheets', [])

            if not selected_sheets:
                st.warning(f"No sheets selected for {file_name}")
                continue

            for sheet in selected_sheets:
                key = f"columns_{file_name}_{sheet}"
                if key not in st.session_state.selected_columns:
                    if file_type == 'Excel':
                        df = pd.read_excel(details['content'], sheet_name=sheet)
                    elif file_type == 'CSV':
                        df = pd.read_csv(details['content'])

                    st.session_state.selected_columns[key] = {'df': df, 'columns': []}

                data = st.session_state.selected_columns[key]
                df = data['df']

                st.write(f"#### {file_name} - {sheet}")
                st.write(f"Rows: {len(df)} | Columns: {list(df.columns)}")

                for column in df.columns:
                    retain_key = f"retain_{file_name}_{sheet}_{column}"
                    dtype_key = f"dtype_{file_name}_{sheet}_{column}"

                    col1, col2 = st.columns([3, 2])

                    with col1:
                        retain = st.checkbox(f"Retain column: {column}", key=retain_key, value=column in data['columns'])
                    if retain:
                        if column not in data['columns']:
                            data['columns'].append(column)
                    else:
                        if column in data['columns']:
                            data['columns'].remove(column)

                    with col2:
                        if retain:
                            inferred_dtype = infer_dtype(df[column])
                            dtype = st.selectbox(
                                f"Datatype for {column}",
                                options=["integer", "float", "string", "datetime"],
                                key=dtype_key,
                                index=["integer", "float", "string", "datetime"].index(inferred_dtype)
                            )
                            if dtype == "datetime":
                                dt_format = st.text_input(f"Datetime format for {column}", key=f"dt_format_{file_name}_{sheet}_{column}")
                                st.session_state.datatypes[dtype_key] = {"type": dtype, "format": dt_format}
                            else:
                                st.session_state.datatypes[dtype_key] = {"type": dtype}

        col1, col2 = st.columns(2)
        if col2.button("Next"):
            st.session_state.step = "review"
            st.experimental_rerun()
        if col1.button("Back"):
            st.session_state.step = "sheet_selection"
            st.experimental_rerun()

    elif st.session_state.step == "review":
        st.write("### Final DataFrames")

        for key, data in st.session_state.selected_columns.items():
            retained_columns = data['columns']
            if retained_columns:
                df = data['df'][retained_columns]

                # Apply datatypes
                for column in retained_columns:
                    dtype_key = f"dtype_{key}_{column}"
                    dtype_info = st.session_state.datatypes.get(dtype_key, {})
                    if dtype_info:
                        dtype = dtype_info.get("type")
                        if dtype == "integer":
                            try:
                                df[column] = df[column].astype(int)
                            except Exception as e:
                                st.error(f"Error converting {column} to integer: {e}")
                        elif dtype == "float":
                            try:
                                df[column] = df[column].astype(float)
                            except Exception as e:
                                st.error(f"Error converting {column} to float: {e}")
                        elif dtype == "string":
                            df[column] = df[column].astype(str)
                        elif dtype == "datetime":
                            dt_format = dtype_info.get("format", None)
                            try:
                                df[column] = pd.to_datetime(df[column], format=dt_format) if dt_format else pd.to_datetime(df[column])
                            except Exception as e:
                                st.error(f"Error converting {column} to datetime with format {dt_format}: {e}")

                st.write(f"#### {key}")
                st.dataframe(df)
                # AgGrid(df)

                # Convert to Excel and provide download link
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Sheet1")
                    writer.save()
                st.download_button(
                    label=f"Download {key}.xlsx",
                    data=output.getvalue(),
                    file_name=f"{key}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        col1, col2 = st.columns(2)
        if col2.button("Back"):
            st.session_state.step = "column_selection"
            st.experimental_rerun()

elif st.session_state.menu_option == "Merge Files":
    st.write("## Merge Files")

    # Step 1: Select files for merging
    st.write("### Step 1: Select Files")
    file_options = list(st.session_state.selected_columns.keys())
    selected_files = st.multiselect("Select two files to merge", file_options)

    if len(selected_files) != 2:
        st.warning("Please select exactly two files to proceed.")
    else:
        st.write("### Step 2: Select Merge Type")
        merge_type = st.radio("How do you want to merge the files?", ["Horizontal", "Vertical"])

        if merge_type == "Horizontal":
            # Initialize merge_keys in session state if not present
            # if "merge_keys" not in st.session_state:
            st.session_state.merge_keys = [{"left": None, "right": None}]

            st.write("### Step 3: Select Keys for Horizontal Merge")

            # Columns for the selected files
            file1_columns = st.session_state.selected_columns[selected_files[0]]["columns"]
            file2_columns = st.session_state.selected_columns[selected_files[1]]["columns"]

            # Render key selection dropdowns
            for i, key_pair in enumerate(st.session_state.merge_keys):
                col1, col2 = st.columns(2)
                
                # Dropdown for left key
                with col1:
                    st.session_state.merge_keys[i]["left"] = st.selectbox(
                        f"Key from File 1 - Pair {i+1}",
                        options=file1_columns,
                        index=file1_columns.index(key_pair["left"]) if key_pair["left"] in file1_columns else 0,
                        key=f"key_left_{i}",
                    )
                
                # Dropdown for right key
                with col2:
                    st.session_state.merge_keys[i]["right"] = st.selectbox(
                        f"Key from File 2 - Pair {i+1}",
                        options=file2_columns,
                        index=file2_columns.index(key_pair["right"]) if key_pair["right"] in file2_columns else 0,
                        key=f"key_right_{i}",
                    )

            # Button to add new key pairs
            if st.button("Add Keys"):
                st.session_state.merge_keys.append({"left": None, "right": None})
                st.experimental_rerun()

            st.write("### Step 4: Merge Method")
            merge_method = st.selectbox("Select merge method", ["left", "right", "inner", "outer"])

            show_summary = st.checkbox("Show Count Summary", value=False)

            if st.button("Submit Horizontal Merge"):
                left_df = st.session_state.selected_columns[selected_files[0]]["df"]
                right_df = st.session_state.selected_columns[selected_files[1]]["df"]

                try:
                    merged_df = pd.merge(
                        left_df,
                        right_df,
                        how=merge_method,
                        left_on=[key["left"] for key in st.session_state.merge_keys],
                        right_on=[key["right"] for key in st.session_state.merge_keys],
                        indicator=True if show_summary else False
                    )

                    st.write("### Merged DataFrame")
                    st.dataframe(merged_df)

                    if show_summary:
                        st.write("### Count Summary")
                        st.dataframe(merged_df["_merge"].value_counts())

                    # Provide download link
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        merged_df.to_excel(writer, index=False, sheet_name="Merged")
                        writer.save()
                    st.download_button(
                        label="Download Merged File",
                        data=output.getvalue(),
                        file_name="merged_file.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Error merging files: {e}")

        elif merge_type == "Vertical":
            st.write("### Step 3: Vertical Merge Configuration")
            df1 = st.session_state.selected_columns[selected_files[0]]["df"]
            df2 = st.session_state.selected_columns[selected_files[1]]["df"]

            if set(df1.columns) != set(df2.columns):
                st.error("Cannot concatenate files vertically as their columns do not match.")
            else:
                if st.button("Submit Vertical Merge"):
                    concatenated_df = pd.concat([df1, df2], ignore_index=True)
                    st.write("### Concatenated DataFrame")
                    st.dataframe(concatenated_df)

                    # Provide download link
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        concatenated_df.to_excel(writer, index=False, sheet_name="Concatenated")
                        writer.save()
                    st.download_button(
                        label="Download Concatenated File",
                        data=output.getvalue(),
                        file_name="concatenated_file.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

if st.session_state.menu_option == "Group and Pivot":
    st.write("### Group and Pivot")

    # Step 1: Select a file
    file_options = list(st.session_state.file_data.keys())
    file_to_pivot = st.selectbox("Select a file for Group and Pivot", file_options)

    if file_to_pivot:
        # Get the dataframe with selected columns
        details = st.session_state.file_data[file_to_pivot]
        sheet_name = details['selected_sheets'][0]  # Assuming one sheet per file
        key = f"columns_{file_to_pivot}_{sheet_name}"
        selected_columns = st.session_state.selected_columns[key]['columns']
        df = st.session_state.selected_columns[key]['df'][selected_columns]

        # Step 2: Select index columns
        st.write("### Select Row Index Columns")
        index_columns = st.multiselect("Choose columns for Row Index", selected_columns)

        if index_columns:
            # Exclude row index columns from further selection
            remaining_columns = [col for col in selected_columns if col not in index_columns]

            # Step 3: Select column index
            st.write("### Select Column Index (Optional)")
            remaining_columns_with_none = remaining_columns + ["None"]  # Add "None" explicitly to the options
            column_index = st.multiselect(
                "Choose columns for Column Index (Optional)",
                remaining_columns_with_none,  # Updated options with "None"
                default=[]  # Default is an empty list to avoid invalid selection
            )

            if "None" in column_index:
                column_index = None  # Convert selection to None if "None" is chosen

            # Step 4: Select values and aggregation functions
            st.write("### Add Values and Aggregation Functions")
            col1, col2 = st.columns([3, 1])

            # Initialize session state for value columns
            if "pivot_value_columns" not in st.session_state:
                st.session_state.pivot_value_columns = []

            # Add columns and aggregation functions
            with col1:
                st.write("###")
                if st.session_state.pivot_value_columns:
                    for i, pair in enumerate(st.session_state.pivot_value_columns):
                        st.selectbox(
                            f"Select Value Column {i + 1}",
                            remaining_columns,
                            key=f"value_column_{i}"
                        )
                        st.selectbox(
                            f"Select Aggregation Function {i + 1}",
                            ["count", "sum", "mean", "first", "last", "max", "min"],
                            key=f"agg_func_{i}"
                        )

            with col2:
                if st.button("Add Column"):
                    st.session_state.pivot_value_columns.append({"column": None, "aggfunc": None})

            # Prepare the pivot table
            if st.button("Submit"):
                # Build the pivot table
                value_columns = [
                    st.session_state[f"value_column_{i}"]
                    for i in range(len(st.session_state.pivot_value_columns))
                ]
                agg_functions = {
                    st.session_state[f"value_column_{i}"]: st.session_state[f"agg_func_{i}"]
                    for i in range(len(st.session_state.pivot_value_columns))
                }

                if value_columns:
                    # Step 5: Generate pivot table
    # Generate the pivot table
                    if column_index:
                        pivot = pd.pivot_table(
                            df,
                            index=index_columns,
                            columns=column_index,
                            values=value_columns,
                            aggfunc=agg_functions
                        ).reset_index()
                    else:
                        pivot = pd.pivot_table(
                            df,
                            index=index_columns,
                            values=value_columns,
                            aggfunc=agg_functions
                        ).reset_index()

                    # Display the pivot table
                    st.write("### Generated Pivot Table")
                    st.dataframe(pivot)

                    # Flatten MultiIndex columns if present
                    if isinstance(pivot.columns, pd.MultiIndex):
                        pivot.columns = ['_'.join(map(str, col)).strip('_') for col in pivot.columns]

                    # Convert to Excel and provide download button
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        pivot.to_excel(writer, index=False, sheet_name="PivotTable")
                        writer.save()

                    st.download_button(
                        label="Download Pivot Table as Excel",
                        data=output.getvalue(),
                        file_name="PivotTable.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# Sort and Filter Section
elif st.session_state.menu_option == "Sort and Filter":
    st.header("Sort and Filter")

    # File Selection
    file_options = list(st.session_state.selected_columns.keys())  # Use existing 'selected_columns'
    selected_file = st.selectbox("Select a file", file_options)

    if selected_file:
        df = st.session_state.selected_columns[selected_file]["df"]  # Retrieve the dataframe
        retained_columns = st.session_state.selected_columns[selected_file]["columns"]

        # Sort or Filter Selection
        action = st.radio("Choose an action", ["Sort", "Filter"])

        if action == "Sort":
            st.subheader("Sort")
            if "sort_columns" not in st.session_state:
                st.session_state.sort_columns = [{"key": None, "method": "Ascending"}]

            # Display existing sorting pairs
            for idx, pair in enumerate(st.session_state.sort_columns):
                col1, col2 = st.columns([2, 1])
                pair["key"] = col1.selectbox(
                    f"Sorting Key {idx + 1}",
                    options=retained_columns,  # Use retained columns
                    index=retained_columns.index(pair["key"]) if pair["key"] in retained_columns else 0,
                    key=f"sort_key_{idx}",
                )
                pair["method"] = col2.selectbox(
                    "Method",
                    ["Ascending", "Descending"],
                    index=(0 if pair["method"] == "Ascending" else 1),
                    key=f"sort_method_{idx}",
                )

            # Add Column Button
            if st.button("Add Column"):
                st.session_state.sort_columns.append({"key": None, "method": "Ascending"})

            # Submit Button
            if st.button("Submit"):
                sort_keys = [col["key"] for col in st.session_state.sort_columns]
                sort_orders = [col["method"] == "Ascending" for col in st.session_state.sort_columns]
                sorted_df = df.sort_values(by=sort_keys, ascending=sort_orders)
                st.dataframe(sorted_df)

                # Provide download option
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    sorted_df.to_excel(writer, index=False, sheet_name="SortedData")
                    writer.save()

                st.download_button(
                    label="Download Sorted File",
                    data=output.getvalue(),
                    file_name="sorted_file.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        elif action == "Filter":
            st.subheader("Filter")
            if "filter_criteria" not in st.session_state:
                st.session_state.filter_criteria = [{"column": None, "operator": "equals", "value": ""}]

            # Display existing filter criteria
            for idx, criterion in enumerate(st.session_state.filter_criteria):
                col1, col2, col3 = st.columns([2, 2, 2])
                criterion["column"] = col1.selectbox(
                    f"Column {idx + 1}",
                    options=retained_columns,  # Use retained columns
                    index=retained_columns.index(criterion["column"]) if criterion["column"] in retained_columns else 0,
                    key=f"filter_column_{idx}",
                )
                criterion["operator"] = col2.selectbox(
                    "Operator",
                    [
                        "equals", "does not equal", "greater than", "greater than or equal to",
                        "less than", "less than or equal to", "begins with", "does not begin with",
                        "ends with", "does not end with", "contains", "does not contain"
                    ],
                    key=f"filter_operator_{idx}",
                )
                criterion["value"] = col3.text_input(
                    "Value",
                    value=criterion["value"],
                    key=f"filter_value_{idx}",
                )

            # Add Criteria Button
            if st.button("Add Criteria"):
                st.session_state.filter_criteria.append({"column": None, "operator": "equals", "value": ""})

            # Submit Button
            if st.button("Submit"):
                filtered_df = df.copy()
                for criterion in st.session_state.filter_criteria:
                    col, op, val = criterion["column"], criterion["operator"], criterion["value"]
                    if op in ["equals", "does not equal", "greater than", "greater than or equal to", "less than", "less than or equal to"]:
                        val = float(val)  # Convert numerical filters
                    if op == "equals":
                        filtered_df = filtered_df[filtered_df[col] == val]
                    elif op == "does not equal":
                        filtered_df = filtered_df[filtered_df[col] != val]
                    elif op == "greater than":
                        filtered_df = filtered_df[filtered_df[col] > val]
                    elif op == "greater than or equal to":
                        filtered_df = filtered_df[filtered_df[col] >= val]
                    elif op == "less than":
                        filtered_df = filtered_df[filtered_df[col] < val]
                    elif op == "less than or equal to":
                        filtered_df = filtered_df[filtered_df[col] <= val]
                    elif op == "begins with":
                        filtered_df = filtered_df[filtered_df[col].str.startswith(val)]
                    elif op == "does not begin with":
                        filtered_df = filtered_df[~filtered_df[col].str.startswith(val)]
                    elif op == "ends with":
                        filtered_df = filtered_df[filtered_df[col].str.endswith(val)]
                    elif op == "does not end with":
                        filtered_df = filtered_df[~filtered_df[col].str.endswith(val)]
                    elif op == "contains":
                        filtered_df = filtered_df[filtered_df[col].str.contains(val)]
                    elif op == "does not contain":
                        filtered_df = filtered_df[~filtered_df[col].str.contains(val)]

                st.dataframe(filtered_df)

                # Provide download option
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name="FilteredData")
                    writer.save()

                st.download_button(
                    label="Download Filtered File",
                    data=output.getvalue(),
                    file_name="filtered_file.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )



