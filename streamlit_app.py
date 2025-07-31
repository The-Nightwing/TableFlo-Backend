import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from firebase_admin import storage
import pandas as pd
from datetime import datetime
import json
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

st.set_page_config(page_title="Visualization Creator", layout="wide")

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage  # Correct import

def initialize_firebase():
    """Initialize Firebase if not already initialized"""
    if not firebase_admin._apps:
        try:
            # Ensure Firebase secrets are available
            if "firebase_service_account" not in st.secrets:
                st.error("❌ Firebase credentials are missing in Streamlit secrets.")
                return False
            
            # Parse credentials from Streamlit secrets
            service_account = dict(st.secrets["firebase_service_account"])

            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(service_account)
            firebase_admin.initialize_app(cred)

            # Initialize Firestore
            st.session_state.db = firestore.client()
            #st.info("✅ Firestore client initialized.")

            # ✅ Correct way to initialize Google Cloud Storage
            storage_client = storage.Client.from_service_account_info(service_account)  
            bucket_name = "file-processing-app.firebasestorage.app"  # Ensure bucket name is correct
            st.session_state.storage_bucket = storage_client.bucket(bucket_name)

            st.success(f"✅ Firebase Storage initialized with bucket: {st.session_state.storage_bucket.name}")
            return True

        except Exception as e:
            st.error(f"🔥 Firebase initialization failed: {e}")
            return False

    # Ensure storage bucket is set even if Firebase is already initialized
    if 'storage_bucket' not in st.session_state:
        try:
            storage_client = storage.Client.from_service_account_info(st.secrets["firebase_service_account"])
            st.session_state.storage_bucket = storage_client.bucket("file-processing-app.firebasestorage.app")
        except Exception as e:
            st.error(f"🔥 Error initializing Firebase Storage: {e}")
            return False

    return True




def get_firestore_client():
    """Get Firestore client from session state"""
    if 'db' not in st.session_state:
        initialize_firebase()
    return st.session_state.db

def get_storage_bucket():
    """Ensure Firebase is initialized before accessing storage bucket"""
    if 'storage_bucket' not in st.session_state or st.session_state.storage_bucket is None:
        st.warning("Storage bucket is missing. Reinitializing Firebase...")
        if not initialize_firebase():  # Try to reinitialize Firebase
            st.error("Firebase initialization failed. Cannot get storage bucket.")
            return None  # Prevent further errors
    
    return st.session_state.storage_bucket

class VisualizationSession:
    def __init__(self, user_email):
        self.user_email = user_email
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'line_bar_series' not in st.session_state:
            st.session_state.line_bar_series = []
        if 'horizontal_bar_series' not in st.session_state:
            st.session_state.horizontal_bar_series = []
        if 'current_visualization' not in st.session_state:
            st.session_state.current_visualization = None
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = {}
            
    def load_data(self, file_name, sheet_name=None):
        """Load data from Firebase Storage"""
        bucket = get_storage_bucket()
        blob_path = f"users/{self.user_email}/data/{file_name}"
        blob = bucket.blob(blob_path)
        
        # Download to temporary file
        local_path = f"/tmp/{file_name}"
        blob.download_to_filename(local_path)
        
        if file_name.endswith('.csv'):
            return pd.read_csv(local_path)
        elif file_name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(local_path, sheet_name=sheet_name)
        else:
            raise ValueError("Unsupported file format")

    def save_visualization(self, config):
        """Save visualization config to Firebase"""
        bucket = get_storage_bucket()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_path = f"users/{self.user_email}/visualizations/viz_{timestamp}.json"
        blob = bucket.blob(blob_path)
        
        blob.upload_from_string(
            json.dumps(config),
            content_type='application/json'
        )
        return blob_path

    def render_visualization_interface(self, df):
        """Render the visualization interface"""
        st.header("Visualization")

        # Select File
        file_options = list(st.session_state.selected_columns.keys())
        selected_file = st.selectbox("Select File", file_options)
        
        if not selected_file:
            return

        # Step 1: Select Visualization Type
        visualization_type = st.radio(
            "Select the type of visualization",
            [
                "Line / Vertical Bars / Stacked Vertical Bars / Combination",
                "Horizontal Bars / Stacked Horizontal Bars",
                "Donut / Pie"
            ]
        )

        if visualization_type == "Line / Vertical Bars / Stacked Vertical Bars / Combination":
            self.render_line_bar_interface(df, selected_file)
        elif visualization_type == "Horizontal Bars / Stacked Horizontal Bars":
            self.render_horizontal_bar_interface(df, selected_file)
        elif visualization_type == "Donut / Pie":
            self.render_pie_donut_interface(df, selected_file)

    def render_line_bar_interface(self, df, selected_file):
        st.subheader("Line / Vertical Bars / Stacked Vertical Bars / Combination")

        # Chart Title
        chart_title = st.text_input("Enter Chart Title")

        # X Axis Selection
        x_axis = st.selectbox("Select X Axis", st.session_state.selected_columns[selected_file]["columns"])

        # Display existing series
        for idx, series in enumerate(st.session_state.line_bar_series):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            series["column"] = col1.selectbox(
                f"Series {idx + 1}",
                st.session_state.selected_columns[selected_file]["columns"],
                key=f"line_bar_series_column_{idx}"
            )
            series["type"] = col2.selectbox(
                "Visualization Type",
                ["Line", "Bar"],
                key=f"line_bar_series_type_{idx}"
            )
            series["axis"] = col3.selectbox(
                "Axis",
                ["Left", "Right"],
                key=f"line_bar_series_axis_{idx}"
            )
            series["color"] = col4.text_input(
                "Colour (Hex Code)",
                key=f"line_bar_series_color_{idx}"
            )

        # Add Series Button
        if st.button("Add Series", key="add_line_bar"):
            st.session_state.line_bar_series.append({
                "column": None,
                "type": "Line",
                "axis": "Left",
                "color": "#000000"
            })

        # Bar Type Selection
        bar_type = None
        if any(series["type"] == "Bar" for series in st.session_state.line_bar_series):
            bar_type = st.radio(
                "Bar Type",
                ["Stacked Bars", "Side-by-Side Bars"]
            )

        # Preview Chart
        if st.session_state.line_bar_series:
            self.preview_line_bar_chart(df, x_axis, chart_title, bar_type)

    def render_horizontal_bar_interface(self, df, selected_file):
        st.subheader("Horizontal Bars / Stacked Horizontal Bars")

        # Chart Title
        chart_title = st.text_input("Enter Chart Title")

        # X Axis Selection
        x_axis = st.selectbox("Select X Axis", st.session_state.selected_columns[selected_file]["columns"])

        # Display existing series
        for idx, series in enumerate(st.session_state.horizontal_bar_series):
            col1, col2 = st.columns([2, 2])
            series["column"] = col1.selectbox(
                f"Series {idx + 1}",
                st.session_state.selected_columns[selected_file]["columns"],
                key=f"horizontal_bar_series_column_{idx}"
            )
            series["color"] = col2.text_input(
                "Colour (Hex Code)",
                key=f"horizontal_bar_series_color_{idx}"
            )

        # Add Series Button
        if st.button("Add Series", key="add_horizontal_bar"):
            st.session_state.horizontal_bar_series.append({
                "column": None,
                "color": "#000000"
            })

        # Bar Type Selection
        bar_type = None
        if len(st.session_state.horizontal_bar_series) > 1:
            bar_type = st.radio(
                "Bar Type",
                ["Stacked Bars", "Side-by-Side Bars"]
            )

        # Preview Chart
        if st.session_state.horizontal_bar_series:
            self.preview_horizontal_bar_chart(df, x_axis, chart_title, bar_type)

    def render_pie_donut_interface(self, df, selected_file):
        st.subheader("Donut / Pie")

        # Chart Type
        chart_type = st.radio("Select Chart Type", ["Donut", "Pie"])

        # Chart Title
        chart_title = st.text_input("Enter Chart Title")

        # Labels and Values
        labels = st.selectbox("Select Labels", st.session_state.selected_columns[selected_file]["columns"])
        values = st.selectbox(
            "Select Values",
            [col for col in st.session_state.selected_columns[selected_file]["columns"] if col != labels]
        )

        # Number of Largest Items
        largest_items = st.number_input(
            "Enter number of largest items to show (Optional)",
            min_value=1,
            step=1
        )

        # Colour Theme
        colour_theme = st.selectbox(
            "Select Colour Theme",
            ["Blue-Grey", "Yellow-Green", "Red-Orange"]
        )

        # Preview Chart
        if labels and values:
            self.preview_pie_donut_chart(df, labels, values, chart_title, chart_type, largest_items, colour_theme)

    def preview_line_bar_chart(self, df, x_axis, title, bar_type):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for series in st.session_state.line_bar_series:
            if series["type"] == "Line":
                fig.add_trace(
                    go.Scatter(
                        x=df[x_axis],
                        y=df[series["column"]],
                        name=series["column"],
                        line=dict(color=series["color"])
                    ),
                    secondary_y=(series["axis"] == "Right")
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=df[x_axis],
                        y=df[series["column"]],
                        name=series["column"],
                        marker_color=series["color"]
                    ),
                    secondary_y=(series["axis"] == "Right")
                )

        if bar_type == "Stacked Bars":
            fig.update_layout(barmode='stack')
        
        fig.update_layout(title=title)
        st.plotly_chart(fig, use_container_width=True)

    def preview_horizontal_bar_chart(self, df, x_axis, title, bar_type):
        fig = go.Figure()
        
        for series in st.session_state.horizontal_bar_series:
            fig.add_trace(
                go.Bar(
                    y=df[x_axis],
                    x=df[series["column"]],
                    name=series["column"],
                    marker_color=series["color"],
                    orientation='h'
                )
            )

        if bar_type == "Stacked Bars":
            fig.update_layout(barmode='stack')
        
        fig.update_layout(title=title)
        st.plotly_chart(fig, use_container_width=True)

    def preview_pie_donut_chart(self, df, labels, values, title, chart_type, largest_items, colour_theme):
        df_grouped = df.groupby(labels)[values].sum().reset_index()
        
        if largest_items:
            df_grouped = df_grouped.nlargest(largest_items, values)
            
        fig = go.Figure(data=[
            go.Pie(
                labels=df_grouped[labels],
                values=df_grouped[values],
                hole=0.4 if chart_type == "Donut" else 0
            )
        ])
        
        fig.update_layout(title=title)
        st.plotly_chart(fig, use_container_width=True)

    def render_preview(self, df, config):
        """Render just the visualization based on config"""
        visualization_type = config.get('type')
        
        if visualization_type == "Line / Vertical Bars / Stacked Vertical Bars / Combination":
            x_axis = config.get('xAxis')
            chart_title = config.get('title', '')
            bar_type = config.get('barType')
            series = config.get('series', [])
            
            # Set the series in session state for preview
            st.session_state.line_bar_series = series
            
            # Preview the chart
            self.preview_line_bar_chart(df, x_axis, chart_title, bar_type)
            
        elif visualization_type == "Horizontal Bars / Stacked Horizontal Bars":
            x_axis = config.get('xAxis')
            chart_title = config.get('title', '')
            bar_type = config.get('barType')
            series = config.get('series', [])
            
            # Set the series in session state for preview
            st.session_state.horizontal_bar_series = series
            
            # Preview the chart
            self.preview_horizontal_bar_chart(df, x_axis, chart_title, bar_type)
            
        elif visualization_type == "Donut / Pie":
            labels = config.get('labels')
            values = config.get('values')
            chart_title = config.get('title', '')
            chart_type = config.get('chartType')
            largest_items = config.get('largestItems')
            colour_theme = config.get('colorTheme')
            
            # Preview the chart
            self.preview_pie_donut_chart(
                df, labels, values, chart_title, 
                chart_type, largest_items, colour_theme
            )

def main():
    # Set page config as the first command
    # Get parameters from URL
    # Ensure Firebase is initialized when the app starts
    query_params = st.query_params
    session_id = query_params.get("session_id", [None])
    mode = query_params.get("mode", ["full"])  # 'preview' or 'full'

    if not session_id:
        st.error("No session ID provided")
        return
    
    try:
        # Load session data from Firebase
        bucket = get_storage_bucket()

        #st.info(f"Using bucket: {bucket.name}")
        #st.info(f"session : {session_id}")
        config_blob = bucket.blob(f"streamlit_sessions/{session_id}/config.json")

        if not config_blob.exists():
            st.error(f"Config file not found: streamlit_sessions/{session_id}/config.json")
            return
        
        session_data = json.loads(config_blob.download_as_string())
        
        # Load data
        if mode == "preview":
            # For preview, load data directly from session storage
            data_blob = bucket.blob(f"streamlit_sessions/{session_id}/data.csv")
            df = pd.read_csv(BytesIO(data_blob.download_as_bytes()))
        else:
            # For full mode, load from original file
            df = load_data_from_firebase(bucket, session_data["email"], 
                                       session_data["fileName"], 
                                       session_data.get("sheetName"))

        # Initialize visualization session
        viz_session = VisualizationSession(session_data["email"])
        
        if mode == "preview":
            # For preview, just render the visualization based on config
            viz_session.render_preview(df, session_data["visualizationConfig"])
        else:
            # For full mode, render the full interface
            viz_session.render_visualization_interface(df)
            
            # Save and Create New buttons (only in full mode)
            col1, col2 = st.columns(2)
            if col1.button("Save"):
                config = {
                    "type": st.session_state.get("visualization_type"),
                    "title": st.session_state.get("chart_title"),
                    # Add other config details
                }
                saved_path = viz_session.save_visualization(config)
                st.success(f"Visualization saved successfully!")
                
            if col2.button("Create New Visualization"):
                st.session_state.line_bar_series = []
                st.session_state.horizontal_bar_series = []
                st.session_state.current_visualization = None
                st.experimental_rerun()
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 