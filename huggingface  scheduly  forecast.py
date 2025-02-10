import os
import requests
from datetime import datetime
import json
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from nixtla import NixtlaClient
import yaml

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize NixtlaClient with your API key
nixtla_client = NixtlaClient(api_key='nixak-1jDopAXEfaOielBz1ncfbHUdsxQuULpM1rrZL0dMmYILolFC1SIp6KrCQsfuArOBIazhXvamCQuPPBw6')

class ForecastRequest(BaseModel):
    timestamps: List[str]
    values: List[float]
    forecast_horizon: int = 2000
    finetune_steps: int = 1000
    freq: str
    target_col: str
    format: str = "json"  # Default format is JSON

# Custom JSON Encoder to handle Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    try:
        # 1. Data Validation and Conversion
        try:
            df = pd.DataFrame({
                'start_time': request.timestamps,
                request.target_col: request.values
            })
            # Convert string timestamps to datetime objects
            df['start_time'] = pd.to_datetime(df['start_time'], errors='raise')
            df[request.target_col] = pd.to_numeric(df[request.target_col], errors='raise')
            logger.info("Dataframe created and converted successfully")
        except ValueError as e:
            logger.error(f"Invalid data format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")

        # 2. Data Preprocessing: Handle Missing and Duplicate Timestamps
        df = df.sort_values('start_time')  # Ensure data is sorted by time
        df = df.drop_duplicates(subset=['start_time'], keep='first')  # Remove duplicate timestamps, keeping the first occurrence
        df = df.set_index('start_time')  # Set 'start_time' as index for resampling
        df = df.resample(request.freq).asfreq()  # Resample to the specified frequency, introducing NaN for missing values
        df = df.interpolate(method='linear')  # Interpolate missing values linearly
        df = df.reset_index()  # Restore 'start_time' as a column
        logger.info("Data preprocessing completed successfully")

        # 3. Nixtla API Call using NixtlaClient
        try:
            # Make forecast using NixtlaClient
            forecast = nixtla_client.forecast(
                df=df,
                h=request.forecast_horizon,
                finetune_steps=request.finetune_steps,
                time_col='start_time',
                target_col=request.target_col,
                freq=request.freq
            )
            logger.info("Nixtla API call successful")
        except Exception as e:
            logger.error(f"Error communicating with the forecasting API: {e}")
            raise HTTPException(status_code=500, detail=f"Error communicating with the forecasting API: {e}")

        # 4. Process Forecast Results
        forecast['start_time'] = forecast['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        forecast_df = pd.DataFrame(forecast)
        logger.info("Data forecast processed")

        # Convert 'start_time' to datetime format
        forecast_df['start_time'] = pd.to_datetime(forecast_df['start_time'])

        # 5. Post-process Forecast Results
        # Extract hour and day of week from the start_time
        df['hour'] = df['start_time'].dt.hour
        df['dayofweek'] = df['start_time'].dt.dayofweek  # Monday=0, Sunday=6

        # Calculate the average value for each hour and day of week
        hourly_avg = df.groupby('hour')[request.target_col].mean()
        daily_avg = df.groupby('dayofweek')[request.target_col].mean()

        # Get the forecast value column name
        forecast_value_col = [col for col in forecast_df.columns if col != 'start_time'][0]

        # Apply the learned patterns to the forecast
        forecast_df['hour'] = forecast_df['start_time'].dt.hour
        forecast_df['dayofweek'] = forecast_df['start_time'].dt.dayofweek

        # Nullify forecast values based on historical patterns
        forecast_df[forecast_value_col] = forecast_df.apply(
            lambda row: 0 if hourly_avg[row['hour']] < 1 or daily_avg[row['dayofweek']] < 1 else row[forecast_value_col],
            axis=1
        )

        # Remove temporary columns
        forecast_df = forecast_df.drop(columns=['hour', 'dayofweek'])
        df = df.drop(columns=['hour', 'dayofweek'])

        # 6. Format Results
        if request.format == "csv":
            csv_data = forecast_df.to_csv(index=False)
            return Response(content=csv_data, media_type="text/csv")
        elif request.format == "yaml":
            yaml_data = yaml.dump(forecast_df.to_dict(orient="records"))
            return Response(content=yaml_data, media_type="application/x-yaml")
        elif request.format == "json":
            result_dict = forecast_df.to_dict(orient='records')
            json_str = json.dumps(result_dict, cls=CustomJSONEncoder)
            return JSONResponse(content=json.loads(json_str))
        else:
            raise HTTPException(status_code=400, detail="Invalid format specified. Supported formats: json, csv, yaml")

    except HTTPException as http_ex:
        logger.error(f"HTTPException: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
    # FastAPI endpoint URL
    # Use environment variable if available, otherwise default to localhost and port from FASTAPI_PORT
    FASTAPI_PORT = os.environ.get("FASTAPI_PORT", "11121")
    FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}/forecast"
    
    import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io  # Import the io module
import os
import numpy as np
import yaml
from datetime import datetime
import logging
import csv  # Import the csv module
from dotenv import load_dotenv  # Import load_dotenv
from plotly.colors import n_colors # Import n_colors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from nixtla import NixtlaClient  # Import NixtlaClient

load_dotenv() # Load environment variables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI endpoint URL
# Use environment variable if available, otherwise default to localhost and port from FASTAPI_PORT
FASTAPI_PORT = os.environ.get("FASTAPI_PORT", "11121")
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}/forecast"

st.set_page_config(
    page_title="🔮 Time Series Forecasting", layout="wide", initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced visual appeal ---
st.markdown(
    """
    <style>
    /* General app background */
    .reportview-container {
        background: linear-gradient(to right, #f0f2f6, #e1e8f2) !important; /* Light background */
    }
    /* Sidebar background */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #f0f2f6, #e1e8f2) !important; /* Light sidebar */
    }
    /* Headers and text */
    h1, h2, h3, h4, h5, h6, p, div, label {
        color: #333333 !important; /* Darker text for contrast */
    }
    /* Buttons */
    .stButton>button {
        color: #007bff !important; /* Primary blue color */
        border: 2px solid #007bff !important;
        background-color: transparent !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #007bff !important;
        color: white !important;
    }
    /* Input fields */
    .stTextInput>label, .stNumberInput>label, .stSelectbox>label, .stDateInput>label {
        color: #555555 !important;
    }
    /* Add a subtle shadow to elements */
    .element-container {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        border-radius: 5px !important;
        padding: 10px !important;
        margin-bottom: 10px !important;
        background-color: rgba(255, 255, 255, 0.8) !important; /* Semi-transparent white for content boxes */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 Time Series Forecasting")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    # Hardcoded API key (not recommended for production)
    #api_key = 'nixak-1jDopAXEfaOielBz1ncfbHUdsxQuULpM1rrZL0dMmYILolFC1SIp6KrCQsfuArOBIazhXvamCQuPPBw6'
    
    horizon = st.number_input("Forecast Horizon", min_value=1, max_value=1000, value=30)
    finetune_steps = st.slider("Finetune Steps", min_value=0, max_value=2000, value=1000)
    freq = st.selectbox(
        "Frequency",
        options=['15min', '30min', 'H', '2H', '3H', '4H', '5H', '6H', '12H', 'D', 'W', 'M', 'Y'],
        index=2,
        help="Frequency of the time series data."
    )

    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your time series data (CSV, Excel, JSON, YAML)", type=["csv", "xlsx", "json", "yaml", "yml"], help="Upload a CSV, Excel, JSON, or YAML file containing your time series data."
    )

# --- Main App Logic ---
st.write("About to display the generate forecast button")  # Debugging statement
data_loaded = False
df = None

if uploaded_file is not None:
    try:
        logger.info(f"Attempting to load file: {uploaded_file.name}")
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file)
                logger.info(f"CSV file loaded successfully using Pandas. Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error parsing CSV file with Pandas: {e}")
                logger.exception(f"Error parsing CSV with Pandas: {e}")
                st.stop()

        elif file_extension == 'xlsx':
            try:
                df = pd.read_excel(uploaded_file)
                logger.info(f"Excel file loaded successfully using Pandas. Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error parsing Excel file with Pandas: {e}")
                logger.exception(f"Error parsing Excel with Pandas: {e}")
                st.stop()

        elif file_extension == 'json':
            try:
                df = pd.read_json(uploaded_file)
                logger.info(f"JSON file loaded successfully using Pandas. Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error parsing JSON file with Pandas: {e}")
                logger.exception(f"Error parsing JSON with Pandas: {e}")
                st.stop()

        elif file_extension in ['yaml', 'yml']:
            try:
                df = pd.DataFrame(yaml.safe_load(uploaded_file))
                logger.info(f"YAML file loaded successfully using Pandas. Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error parsing YAML file with Pandas: {e}")
                logger.exception(f"Error parsing YAML with Pandas: {e}")
                st.stop()

        else:
            st.error("❌ Unsupported file format. Please upload a CSV, Excel, JSON, or YAML file.")
            logger.error(f"Unsupported file format: {file_extension}")
            st.stop()

        st.success("✅ Data loaded successfully!")
        data_loaded = True

        # --- Column Selection ---
        st.sidebar.header("📊 Column Selection")
        time_col = st.sidebar.selectbox("Select Timestamp Column", df.columns, help="Column containing the timestamps.")
        value_col = st.sidebar.selectbox("Select Value Column", df.columns, help="Column containing the values to forecast.")

        if value_col == time_col:
            st.error("❌ Value column cannot be the same as the Timestamp column")
            logger.error("Value column and Timestamp column are the same.")
            st.stop()

        # --- Convert Value Column to Numeric ---
        try:
            # Convert to numeric, coercing errors
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            logger.info(f"Value column '{value_col}' converted to numeric.")

            # Handle potential NaN values (failed conversions)
            if df[value_col].isnull().any():
                st.warning(f"Some values in {value_col} could not be converted to numeric and were replaced with NaN.")
                logger.warning(f"NaN values found in value column '{value_col}'.")
                df = df.dropna(subset=[value_col])
                logger.info(f"Rows with NaN values in '{value_col}' dropped. Shape: {df.shape}")

        except Exception as e:
            st.error(f"Error converting {value_col} to numeric: {e}")
            logger.exception(f"Error converting value column to numeric: {e}")
            st.stop()

        # --- Convert Timestamp Column to Datetime ---
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            logger.info(f"Timestamp column '{time_col}' converted to datetime.")

            # Handle potential NaT values (failed conversions)
            if df[time_col].isnull().any():
                st.warning(f"Some values in {time_col} could not be converted to datetime. These rows will be dropped.")
                logger.warning(f"NaT values found in timestamp column '{time_col}'.")
                df = df.dropna(subset=[time_col])
                logger.info(f"Rows with NaT values in '{time_col}' dropped. Shape: {df.shape}")

        except Exception as e:
            st.error(f"Error converting {time_col} to datetime: {e}")
            logger.exception(f"Error converting timestamp column to datetime: {e}")
            st.stop()

        # --- Data Preview ---
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ An error occurred during data loading: {e}")
        logger.exception(f"An error occurred during data loading: {e}")
        st.stop()

if data_loaded:  # The button should ALWAYS appear if data_loaded is True
    if st.button("✨ Generate Forecast"):
        if df is not None:
            with st.spinner("⏳ Generating forecast..."):
                try:
                    # Ensure no Nulls in the data being sent to the API
                    df = df.dropna(subset=[time_col, value_col])
                    logger.info(f"Null values dropped before API call. Shape: {df.shape}")

                    # Convert timestamps to string and values to list
                    timestamps = [ts.isoformat() for ts in df[time_col]]
                    values = df[value_col].tolist()

                    payload = {
                        "timestamps": timestamps,
                        "values": values,
                        "forecast_horizon": horizon,
                        "finetune_steps": finetune_steps,
                        "freq": freq,
                        "target_col": value_col,
                        "format": "json"  # Default format
                    }

                    response = requests.post(FASTAPI_URL, json=payload)
                    response.raise_for_status()  # Raise HTTPError for bad responses
                    logger.info(f"API call successful. Status code: {response.status_code}")
                    forecast_data = response.json()

                    # Convert forecast data to DataFrame
                    forecast_df = pd.DataFrame(forecast_data)

                    # Determine the forecast value column name
                    forecast_value_col = [col for col in forecast_df.columns if col != time_col][0]

                    # Convert back to datetime for plotting
                    forecast_df[time_col] = pd.to_datetime(forecast_df[time_col])

                    # --- Plotting ---
                    st.subheader("📈 Time Series Visualization")
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Historical Data vs Forecast', 'Combined Data (Inner Join)')
                    )

                    # Historical Data
                    fig.add_trace(go.Scatter(
                        x=df[time_col],
                        y=df[value_col],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='#636EFA'),
                        showlegend=False
                    ), row=1, col=1)

                    # Forecast Data
                    fig.add_trace(go.Scatter(
                        x=forecast_df[time_col],
                        y=forecast_df[forecast_value_col],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#FFA15A'),
                        showlegend=False
                    ), row=1, col=1)

                    # Combined Data
                    fig.add_trace(go.Scatter(
                        x=df[time_col],
                        y=df[value_col],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='#636EFA'),
                        showlegend=False
                    ), row=2, col=1)
                    fig.add_trace(go.Scatter(
                        x=forecast_df[time_col],
                        y=forecast_df[forecast_value_col],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#FFA15A'),
                        showlegend=False
                    ), row=2, col=1)

                    fig.update_layout(
                        title="Time Series Forecast",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        template="plotly_white",  # Changed to white template
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # --- Forecast Data Display ---
                    st.subheader("Forecast Data")
                    st.dataframe(forecast_df)

                    # --- Download Forecast Data ---
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download forecast data as CSV",
                        data=csv,
                        file_name="forecast.csv",
                        mime="text/csv",
                    )

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Error communicating with backend: {e}")
                    logger.exception(f"Error communicating with backend: {e}")
                except Exception as e:
                    st.error(f"❌ An error occurred during forecasting: {e}")
                    logger.exception(f"Error occurred during forecasting: {e}")
        else:
            st.warning("Please upload data and select columns to generate a forecast.")


