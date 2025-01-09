from prophet import Prophet
import pandas as pd
import numpy as np
from cachetools import Cache as cache
from datetime import timedelta

class ProphetAIModel:
    
    def process_historical_data(self, historical_data):
        # Convert 'Date' and 'Time' columns to datetime objects, handling errors
        historical_data['DateTime'] = pd.to_datetime(historical_data['Date'] + ' ' + historical_data['Time'], errors='coerce')

        # Select the specified columns and rename 'DateTime' to 'ds'
        df = historical_data[['DateTime', 'Offered', 'Handled', 'AHT']]
        df = df.rename(columns={'DateTime': 'ds'})
        
        # Ensure proper sorting and handling
        df = df.sort_values(['ds'])
        df['ds'] = pd.to_datetime(df['ds'])  # Ensure 'ds' is in datetime format
        return df
    
    def prophet_forecast(self, data, column, start_date, end_date, interval_min):
        cache_key = f"prophet_forecast_{column}_{start_date}_{end_date}_{interval_min}"
        cached_result = cache.get(cache_key)

        if cached_result is not None:
            return cached_result

        # Prepare the data for Prophet
        df = data[['ds', column]].rename(columns={'ds': 'ds', column: 'y'})
        
        # Initialize the Prophet model with custom seasonality
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=True, 
            yearly_seasonality=True
        )
        
        # Add custom seasonalities (e.g., quarterly, 48-hour)
        model.add_seasonality(
            name='quarterly',
            period=91.25,  # 365/4 days
            fourier_order=10,
            mode='additive'  # Additive mode for quarterly seasonality
        )
        
        model.add_seasonality(
            name='48_hours',
            period=48 * 0.5,  # 48 half-hours (for 30-minute intervals)
            fourier_order=8,
            mode='multiplicative'  # Multiplicative mode for 48-hour seasonality
        )
        
        # Fit the model
        model.fit(df)
        
        # Create a future dataframe
        future_dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval_min}T')
        future = pd.DataFrame({'ds': future_dates})
        
        # Make the forecast
        forecast = model.predict(future)
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result = result.rename(columns={'ds': 'timestamp', 'yhat': column})
        
        # Cache the result
        cache.set(cache_key, result, timeout=3600)
        
        return result
    
    def process_contact_type(self, start_date, end_date, historical_data):

        # Process the historical data to prepare it for the model
        df = self.process_historical_data(historical_data)
        
        # Convert the end_date to a datetime object
        future_end_date_dt = pd.to_datetime(end_date) + timedelta(hours=23, minutes=59)
        
        # Get forecasts for both contacts and AHT
        contacts_forecast = self.prophet_forecast(df, 'Offered', start_date, future_end_date_dt, interval_min=30)
        aht_forecast = self.prophet_forecast(df, 'AHT', start_date, future_end_date_dt, interval_min=30)
        
        # Merge the forecasts for contacts and AHT
        future_df = pd.merge(contacts_forecast, aht_forecast, on='timestamp')
        
        # Ensure non-negative forecasts for Offered and AHT
        future_df['Offered'] = np.maximum(future_df['Offered'], 0)
        future_df['AHT'] = np.maximum(future_df['AHT'], 0)
        
        # Add date and time columns to the dataframe
        future_df['history_date'] = future_df['timestamp'].dt.date
        future_df['history_time'] = future_df['timestamp'].dt.time
        
        # Prepare the output data
        ct_output_data = future_df.apply(
            lambda row: {
                "time": row['history_time'].strftime('%H:%M:%S'),
                "Offered": int(row['Offered']),
                "AHT": int(row['AHT']),
                "history_date": row['history_date'].isoformat(),
                "requirements": 0,
            },
            axis=1
        ).tolist()
        
        return ct_output_data
    
    def call_me(self, start_date, end_date, historical_data):
        # Call process_contact_type with start and end dates and historical data
        output_data = self.process_contact_type(start_date, end_date, historical_data) 

        # Print the output data
        print(output_data)

# Example usage:
if __name__ == "__main__":
    # Load the historical data
    dff = pd.read_csv('que.csv')

    # Initialize the model
    model = ProphetAIModel()

    # Call the method with specific start_date, end_date, and the historical data
    model.call_me('2025-01-01', '2025-06-01', dff)
