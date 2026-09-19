"""
Logistics Analytics & Supply Chain Optimization Pipeline
Task 1: Strategic Planning & Code Setup
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def clean_logistics_data(file_path):
    """
    Cleans raw logistics shipment data.
    Removes missing coordinates and filters transit time outliers.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['delivery_latitude', 'delivery_longitude', 'shipment_weight'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Calculate transit duration in hours
    df['delivery_time_hours'] = (
        pd.to_datetime(df['actual_delivery']) - pd.to_datetime(df['dispatch_time'])
    ).dt.total_seconds() / 3600.0
    
    # Filter transit times beyond normal regional threshold (48 hours)
    df = df[df['delivery_time_hours'] < 48]
    return df

def forecast_demand(df):
    """
    Predicts regional order volume using calendar signals and promotion history.
    """
    df['day_of_year'] = df['order_date'].dt.dayofyear
    df['day_of_week'] = df['order_date'].dt.dayofweek
    
    X = df[['day_of_year', 'day_of_week', 'historical_promotions']]
    y = df['daily_order_volume']
    
    model = LinearRegression()
    model.fit(X, y)
    df['forecasted_demand'] = model.predict(X)
    return model, df

def cluster_delivery_zones(df, num_clusters=5):
    """
    Groups delivery locations into geospatial clusters to optimize routing density.
    """
    coords = df[['delivery_latitude', 'delivery_longitude']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['delivery_cluster'] = kmeans.fit_predict(coords)
    return df, kmeans.cluster_centers_

def optimize_safety_stock(lead_time_days, demand_std_dev, service_level_z=1.65):
    """
    Calculates dynamic safety stock levels based on lead time and demand variability.
    Service level z=1.65 targets a 95% SLA.
    """
    safety_stock = service_level_z * demand_std_dev * np.sqrt(lead_time_days)
    return safety_stock

if __name__ == "__main__":
    print("Logistics Data Analytics Pipeline initialized successfully.")
