# -*- coding: utf-8 -*-
"""
02_integrate_data.py - Data integration and feature generation for ANFIS
"""
import pandas as pd
import numpy as np
from datetime import datetime

def load_and_integrate():
    """Integrate all datasets for ANFIS model"""
    print("=" * 60)
    print("Data Integration with COVID Discount")
    print("=" * 60)
    
    # 1. Monthly demand and price data
    print("\n[1] Loading monthly demand data...")
    df_demand = pd.read_csv('../data/demand_monthly_with_price.csv')
    df_demand.rename(columns={df_demand.columns[0]: 'date_str'}, inplace=True)
    df_demand['date'] = pd.to_datetime(df_demand['date_str'], format='%Y.%m.%d')
    df_demand = df_demand.set_index('date')
    df_demand = df_demand[['Current_Demand', 'Avg_Fee', 'price_delta_pct', 'price_level']]
    print(f"Demand data: {len(df_demand)} months")
    
    # 2. Temperature and precipitation data - monthly aggregation
    print("\n[2] Loading temperature/precipitation data...")
    df_climate = pd.read_excel('../data/ComTempPrec.xlsx')
    df_climate['Date'] = pd.to_datetime(df_climate['Date'])
    df_climate = df_climate.set_index('Date')
    df_climate = df_climate.resample('MS').agg({
        'Temperature(℃)': 'mean',
        'Precipitation(mm)': 'sum'
    })
    df_climate.columns = ['Temperature', 'Precipitation']
    print(f"Climate data: {len(df_climate)} months")
    
    # 3. Population data
    print("\n[3] Loading population data...")
    df_pop = pd.read_excel('../data/population_month.xlsx')
    date_str = df_pop['Date'].astype(str)
    df_pop['year'] = date_str.str.split('.').str[0].astype(int)
    df_pop['month'] = date_str.str.split('.').str[1].astype(int)
    df_pop['date'] = pd.to_datetime(df_pop[['year', 'month']].assign(day=1))
    df_pop = df_pop.set_index('date')
    df_pop = df_pop[['population']]
    print(f"Population data: {len(df_pop)} months")
    
    # 4. Data merge
    print("\n[4] Merging data...")
    # Remove duplicates
    df_demand = df_demand[~df_demand.index.duplicated(keep='first')]
    df_climate = df_climate[~df_climate.index.duplicated(keep='first')]
    df_pop = df_pop[~df_pop.index.duplicated(keep='first')]
    
    df = df_demand.copy()
    df = df.join(df_climate, how='left')
    df = df.join(df_pop, how='left')
    
    # 5. Feature generation
    print("\n[5] Generating ANFIS model features...")
    
    # Seasonality
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Lag features (previous month demand)
    df['Prev_Demand'] = df['Current_Demand'].shift(1)
    
    # effective_fee: normalize by average
    df['effective_fee'] = df['Avg_Fee'] / df['Avg_Fee'].mean()
    
    # Normalized population
    df['population_norm'] = df['population'] / df['population'].iloc[0]
    
    # Month cycle encoding (sine, cosine)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 6. COVID discount (50% for small businesses, 2021.07-2022.06)
    print("\n[6] Adding COVID discount variable...")
    df['covid_discount'] = 0.0
    df['covid_discount'] = df['covid_discount'].astype(float)
    
    # Discount period: 2021.07 - 2022.06
    discount_start = pd.Timestamp('2021-07-01')
    discount_end = pd.Timestamp('2022-06-30')
    
    # Commercial sector ratio approx 23% (General + Bathhouse)
    commercial_ratio = 0.23
    # 50% discount applied only to commercial sector
    weighted_discount = 0.5 * commercial_ratio
    
    df.loc[(df.index >= discount_start) & (df.index <= discount_end), 'covid_discount'] = weighted_discount
    
    # Adjusted effective fee
    df['effective_fee_adjusted'] = df['effective_fee'] * (1 - df['covid_discount'])
    
    print(f"COVID discount period: {df[df['covid_discount'] > 0].index.min()} ~ {df[df['covid_discount'] > 0].index.max()}")
    print(f"Months with discount: {len(df[df['covid_discount'] > 0])}")
    
    # Handle missing values
    df = df.bfill().ffill()
    
    print("\n[7] Final data structure:")
    print(df.head(10))
    print(f"\nData period: {df.index.min()} ~ {df.index.max()}")
    print(f"Total {len(df)} months")
    print("\nColumns:", df.columns.tolist())
    
    # 7. Save to CSV
    output_file = '../data/anfis_dataset_with_covid.csv'
    df.to_csv(output_file)
    print(f"\n[OK] Integrated data saved: {output_file}")
    
    return df

if __name__ == "__main__":
    df = load_and_integrate()

