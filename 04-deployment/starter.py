#!/usr/bin/env python
# coding: utf-8
import pickle
import argparse

import pandas as pd
import numpy as np



def read_data(filename, year, month):
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def prepare_dictionaries(df):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def get_std(predictions):
    std_dev = np.std(predictions)
    print(f"Standard Deviation of the predicted duration: {std_dev}")

def get_std(predictions):
    mean = np.mean(predictions)
    print(f"Mean of the predicted duration: {mean}")

def apply_model(model_path, input_file, output_file, year, month):
    print(f'reading the data from {input_file} ...')
    df = read_data(input_file, year, month)
    dicts = prepare_dictionaries(df)

    print(f'loading the model from {model_path} ...')
    dv, model =  load_model(model_path)
    X_val = dv.transform(dicts)
    
    print(f'applying the model...')
    y_pred = model.predict(X_val)
    
    get_std(y_pred)

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    print(f'saving the result to {output_file} ...')
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )

def run():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--year", default=2023, type=int, help="Year of the file")
    ap.add_argument("-m", "--month", default=3, type=int, help="Month of the file")
    args = ap.parse_args()

    year = args.year
    month = args.month
    taxi_type = 'yellow'
    model_path = 'model.bin'
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    
    apply_model(model_path=model_path, 
                input_file=input_file, 
                output_file=output_file, 
                year=year, month=month)
if __name__ == '__main__':
    run()


