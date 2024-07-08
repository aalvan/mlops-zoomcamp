import pandas as pd
from deepdiff import DeepDiff
from batch import prepare_data, dt


def test_prepare_data():

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    categorical = ['PULocationID', 'DOLocationID']
    
    actual_df = prepare_data(df, categorical)
    
    expected_df = [
        {'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': dt(1, 1), 'tpep_dropoff_datetime':  dt(1, 10), 'duration': 9.0},
        {'PULocationID': '1', 'DOLocationID': '1', 'tpep_pickup_datetime': dt(1, 2), 'tpep_dropoff_datetime':  dt(1, 10), 'duration': 8.0}
    ]
    expected_df = pd.DataFrame(expected_df)

    diff = DeepDiff(actual_df.to_dict, expected_df.to_dict)
    print(diff)

    assert 'type_changes' not in diff
