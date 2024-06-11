if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

from sklearn.linear_model import LinearRegression

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    X_train = data['build'][0]
    y_train = data['build'][1]
    dv = data['build'][2]
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return lr.intercept_, dv