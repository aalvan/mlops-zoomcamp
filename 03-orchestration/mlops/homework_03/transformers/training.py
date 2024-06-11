if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.linear_model import LinearRegression
@transformer
def training_sklearn(X_train, y_train, dv, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    model = LinearRegression()
    print("Training model")
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    rmse = mean_squared_error(y_train, predictions, squared=False)
    print(f"RMSE: {rmse}")
    print(f"intercep{model.intercept_}")

    return dv, model