from sklearn.metrics import mean_squared_error


def rmse(pred, test):
    return mean_squared_error(pred, test) ** 0.5
