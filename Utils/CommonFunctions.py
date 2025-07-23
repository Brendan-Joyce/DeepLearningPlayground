
def NormalScaling(series):
    return (series - series.mean())/series.std()
    
def MinMaxScaling(series):
    return (series - series.min())/(series.max() - series.min())