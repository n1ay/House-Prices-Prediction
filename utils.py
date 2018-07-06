from random import gauss
from pandas import isnull, DataFrame
from sklearn.preprocessing import MinMaxScaler

def replace_NA_rand_gauss(column):
    mean_val = column.mean(skipna=True)
    std_val = column.std(skipna=True)
    return column.apply(lambda x: gauss(mean_val, std_val) if isnull(x) else x)

def scale_matrices(df_train, df_test):
    scaler = MinMaxScaler(copy=False)
    df_merged = df_train.append(df_test, ignore_index=True)

    scaler.fit(df_merged)
    scaled_matrix = DataFrame(scaler.transform(df_merged), columns=df_train.columns)
    df_train = scaled_matrix.iloc[0:len(df_train), :]
    df_test = scaled_matrix.iloc[len(df_train):, :]

    return df_train, df_test
