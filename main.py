import numpy as np
import pandas as pd
from encode import build_feature_encoders, encode_df
from utils import replace_NA_rand_gauss, scale_matrices, show_correlation_matrix
from sklearn.ensemble import GradientBoostingRegressor

def encode_remove_NA_train(df, encoders):

    y_train = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    X_train = encode_df(df, encoders)

    # remove NA from numeric values
    # remove NA from MasVnrArea column by assuming NA=0
    X_train['MasVnrArea'].replace(np.NaN, 0, inplace=True)

    # replace NA by random value of normal distribution from LotFrontage
    X_train['LotFrontage'] = replace_NA_rand_gauss(X_train['LotFrontage'])

    # replace NA by random value of normal distribution from GarageYrBlt
    X_train['GarageYrBlt'] = replace_NA_rand_gauss(X_train['GarageYrBlt'])

    return X_train, y_train

def encode_remove_NA_test(df, encoders):

    X_test = encode_df(df, encoders)

    # remove NA from numeric values
    # remove NA from MasVnrArea column by assuming NA=0
    X_test['MasVnrArea'].replace(np.NaN, 0, inplace=True)

    # replace NA by random value of normal distribution from LotFrontage
    X_test['LotFrontage'] = replace_NA_rand_gauss(X_test['LotFrontage'])

    # replace NA by random value of normal distribution from GarageYrBlt
    X_test['GarageYrBlt'] = replace_NA_rand_gauss(X_test['GarageYrBlt'])

    # replace NA by random value of normal distribution from GarageArea
    X_test['GarageArea'] = replace_NA_rand_gauss(X_test['GarageArea'])

    # replace NA by random value of normal distribution from GarageCars
    X_test['GarageCars'] = replace_NA_rand_gauss(X_test['GarageCars'])

    # replace NA by random value of normal distribution from BsmtHalfBath
    X_test['BsmtHalfBath'] = replace_NA_rand_gauss(X_test['BsmtHalfBath'])

    # replace NA by random value of normal distribution from BsmtFullBath
    X_test['BsmtFullBath'] = replace_NA_rand_gauss(X_test['BsmtFullBath'])

    # replace NA by random value of normal distribution from TotalBsmtSF
    X_test['TotalBsmtSF'] = replace_NA_rand_gauss(X_test['TotalBsmtSF'])

    # replace NA by random value of normal distribution from BsmtUnfSF
    X_test['BsmtUnfSF'] = replace_NA_rand_gauss(X_test['BsmtUnfSF'])

    # replace NA by random value of normal distribution from BsmtFinSF1
    X_test['BsmtFinSF1'] = replace_NA_rand_gauss(X_test['BsmtFinSF1'])

    # replace NA by random value of normal distribution from BsmtFinSF2
    X_test['BsmtFinSF2'] = replace_NA_rand_gauss(X_test['BsmtFinSF2'])

    return X_test


def main():
    encoders = build_feature_encoders()

    X_train, y_train = encode_remove_NA_train(pd.read_csv('data/train.csv'), encoders)
    X_test = encode_remove_NA_test(pd.read_csv('data/test.csv'), encoders)

    X_train, X_test = scale_matrices(X_train, X_test)

    #show_correlation_matrix(X_train, y_train)

    ensemble = GradientBoostingRegressor(max_depth=16, learning_rate=.08, min_samples_leaf=9,
                                min_samples_split=9, n_estimators=300)

    ensemble.fit(X_train, y_train)
    prediction = ensemble.predict(X_test)
    print("Score: ", ensemble.score(X_train, y_train))

    for i in prediction:
        print(i)


if __name__ == "__main__":
    main()
