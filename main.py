import numpy as np
import pandas as pd
from encode import build_feature_encoders, encode_df
from utils import replace_NA_rand_gauss, replace_NA_rand_uniform, scale_matrices, show_correlation_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

def encode_remove_NA_train(df, encoders, method):

    y_train = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    X_train = encode_df(df, encoders)

    # remove NA from numeric values
    # remove NA from MasVnrArea column by assuming NA=0
    X_train['MasVnrArea'].replace(np.NaN, 0, inplace=True)

    # replace NA by random value of normal distribution from LotFrontage
    X_train['LotFrontage'] = method(X_train['LotFrontage'])

    # replace NA by random value of normal distribution from GarageYrBlt
    X_train['GarageYrBlt'] = method(X_train['GarageYrBlt'])

    return X_train, y_train

def encode_remove_NA_test(df, encoders, method):

    X_test = encode_df(df, encoders)

    # remove NA from numeric values
    # remove NA from MasVnrArea column by assuming NA=0
    X_test['MasVnrArea'].replace(np.NaN, 0, inplace=True)

    # replace NA by random value of normal distribution from LotFrontage
    X_test['LotFrontage'] = replace_NA_rand_uniform(X_test['LotFrontage'])

    # replace NA by random value of normal distribution from GarageYrBlt
    X_test['GarageYrBlt'] = replace_NA_rand_uniform(X_test['GarageYrBlt'])

    # replace NA by random value of normal distribution from GarageArea
    X_test['GarageArea'] = replace_NA_rand_uniform(X_test['GarageArea'])

    # replace NA by random value of normal distribution from GarageCars
    X_test['GarageCars'] = replace_NA_rand_uniform(X_test['GarageCars'])

    # replace NA by random value of normal distribution from BsmtHalfBath
    X_test['BsmtHalfBath'] = replace_NA_rand_uniform(X_test['BsmtHalfBath'])

    # replace NA by random value of normal distribution from BsmtFullBath
    X_test['BsmtFullBath'] = replace_NA_rand_uniform(X_test['BsmtFullBath'])

    # replace NA by random value of normal distribution from TotalBsmtSF
    X_test['TotalBsmtSF'] = replace_NA_rand_uniform(X_test['TotalBsmtSF'])

    # replace NA by random value of normal distribution from BsmtUnfSF
    X_test['BsmtUnfSF'] = replace_NA_rand_uniform(X_test['BsmtUnfSF'])

    # replace NA by random value of normal distribution from BsmtFinSF1
    X_test['BsmtFinSF1'] = replace_NA_rand_uniform(X_test['BsmtFinSF1'])

    # replace NA by random value of normal distribution from BsmtFinSF2
    X_test['BsmtFinSF2'] = replace_NA_rand_uniform(X_test['BsmtFinSF2'])

    return X_test


def main():
    encoders = build_feature_encoders()
    method = replace_NA_rand_uniform
	
    X_train, y_train = encode_remove_NA_train(pd.read_csv('data/train.csv'), encoders, method)
    X_test = encode_remove_NA_test(pd.read_csv('data/test.csv'), encoders, method)

    X_train, X_test = scale_matrices(X_train, X_test)    

    #show_correlation_matrix(X_train, y_train)

    grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), cv=5, param_grid={
        'max_depth': [8],
        'learning_rate': [0.1],
        'min_samples_leaf': [9],
        'min_samples_split': [9],
        'n_estimators': [400],
    }, n_jobs=4, pre_dispatch=8)

    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_test)
    print("Best estimator: ", grid_search.best_estimator_)
    print("Best estimator score:", grid_search.score(X_train, y_train))

    for i in prediction:
        print(i)

if __name__ == "__main__":
    main()
