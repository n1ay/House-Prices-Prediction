import numpy as np
import pandas as pd
from encode import build_feature_encoders, encode_df
from utils import scale_matrices, show_correlation_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def encode_remove_NA_train(df, encoders):

    y_train = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    X_train = encode_df(df, encoders)

    # remove NA from numeric values
    # remove NA from MasVnrArea column by assuming NA=0
    X_train['MasVnrArea'].fillna(0, inplace=True)

    # drop LotFrontage, since it has too many NAs
    X_train.drop(labels=['LotFrontage'], axis=1, inplace=True)

    # replace NA by value of normal distribution from YearBuild
    X_train.loc[X_train.GarageYrBlt.isnull(), 'GarageYrBlt'] = X_train.loc[X_train.GarageYrBlt.isnull(),'YearBuilt']

    return X_train, y_train

def encode_remove_NA_test(df, encoders):

    X_test = encode_df(df, encoders)

    # remove NA from numeric values
    # remove NA from MasVnrArea column by assuming NA=0
    X_test['MasVnrArea'].fillna(0, inplace=True)

    # drop LotFrontage, since it has too many NAs
    X_test.drop(labels=['LotFrontage'], axis=1, inplace=True)

    # replace NA by random value of normal distribution from GarageYrBlt
    X_test.loc[X_test.GarageYrBlt.isnull(), 'GarageYrBlt'] = X_test.loc[X_test.GarageYrBlt.isnull(), 'YearBuilt']

    # replace NA by random value of normal distribution from GarageArea
    X_test['GarageArea'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from GarageCars
    X_test['GarageCars'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from BsmtHalfBath
    X_test['BsmtHalfBath'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from BsmtFullBath
    X_test['BsmtFullBath'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from TotalBsmtSF
    X_test['TotalBsmtSF'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from BsmtUnfSF
    X_test['BsmtUnfSF'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from BsmtFinSF1
    X_test['BsmtFinSF1'].fillna(0, inplace=True)

    # replace NA by random value of normal distribution from BsmtFinSF2
    X_test['BsmtFinSF2'].fillna(0, inplace=True)

    return X_test


def main():
    encoders = build_feature_encoders()
	
    X_train, y_train = encode_remove_NA_train(pd.read_csv('data/train.csv'), encoders)
    X_test = encode_remove_NA_test(pd.read_csv('data/test.csv'), encoders)

    X_train, X_test = scale_matrices(X_train, X_test)
    y_train = y_train.apply(lambda x: np.log(x))

    #show_correlation_matrix(X_train, y_train)

    grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), cv=5, param_grid={
        'max_depth': [6, 8, 10, 12],
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
        print(np.exp(i))

if __name__ == "__main__":
    main()
