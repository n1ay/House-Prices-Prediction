import numpy as np
import pandas as pd
from encode import build_feature_encoders, encode_df
from utils import replace_NA_rand_gauss, replace_NA_rand_uniform, scale_matrices, show_correlation_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adadelta

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

	X_train = X_train.values.reshape((1460, 80))
	y_train = y_train.values.reshape((1460, 1))
	X_test = X_test.values.reshape((1459, 80))

	model = Sequential()
	model.add(Dense(300, activation='relu', input_dim=80))
	model.add(Dropout(0.25))
	model.add(Dense(700, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(1, activation='relu'))

	model.compile(loss='mean_squared_error', optimizer=Adadelta(), metrics=['accuracy'])
	model.fit(x=X_train, y=y_train, batch_size=1, shuffle=True, epochs=15, verbose=2)
	prediction = model.predict(x=X_test)

	for i in prediction:
		print(i)

if __name__ == "__main__":
    main()
