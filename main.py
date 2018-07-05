import numpy as np
import pandas as pd
from encode import build_feature_encoders
from random import gauss
from utils import replace_NA_rand_gauss

def main():
    df = pd.read_csv('data/train.csv')

    #encode non-numeric values
    encoders = build_feature_encoders()
    encoded_df = pd.DataFrame()
    for i in df:
        if i in encoders:
            encoded_df[i] = encoders[i].transform(df[i].tolist())
        else:
            encoded_df[i] = df[i]


    #remove NA from numeric values
    #remove NA from MasVnrArea column by assuming NA=0
    encoded_df['MasVnrArea'].replace(np.NaN, 0, inplace=True)

    #replace NA by random value of normal distribution from LotFrontage
    encoded_df['LotFrontage'] = replace_NA_rand_gauss(encoded_df['LotFrontage'])

    #replace NA by random value of normal distribution from GarageYrBlt
    encoded_df['GarageYrBlt'] = replace_NA_rand_gauss(encoded_df['GarageYrBlt'])


if __name__ == "__main__":
    main()
