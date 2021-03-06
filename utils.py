from random import gauss, uniform
from pandas import isnull, DataFrame
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def replace_NA_rand_gauss(column):
    mean_val = column.mean(skipna=True)
    std_val = column.std(skipna=True)
    return column.apply(lambda x: gauss(mean_val, std_val) if isnull(x) else x)
    
def replace_NA_rand_uniform(column):
	min_val = column.min(skipna=True)
	max_val = column.max(skipna=True)
	return column.apply(lambda x: uniform(min_val, max_val) if isnull(x) else x)

def scale_matrices(df_train, df_test):
    scaler = MinMaxScaler(copy=False)
    df_merged = df_train.append(df_test, ignore_index=True)

    scaler.fit(df_merged)
    scaled_matrix = DataFrame(scaler.transform(df_merged), columns=df_train.columns)
    df_train = scaled_matrix.iloc[0:len(df_train), :]
    df_test = scaled_matrix.iloc[len(df_train):, :]

    return df_train, df_test

def show_correlation_matrix(X, y):
    sns.set(style="white")

    df = X.copy()
    df['SalePrice']=y
    corr = df.corr()

    plt.figure(figsize=(10,8))
    ax = sns.heatmap(corr, vmax=1, square=True, annot=False ,fmt='.2f', cmap ='GnBu', cbar_kws={"shrink": .5}, robust=True, linewidths=.5)
    plt.title('Correlation matrix between the features', fontsize=20)
    plt.show()
