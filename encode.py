from sklearn.preprocessing import LabelEncoder
import pandas as pd

#using LabelEncoder from sklearn
def build_feature_encoders():
    return {
        'MSZoning': LabelEncoder().fit(['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM', 'nan']),
        'Street': LabelEncoder().fit(['Grvl', 'Pave']),
        'Alley': LabelEncoder().fit(['Grvl', 'Pave', 'nan']),
        'LotShape': LabelEncoder().fit(['Reg', 'IR1', 'IR2', 'IR3']),
        'LandContour': LabelEncoder().fit(['Lvl', 'Bnk', 'HLS', 'Low']),
        'Utilities': LabelEncoder().fit(['AllPub', 'NoSewr', 'NoSeWa', 'ELO', 'nan']),
        'LotConfig': LabelEncoder().fit(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']),
        'LandSlope': LabelEncoder().fit(['Gtl', 'Mod', 'Sev']),
        'Neighborhood': LabelEncoder().fit(['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                                            'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                                            'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                                            'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']),
        'Condition1': LabelEncoder().fit(['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']),
        'Condition2': LabelEncoder().fit(['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']),
        'BldgType': LabelEncoder().fit(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs']),
        'HouseStyle': LabelEncoder().fit(['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']),
        'RoofStyle': LabelEncoder().fit(['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed']),
        'RoofMatl': LabelEncoder().fit(['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl']),
        'Exterior1st': LabelEncoder().fit(['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard',
                                           'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                                           'VinylSd', 'Wd Sdng', 'WdShing', 'nan']),
        'Exterior2nd': LabelEncoder().fit(['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'HdBoard',
                                           'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco',
                                           'VinylSd', 'Wd Sdng', 'Wd Shng', 'nan']),
        'MasVnrType': LabelEncoder().fit(['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone', 'nan']),
        'ExterQual': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'ExterCond': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'Foundation': LabelEncoder().fit(['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood']),
        'BsmtQual': LabelEncoder(). fit(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'nan']),
        'BsmtCond': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'nan']),
        'BsmtExposure': LabelEncoder().fit(['Gd', 'Av', 'Mn', 'No', 'nan']),
        'BsmtFinType1': LabelEncoder().fit(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'nan']),
        'BsmtFinType2': LabelEncoder().fit(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'nan']),
        'Heating': LabelEncoder().fit(['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall']),
        'HeatingQC': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'CentralAir': LabelEncoder().fit(['N', 'Y']),
        'Electrical': LabelEncoder().fit(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', 'nan']),
        'KitchenQual': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'nan']),
        'Functional': LabelEncoder().fit(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal', 'nan']),
        'FireplaceQu': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'nan']),
        'GarageType': LabelEncoder().fit(['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'nan']),
        'GarageFinish': LabelEncoder().fit(['Fin', 'RFn', 'Unf', 'nan']),
        'GarageQual': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'nan']),
        'GarageCond': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'nan']),
        'PavedDrive': LabelEncoder().fit(['Y', 'P', 'N']),
        'PoolQC': LabelEncoder().fit(['Ex', 'Gd', 'TA', 'Fa', 'nan']),
        'Fence': LabelEncoder().fit(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'nan']),
        'MiscFeature': LabelEncoder().fit(['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'nan']),
        'SaleType': LabelEncoder().fit(['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth', 'nan']),
        'SaleCondition': LabelEncoder().fit(['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'])
    }

def encode_df(df, encoders):
    encoded_df = pd.DataFrame()
    for i in df:
        if i in encoders:
            encoded_df[i] = encoders[i].transform(df[i].tolist())
        else:
            encoded_df[i] = df[i]

    return encoded_df