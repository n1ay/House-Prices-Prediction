import unittest
from encode import build_feature_encoders
import pandas as pd
from main import encode_remove_NA_train, encode_remove_NA_test
from utils import scale_matrices

class PreprocessTest(unittest.TestCase):
    def test_NA_train_df(self):
        encoders = build_feature_encoders()
        X_train, y_train = encode_remove_NA_train(pd.read_csv('data/train.csv'), encoders)
        NA_rows = X_train[X_train.isnull().any(axis=1)]+y_train[y_train.isnull()]
        self.assertEqual(len(NA_rows), 0)

    def test_NA_test_df(self):
        encoders = build_feature_encoders()
        X_test = encode_remove_NA_test(pd.read_csv('data/test.csv'), encoders)
        NA_rows = X_test[X_test.isnull().any(axis=1)]
        self.assertEqual(len(NA_rows), 0)

    def test_length_train_df(self):
        encoders = build_feature_encoders()
        X_train, y_train = encode_remove_NA_train(pd.read_csv('data/train.csv'), encoders)
        self.assertEqual(len(X_train), len(y_train))

    def test_length_after_normalization(self):
        encoders = build_feature_encoders()
        X_train, y_train = encode_remove_NA_train(pd.read_csv('data/train.csv'), encoders)
        X_test = encode_remove_NA_test(pd.read_csv('data/test.csv'), encoders)
        X_train_after, X_test_after = scale_matrices(X_train, X_test)
        self.assertEqual(len(X_train), len(X_train_after))
        self.assertEqual(len(X_test), len(X_test_after))
