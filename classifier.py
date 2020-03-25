
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
# https://xgboost.readthedocs.io/en/latest/parameter.html


# CARICAMENTO FEATURES

def load_features(gender):

    try:
        if gender == 'male':
            X_train = pd.read_json(r'features/X_train_male_features.json')
            y_train = pd.read_json(r'features/y_train_male.json')
            X_test = pd.read_json(r'features/X_test_male_features.json')
            y_test = pd.read_json(r'features/y_test_male.json')
        elif gender == 'female':
            X_train = pd.read_json(r'features/X_train_female_features.json')
            y_train = pd.read_json(r'features/y_train_female.json')
            X_test = pd.read_json(r'features/X_test_female_features.json')
            y_test = pd.read_json(r'features/y_test_female.json')
        elif gender == 'all':
            X_train = pd.read_json(r'features/X_train_all_features.json')
            y_train = pd.read_json(r'features/y_train_all.json')
            X_test = pd.read_json(r'features/X_test_all_features.json')
            y_test = pd.read_json(r'features/y_test_all.json')


        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_train, y_train, X_test, y_test

    except:
        raise Exception("Occorre prima calcolare le features")


def train(X_train, y_train, X_test, y_test):
    y_train = np.array(y_train).flatten()
    y_train = np.round(y_train).astype(int)
    y_train = y_train[0::2]

    model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                          gamma=0, learning_rate=0.2, max_delta_step=0.1, max_depth=6,
                          booster='gbtree', min_child_weight=1, missing=None, n_estimators=1000,
                          objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                          scale_pos_weight=1, seed=0, verbosity=1, subsample=1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test = np.array(y_test).flatten().astype(int)
    y_test = y_test[0::2]

    print(classification_report(y_test, y_pred))
    return model
