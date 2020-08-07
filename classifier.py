
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
# https://xgboost.readthedocs.io/en/latest/parameter.html


def multiple_train(X_train, y_train, X_test, y_test):

    temp = []
    for i in y_train:
        index = np.argmax(i)
        temp.append(index)
    y_train = np.round(temp).astype(int)

    model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                          gamma=0, learning_rate=0.2, max_delta_step=0.1, max_depth=6,
                          booster='gbtree', min_child_weight=1, missing=None, n_estimators=1000,
                          objective='multi:softprob', num_class=8, reg_alpha=0, reg_lambda=1,
                          scale_pos_weight=1, seed=0, verbosity=1, subsample=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    temp = []
    for i in y_test:
        index = np.argmax(i)
        temp.append(index)
    y_test = np.round(temp).astype(int)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model


def binary_train(X_train, y_train, X_test, y_test):

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
    print(confusion_matrix(y_test, y_pred))

    return model
