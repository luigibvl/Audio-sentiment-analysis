
import numpy as np
import pandas as pd
import librosa
import random
import os
from sklearn.model_selection import StratifiedShuffleSplit  # , StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tqdm import tqdm


# OTTENIMENTO FEATURE
# tqdm -> serve solo a dare un effetto grafico sull'avanzamento della percentuale
# paramentri di librosa.load()
# res_type -> tipo di campionamento, la migliore dovrebbe essere la kaiser_best
# duration -> legge i byte corrispondenti a input_duration secondi del file
# sr -> frequenza di campionamento (in questo caso li ricampiono a 44100 kz)
# offset -> legge il file a partire da questo offset (in secondi)

input_duration = 3


def get_features(data2_df, data3_df, gender):
    data = pd.DataFrame(columns=['feature'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(
            data2_df.path[i], res_type='kaiser_best', duration=input_duration, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)
        # mfccs -> http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        data.loc[i] = [feature]

    features_labeled = concat_label_features(data2_df, data)
    syn_data1 = augmentation_1(data2_df)
    syn_data2 = augmentation_2(data2_df)
    combined_df = concat_all_data(syn_data1, syn_data2, features_labeled)
    X_train, y_train, X_test, y_test = data_split(combined_df)
    save_features(X_train, y_train, X_test, y_test, gender)


# return nuovo dataframe ottenuto concatenando le label con le feature
def concat_label_features(data2_df, data):
    # df3 -> lista delle feature
    df3 = pd.DataFrame(data['feature'].values.tolist())
    labels = data2_df.label
    newdf = pd.concat([df3, labels], axis=1)
    rnewdf = newdf.rename(index=str, columns={"0": "label"})
    # sostituisco i valori NaN con 0
    rnewdf.isnull().sum().sum()
    rnewdf = rnewdf.fillna(0)
    return rnewdf


# Adding White Noise
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
def noise(data):

    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data = data.astype('float64') + noise_amp * \
        np.random.normal(size=data.shape[0])
    return data


# Pitch Tuning
def pitch(data, sample_rate):

    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                       sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data


# Augmentation Method 1
def augmentation_1(data2_df):
    syn_data1 = pd.DataFrame(columns=['feature', 'label'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(
            data2_df.path[i], res_type='kaiser_best', duration=input_duration, sr=22050 * 2, offset=0.5)
        if data2_df.label[i]:
            X = noise(X)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            a = random.uniform(0, 1)
            syn_data1.loc[i] = [feature, data2_df.label[i]]

    syn_data1 = syn_data1.reset_index(drop=True)
    return syn_data1


# Augmentation Method 2
def augmentation_2(data2_df):
    syn_data2 = pd.DataFrame(columns=['feature', 'label'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(
            data2_df.path[i], res_type='kaiser_best', duration=input_duration, sr=22050 * 2, offset=0.5)
        if data2_df.label[i]:
            X = pitch(X, sample_rate)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            a = random.uniform(0, 1)
            syn_data2.loc[i] = [feature, data2_df.label[i]]

    syn_data2 = syn_data2.reset_index(drop=True)
    return syn_data2


# concatenazione dei dati iniziali coi dati aumentati
def concat_all_data(syn_data1, syn_data2, rnewdf):
    df4 = pd.DataFrame(syn_data1['feature'].values.tolist())
    labels4 = syn_data1.label
    syndf1 = pd.concat([df4, labels4], axis=1)
    syndf1 = syndf1.rename(index=str, columns={"0": "label"})
    syndf1 = syndf1.fillna(0)

    df4 = pd.DataFrame(syn_data2['feature'].values.tolist())
    labels4 = syn_data2.label
    syndf2 = pd.concat([df4, labels4], axis=1)
    syndf2 = syndf2.rename(index=str, columns={"0": "label"})
    syndf2 = syndf2.fillna(0)

    # Combining the Augmented data with original
    # combined_df avrà come dimensione la dimensione del train set * 3
    combined_df = pd.concat([rnewdf, syndf1, syndf2], ignore_index=True)
    combined_df = combined_df.fillna(0)
    return combined_df


# Stratified Shuffle Split
# combined_df = rnewdf

# Split dei dati per il test set e train set
def data_split(combined_df):
    X = combined_df.drop(['label'], axis=1)
    y = combined_df.label

    xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
    #xxx = StratifiedKFold(5, shuffle=True, random_state=12)

    for train_index, test_index in xxx.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train.isna().sum().sum()

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    return X_train, y_train, X_test, y_test


def save_features(X_train, y_train, X_test, y_test, gender):

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    '''
    print('X_train')
    print(X_train)
    print('y_train')
    print(y_train)
    print('X_test')
    print(X_test)
    print('y_test')
    print(y_test)
    '''

    ########### SALAVATAGGIO FEATURES ############

    df_X_train = pd.DataFrame(X_train)
    df_y_train = pd.DataFrame(y_train)
    df_X_test = pd.DataFrame(X_test)
    df_y_test = pd.DataFrame(y_test)

    if os.path.exists('features') == False:
        os.mkdir('features')

    if gender == 'male':
        df_X_train.to_json(r'features/X_train_male_features.json')
        df_y_train.to_json(r'features/y_train_male.json')
        df_X_test.to_json(r'features/X_test_male_features.json')
        df_y_test.to_json(r'features/y_test_male.json')
    elif gender == 'female':
        df_X_train.to_json(r'features/X_train_female_features.json')
        df_y_train.to_json(r'features/y_train_female.json')
        df_X_test.to_json(r'features/X_test_female_features.json')
        df_y_test.to_json(r'features/y_test_female.json')
    elif gender == 'all':
        df_X_train.to_json(r'features/X_train_all_features.json')
        df_y_train.to_json(r'features/y_train_all.json')
        df_X_test.to_json(r'features/X_test_all_features.json')
        df_y_test.to_json(r'features/y_test_all.json')
    else:
        ### si dovrebbe gestire l'errore
        pass
