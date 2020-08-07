
import sys
from reader import get_datatrain_and_datatest
from features_maker import get_features, pitch, noise, load_features, concat_all_data
from classifier import binary_train, multiple_train
import pickle
import librosa
import numpy as np
import pandas as pd


# AUDIO SENTIMENT ANALYSIS
# L'obiettivo è capire se dato un file.wav, esso esprime
# un'emozione negativa o positiva. Dunque prima bisogna classificare il genere sessuale
# dell'attore che sta parlando e dopo classificarne l'emozione associata alla
# sua voce.


# prima si ottengono le features e i modelli per il task di interesse
# le features saranno salvate nella cartella features,
# mentre i modelli nella cartella models

############## SETTARE A FALSE LA VARIABILE not_already_computed
############## UNA VOLTA OTTENUTE LE FEATURES E I MODELLI

not_already_computed = True
#not_already_computed = False

all_gender = ['female','male','all']

if not_already_computed:
    for gender in all_gender:
        data_df = get_datatrain_and_datatest(gender)
        get_features(data_df, gender)
        X_train, y_train, X_test, y_test = load_features(gender)


        #########*****************CLASSIFICAZIONE*****************###########
        #####           Nota bene: occorre stabilire in reader.py       #####
        ##### il labeling dei dati consistente col classifcatore scelto #####
        #####################################################################


        # per avere il multiclassificatore
        # commentare la parte relativa al classificatore binario per le emozioni
        if gender == 'all':
            model = binary_train(X_train,y_train, X_test, y_test)
        else:
            model = multiple_train(X_train,y_train, X_test, y_test)


        # per avere il classificatore binario
        # commentare la parte relativa al classificatore multiplo per le emozioni
        # model = binary_train(X_train,y_train, X_test, y_test)

        pickle.dump(model, open("models/"+gender+".pickle.dat", "wb"))



try:
    input = 'input/input.wav'
    input_duration = 3
    X, sample_rate = librosa.load(input, res_type='kaiser_best', duration=input_duration, sr=22050 * 2, offset=0.5)
    sample_rate = np.array(sample_rate)
    features = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    features = pd.DataFrame(features)
    features = features.fillna(0)

    # Data alignment
    num_rows = features.shape[0]
    max_row=259
    if num_rows<max_row:
        for i in range(num_rows,max_row):
            features.loc[i]=0

    features = np.reshape(features,(features.shape[0],1))
    if num_rows>max_row:
        features = features[:max_row]
    features = features.T.to_numpy()

    #### Load model and evaluate

    model = pickle.load(open("models/all.pickle.dat", "rb"))
    temp = model.predict(features)

    if temp[0]==0:
        ## is male
        model = pickle.load(open("models/male.pickle.dat", "rb"))
        result = model.predict(features)
    elif temp[0]==1:
        model = pickle.load(open("models/female.pickle.dat", "rb"))
        result = model.predict(features)
    print(result[0])
except:
    raise Exception("Uh-Oh! I dati non sono stati gestiti correttamente")
