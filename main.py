
import sys
from reader import get_datatrain_and_datatest
from features_maker import get_features
from classifier import train, load_features


# AUDIO SENTIMENT ANALYSIS
# L'obiettivo è capire se dato un file.wav, esso esprime
# un'emozione negativa o positiva. Dunque prima bisogna classificare il genere sessuale
# dell'attore che sta parlando e dopo classificarne l'emozione associata alla
# sua voce. La variabile dataset andrà a modificare il comportamento
# del classificatore (associando 'all' a tale variabile andremo a classificare
# il genere sessuale, mentre associandogli 'male' o 'female' andremo a classificare
# l'emozione)

dataset = None

############## DA COMMENTARE UNA VOLTA OTTENUTE LE FEATURES

# prima si ottengono le features per il task di interesse
# le features saranno salvate nella cartella features, quindi non sarà più necesario ricalcolarle
# uncomment on what you want to use
#dataset = 'female'
#dataset = 'male'
#dataset = 'all'  # for gender recognition

##############

if dataset == 'female' or dataset == 'male' or dataset == 'all':
    train_df, test_df = get_datatrain_and_datatest(dataset)
    get_features(train_df, test_df, dataset)

###############


# L'idea è di farmi ritornare il risultato della classificazione
# in base al genere sessuale e in base quest'ultimo lanciare un nuovo classificatore,
# quindi prima costruisco un classificatore per classificare il genere sessuale
# e poi lancio il classificare emozionale in base al genere sessuale

# uncomment on what you want to use
# il classificatore utilizzerà questo dataset (se non sono state calcolate
# le features per il task d'interesse sarà sollevata un'eccezione)
dataset = 'female'
#dataset = 'male'
#dataset = 'all'  # for gender recognition

if dataset == None:
    print('Modifica il valore della variabile dataset')
    sys.exit(0)

# caricamento features e classificazione
X_train, y_train, X_test, y_test = load_features(dataset)
print("Dimensione dei dati di training: ", end = '')
print(len(X_train))
print("Dimensione dei dati di test: ", end = '')
print(len(X_test))
model = train(X_train, y_train, X_test, y_test)


'''
# istanza è un'ipotetica istanza audio che vorremmo classificare
# prima classifico il genere, dunque y sarà maschio o femmina
y = model.predict(istanza)
if y == 0:  # ipotizzando che 0 sia maschio
    X_train, y_train, X_test, y_test = load_features('male')
    new_model = train(X_train, y_train, X_test, y_test)
elif y == 1:
    X_train, y_train, X_test, y_test = load_features('female')
    new_model = train(X_train, y_train, X_test, y_test)
else:
    pass

try:
    y = new_model.predict(istanza)
    print(y)
except:
    print('Errore nella costruzione del modello')
'''
