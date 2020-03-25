
import os
import pandas as pd


# Create DataFrame for Data

def create_df():

    # Data Directory
    dir_list = os.listdir('dataset_RAVDESS_TESS')
    dir_list.sort()

    data_df = pd.DataFrame(
        columns=['path', 'source', 'actor', 'gender', 'emotion'])

    count = 0

    # creo la struttura data_df la quale conterrà un entry per ogni file wav
    for i in dir_list:
        if i != '.DS_Store':
            file_list = os.listdir('dataset_RAVDESS_TESS/' + i)
            for f in file_list:
                nm = f.split('.')[0].split('-')
                path = 'dataset_RAVDESS_TESS/' + i + '/' + f
                src = int(nm[1])
                actor = int(nm[-1])  # ultimo valore del vettore
                emotion = int(nm[2])

                if int(actor) % 2 == 0:
                    gender = "female"
                else:
                    gender = "male"

                if nm[3] == '01':
                    intensity = 0
                else:
                    intensity = 1

                if nm[4] == '01':
                    statement = 0
                else:
                    statement = 1

                if nm[5] == '01':
                    repeat = 0
                else:
                    repeat = 1

                data_df.loc[count] = [path, src, actor, gender, emotion]
                count += 1

    return data_df


############# labeling ###############

# 2 class: Positive & Negative
# Positive: Calm, Happy, Neutral, Surprised
# Negative: Angry, Fearful, Sad, Disgust

def data_labeling(data_df, gender):

    label2_list = []

    for i in range(len(data_df)):
        if data_df.emotion[i] == 1:  # Neutral
            lb = "_positive"
        elif data_df.emotion[i] == 2:  # Calm
            lb = "_positive"
        elif data_df.emotion[i] == 3:  # Happy
            lb = "_positive"
        elif data_df.emotion[i] == 4:  # Sad
            lb = "_negative"
        elif data_df.emotion[i] == 5:  # Angry
            lb = "_negative"
        elif data_df.emotion[i] == 6:  # Fearful
            lb = "_negative"
        elif data_df.emotion[i] == 7:  # Disgust
            lb = "_negative"
        elif data_df.emotion[i] == 8:  # Surprised
            lb = "_positive"
        else:
            lb = "_none"

        # Add gender to the label
        if gender == 'all':
            label2_list.append(data_df.gender[i])
        elif gender == 'male' or gender == 'female':
            label2_list.append(data_df.gender[i] + lb)


    data_df['label'] = label2_list
    return data_df

############ end labeling #############


# Female Data Set

def filter_male_dataset(data_df):
    female_data_df = data_df.copy()
    female_data_df = female_data_df[female_data_df.label != "male_none"]
    female_data_df = female_data_df[female_data_df.label != "female_none"]
    female_data_df = female_data_df[female_data_df.label != "male_happy"]
    female_data_df = female_data_df[female_data_df.label != "male_angry"]
    female_data_df = female_data_df[female_data_df.label != "male_sad"]
    female_data_df = female_data_df[female_data_df.label != "male_fearful"]
    female_data_df = female_data_df[female_data_df.label != "male_calm"]
    female_data_df = female_data_df[female_data_df.label != "male_positive"]
    female_data_df = female_data_df[female_data_df.label !=
                                    "male_negative"].reset_index(drop=True)

    tmp1 = female_data_df[female_data_df.actor == 22]
    tmp2 = female_data_df[female_data_df.actor == 24]
    female_test_df = pd.concat(
        [tmp1, tmp2], ignore_index=True).reset_index(drop=True)
    female_data_df = female_data_df[female_data_df.actor != 22]
    female_data_df = female_data_df[female_data_df.actor != 24].reset_index(
        drop=True)
    female_train_df = female_data_df

    return female_train_df, female_test_df


# Male Data Set
def filter_female_dataset(data_df):

    male_data_df = data_df.copy()
    male_data_df = male_data_df[male_data_df.label != "male_none"]
    male_data_df = male_data_df[male_data_df.label !=
                                "female_none"].reset_index(drop=True)
    male_data_df = male_data_df[male_data_df.label != "female_neutral"]
    male_data_df = male_data_df[male_data_df.label != "female_happy"]
    male_data_df = male_data_df[male_data_df.label != "female_angry"]
    male_data_df = male_data_df[male_data_df.label != "female_sad"]
    male_data_df = male_data_df[male_data_df.label != "female_fearful"]
    male_data_df = male_data_df[male_data_df.label != "female_calm"]
    male_data_df = male_data_df[male_data_df.label != "female_positive"]
    male_data_df = male_data_df[male_data_df.label !=
                                "female_negative"].reset_index(drop=True)

    tmp1 = male_data_df[male_data_df.actor == 21]
    tmp2 = male_data_df[male_data_df.actor == 22]
    tmp3 = male_data_df[male_data_df.actor == 23]
    tmp4 = male_data_df[male_data_df.actor == 24]
    male_test_df = pd.concat(
        [tmp1, tmp3], ignore_index=True).reset_index(drop=True)
    male_data_df = male_data_df[male_data_df.actor != 21]
    male_data_df = male_data_df[male_data_df.actor != 22]
    male_data_df = male_data_df[male_data_df.actor !=
                                23].reset_index(drop=True)
    male_data_df = male_data_df[male_data_df.actor !=
                                24].reset_index(drop=True)
    male_train_df = male_data_df

    return male_train_df, male_test_df


# Dataset per il classificatore del genere sessuale

def filter_dataset_for_gender_recognition(data_df):

    data_df_gender_recognition = data_df.copy()
    tmp1 = data_df_gender_recognition[data_df_gender_recognition.actor == 21]
    tmp2 = data_df_gender_recognition[data_df_gender_recognition.actor == 22]
    tmp3 = data_df_gender_recognition[data_df_gender_recognition.actor == 23]
    tmp4 = data_df_gender_recognition[data_df_gender_recognition.actor == 24]
    data_df_gender_recognition = data_df_gender_recognition[data_df_gender_recognition.actor != 21]
    data_df_gender_recognition = data_df_gender_recognition[data_df_gender_recognition.actor != 22]
    data_df_gender_recognition = data_df_gender_recognition[data_df_gender_recognition.actor != 23]
    data_df_gender_recognition = data_df_gender_recognition[data_df_gender_recognition.actor != 24].reset_index(
        drop=True)

    gender_recognition_test_df = pd.concat(
        [tmp1, tmp2, tmp3, tmp4], ignore_index=True).reset_index(drop=True)

    gender_recognition_train_df = data_df_gender_recognition

    return gender_recognition_train_df, gender_recognition_test_df


def get_datatrain_and_datatest(gender):

    data_df = create_df()
    data_df = data_labeling(data_df, gender)


    if gender == 'male':
        temp = filter_female_dataset(data_df)
        train_data = temp[0]
        test_data = temp[1]
    elif gender == 'female':
        temp = filter_male_dataset(data_df)
        train_data = temp[0]
        test_data = temp[1]
    elif gender == 'all':
        temp = filter_dataset_for_gender_recognition(data_df)
        train_data = temp[0]
        test_data = temp[1]
    else:
        pass

    return train_data, test_data
