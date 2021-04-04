import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random


def get_input_and_output(num_samples=50):
    # Read excel file
    column_names = ["Crop", "Season",
                    "Rainfall Min",
                    "Rainfall Max",
                    "Relative Humidity Min",
                    "Relative Humidity Max",
                    "Min Temp", "Max Temp",
                    "pH Min", "pH Max", ]

    df = pd.read_excel("input.xlsx", names=column_names)

    # Generate crop ids
    unique_crops = df.Crop.unique()
    crop_ids = {crop: i for i, crop in enumerate(unique_crops)}

    # Generate season ids
    unique_seasons = df.Season.unique()
    season_ids = {season: i for i, season in enumerate(unique_seasons)}

    # Assign ids in data
    # df.replace({'Crop': crop_ids, 'Season': season_ids}, inplace=True)

    data_values = []

    input_len = len(df)
    for i in range(input_len):
        crop = df.loc[i, "Crop"]
        season = df.loc[i, "Season"]
        rainfall_min = float(df.loc[i, "Rainfall Min"])
        rainfall_max = float(df.loc[i, "Rainfall Max"])
        relative_humidity_min = float(df.loc[i, "Relative Humidity Min"])
        relative_humidity_max = float(df.loc[i, "Relative Humidity Max"])
        min_temp = float(df.loc[i, "Min Temp"])
        max_temp = float(df.loc[i, "Max Temp"])
        p_h_min = float(df.loc[i, "pH Min"])
        p_h_max = float(df.loc[i, "pH Max"])

        for j in range(num_samples):
            obj = [
                crop_ids[crop],
                season_ids[season],
                random.uniform(rainfall_min, rainfall_max),
                random.uniform(relative_humidity_min, relative_humidity_max),
                random.uniform(min_temp, max_temp),
                random.uniform(p_h_min, p_h_max),
            ]
            data_values.append(obj)

    final_cols = ["Crop", "Season", "Rainfall", "Humidity", "Temp", "pH"]
    final_df = pd.DataFrame(data_values, columns=final_cols)

    # Separate input and output as X and y respectively
    X = final_df.iloc[:, 1:].values
    y = final_df.iloc[:, 0].values
    return X, y, crop_ids, season_ids


def get_trained_model(num_samples):
    X, y, crop_ids, season_ids = get_input_and_output(num_samples)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    # Train the KNN Model
    knn_classifier.fit(X, y)
    return knn_classifier, crop_ids, season_ids


def test_model():
    print("Evaulating model on 50 samples from ranges of each crop")
    X, y, _, _ = get_input_and_output()
    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # Create KNN Model object with k=3
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    # Train the KNN Model
    knn_classifier.fit(X_train, y_train)

    # predict the test values
    y_pred = knn_classifier.predict(X_test)

    # Evaluate model
    score = accuracy_score(y_test, y_pred)
    print("accuracy:\t%0.3f" % score)
    # confusion_matrix_val = confusion_matrix(y_test, y_pred)

    # print(classification_report(y_test, y_pred, zero_division=0))


# Scaling code can be used if required
def scale_data():
    # Normalize the range of features, so that data is on same scale and gives better
    # results, it's optional part and can be commented out.
    # Scaling part start
    # from sklearn.preprocessing import StandardScaler
    #
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    #
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # Scaling part end
    pass


if __name__ == '__main__':
    test_model()
