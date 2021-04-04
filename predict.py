import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from main import get_trained_model


def predict(num_samples):
    season = input('Enter the Season: ')
    rf = float(input('Enter the Rainfall of the area: '))
    rh = float(input('Enter the Relative Humidity of the area: '))
    temp = float(input('Enter the Temperature: '))
    ph = float(input('Enter the soil pH: '))

    model, crop_ids, season_ids = get_trained_model(num_samples)

    X_test = np.array([[season_ids[season], rf, rh, temp, ph]])
    # X_test = np.array([[season_ids['Rabi'], 800.0, 80.0, 13.0, 6.0]])
    y_pred = model.predict(X_test)

    reverse_crop_ids = {val: key for key, val in crop_ids.items()}
    print(reverse_crop_ids[int(y_pred[0])])


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py num_samples")
        quit()
    num_samples = int(sys.argv[1])
    predict(num_samples)
