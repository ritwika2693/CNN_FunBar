# Loading Packages

from __future__ import division, print_function
import _ctypes
import numpy as np
import pandas as pd
import os
import time
import math
import statistics
from collections import Counter
from Bio import SeqIO
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
# from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing import sequence
from keras import initializers
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, ZeroPadding1D, TimeDistributed, Conv1D, MaxPooling1D, LSTM, Bidirectional, \
    Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau
from sklearn.utils import shuffle, resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, cohen_kappa_score, auc, roc_auc_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

Result_dir = 'Order_Classification_Results'


def order_model(data):
    Check_dir_Result = os.path.isdir(Result_dir)
    if not Check_dir_Result:
        os.makedirs(Result_dir)
        print("created folder : ", Result_dir)

    else:
        print(Result_dir, "folder already exists.")

    filepath = os.path.join('Results', 'classification_feature','features.csv')
    data = pd.read_csv(filepath)
    data1 = data.iloc[:, ]
    X = data1.iloc[:, 1:]

    scaler = StandardScaler()
    Z = scaler.fit(X)
    # print(Z)
    # print(scaler.mean_)
    X = scaler.transform(X)
    print(X)

    X = X.reshape(X.shape + (1,))
    X = np.array(X, dtype=float)
    order_path = os.path.join(os.getcwd(), 'static', 'Order_names.csv')
    order_names = pd.read_csv(order_path)
    order_names = order_names.iloc[:, ]

    # Order Level Prediction
    model_path = os.path.join(os.getcwd(), 'static',
                              'models', 'Order_Model.h5')

    # load the saved model
    order_model = load_model(model_path)
    order_model.summary()

    order_test_prediction = order_model.predict(X, verbose=1)

    predict_probability = pd.DataFrame(order_test_prediction)
    predict_probability.to_csv(
        "Order_Classification_Results/PredictionProbabilities_order.csv")

    # Setting up Appropriate thresholds for designating outlier points as "unknown order"

    predicted_order = []
    unknown = 'unknown'
    for i in order_test_prediction:
        if i.max() < 0.90:
            # print (unknown)
            predicted_order.append(unknown)
        else:
            # print (np.argmax(i))
            j = np.argmax(i)
            order = order_names["Order"][j]
            predicted_order.append(order)

    final_predictions = pd.DataFrame()
    final_predictions['Order'] = predicted_order
    final_predictions.to_csv(
        "Order_Classification_Results/Predicted_Orders.csv")
