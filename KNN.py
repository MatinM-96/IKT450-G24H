import numpy as np
import torch
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt

class MyKnn:
    def __init__(self, k):
        self.k = k

    def fit (self,x, y):
        self.xTrain = x
        self.yTrain = y

    def calculateTheDistance(self, x_data):
        distances = []

        #iterating over each point and find the distance
        for x in  self.xTrain:
            #use the Euclidean Distance methode to find the distance between each point
            Squre_distance = (x_data- x)**2
            sumOFSquared =  np.sum(Squre_distance)
            distance = np.sqrt(sumOFSquared)
            #append each distance to our array
            distances.append(distance)


def runKNN():
    knn = MyKnn(5)
    # declare the file path
    file_path = 'pima-indians-diabetes.csv'
    #read the csv file
    df = pd.read_csv(file_path, header=None)
    x = df.drop(columns=[df.shape[1] - 1])  # All columns except the last one
    y = df[df.shape[1] - 1]  # The last column
    # Convert X and y to numpy arrays for easier handling
    x = x.values
    y = y.values
    model = MyKnn(k=3)
    model.fit(x, y)

    print("Matin mhoam")


