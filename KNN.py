import numpy as np


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


            #use the Euclidean Distance metode to find the distance between each point
            Squre_distance = (x_data- x)**2
            sumOFSquared =  np.sum(Squre_distance)
            distance = np.sqrt(sumOFSquared)


            #append each distnace to our array
            distances.append(distance)

        sortedIndicate = np.argsort(distances)

        lastResult = []


        for i in range(self.k):
            lastResult.append(sortedIndicate[i])

        return lastResult


def runKNN():
    knn = MyKnn(5)

    # declare the file path
    file_path = 'pima-indians-diabetes.csv'

    #read the csv file
    df = pd.read_csv(file_path, header=None)

    x = df.drop(columns=[df.shape[1] - 1])  # All columns except the last one

   # normlay in the last clomun stored the answer
    y = df[df.shape[1] - 1]  # The last column

    # Convert X and y to numpy arrays for easier handling
    x = x.values
    y = y.values

    model = MyKnn(k=3)

    model.fit(x, y)

    test_index = 0  # Index of the data point you want to test
    test_data_point = x[test_index]


    distance = model.calculateTheDistance(test_data_point)

    print(f"Distances from test data point at index {test_index} to all training points:")
    print(distance)




    print("Matin mhoam")


