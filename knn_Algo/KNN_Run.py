import pandas as pd
from  knn_Algo.KNN import KNN



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt



def RunKNN():


    file_path = 'pima-indians-diabetes.csv'
    df = pd.read_csv(file_path, header=None)

    # The last column (index 8) is the target, and the others are features
    X = df.drop(columns=[8])
    y = df[8]  # T



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracy_scores = []

    for k in k_values:
        knn = KNN(k=k)
        knn.fit(X_train.values, y_train.values)
        predictions = knn.predict(X_test.values)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)

    # Choose the best K
    best_k = k_values[np.argmax(accuracy_scores)]
    print(f"Best K: {best_k}")

    # Step 5: Evaluate the model with the best K
    knn = KNN(k=best_k)
    knn.fit(X_train.values, y_train.values)
    predictions = knn.predict(X_test.values)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    # F1 Score
    f1 = f1_score(y_test, predictions)
    # Precision
    precision = precision_score(y_test, predictions)
    # Recall
    recall = recall_score(y_test, predictions)
    # Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Mean Squared Error: {mse}")
    print(f"Confusion Matrix:\n{cm}")

    # Step 6: Plot accuracy vs K
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.title('Accuracy vs K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
