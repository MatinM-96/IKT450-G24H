import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt

class MyKnn:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.xTrain = x
        self.yTrain = y

    def calculateTheDistance(self, x_data):
        distances = []

        # Iterating over each point and finding the distance
        for x in self.xTrain:

            # Use the Euclidean Distance method to find the distance between each point
            square_distance = (x_data - x) ** 2
            sum_of_squares = np.sum(square_distance)
            distance = np.sqrt(sum_of_squares)

            # Append each distance to our array
            distances.append(distance)

        return distances

    def predict(self, x_data):

        distances = self.calculateTheDistance(x_data)
        sorted_indices = np.argsort(distances)
        k_nearest_labels = []

        # Find the k nearest labels and append them to an array
        for i in range(self.k):
            k_nearest_labels.append(self.yTrain[sorted_indices[i]])

        # .most_common(1): Find the most frequent label among the nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)

        # Return most_common[0][0]: Extract and return the most frequent label
        return most_common[0][0]

    def predict_batch(self, x_data):

        predicted_labels = []
        for x in x_data:
            prediction = self.predict(x)
            predicted_labels.append(prediction)
        return np.array(predicted_labels)

def evaluate_knn(k, X_train, X_test, y_train, y_test):
    knn = MyKnn(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict_batch(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        'k': k,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'mse': mse,
        'conf_matrix': conf_matrix
    }

def plot_metrics(results):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot([r['k'] for r in results], [r['accuracy'] for r in results], marker='o', label='Accuracy')
    plt.title('Accuracy vs k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r['k'] for r in results], [r['mse'] for r in results], marker='o', label='Mean Squared Error')
    plt.title('Mean Squared Error vs k')
    plt.xlabel('k')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def runKNN():
    # Step 1: Load the dataset
    df = pd.read_csv('pima-indians-diabetes.csv', header=None)
    # Assuming the last column is the target (Outcome)
    X = df.iloc[:, :-1]  # All columns except the last one (features)
    y = df.iloc[:, -1]  # The last column (target)

    # Step 2: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    # Step 3: Experiment with different values of k
    k_values = [1, 3, 5, 7, 9]
    results = []
    for k in k_values:
        result = evaluate_knn(k, X_train, X_test, y_train, y_test)
        results.append(result)

    # Step 4: Display the evaluation metrics for each k
    for res in results:
        print(f"k={res['k']}: Accuracy={res['accuracy']:.4f}, F1 Score={res['f1_score']:.4f}, "
              f"Precision={res['precision']:.4f}, Recall={res['recall']:.4f}, MSE={res['mse']:.4f}")
        print(f"Confusion Matrix:\n{res['conf_matrix']}\n")

    # Step 5: Plot accuracy and loss graph
    plot_metrics(results)

if __name__ == "__main__":
    runKNN()
