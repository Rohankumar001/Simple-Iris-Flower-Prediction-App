import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Train the model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Save the model to a pickle file
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
