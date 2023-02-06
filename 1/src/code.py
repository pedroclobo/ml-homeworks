from scipy.io.arff import loadarff
from sklearn import metrics, tree
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
	data, meta = loadarff("pd_speech.arff")
	df = pd.DataFrame(data)
	df["class"] = df["class"].str.decode("utf-8")
	df.dropna(inplace=True)

	X, y = df.iloc[:, 0:-1], df.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X,
	                                                    y,
	                                                    train_size=0.7,
	                                                    stratify=y,
	                                                    random_state=1)

	predictor = tree.DecisionTreeClassifier()
	num_features = (5, 10, 40, 100, 250, 700)
	training_accuracy = []
	testing_accuracy = []

	for features in num_features:
		selector = SelectKBest(mutual_info_classif, k=features)
		selector.fit(X_train, y_train)
		X_train_new = selector.transform(X_train)
		X_test_new = selector.transform(X_test)

		predictor.fit(X_train_new, y_train)
		y_pred_train = predictor.predict(X_train_new)
		y_pred_test = predictor.predict(X_test_new)

		training_accuracy += [metrics.accuracy_score(y_train, y_pred_train)]
		testing_accuracy += [metrics.accuracy_score(y_test, y_pred_test)]

	plt.plot(num_features, training_accuracy, marker=".")
	plt.plot(num_features, testing_accuracy, marker=".")
	plt.xlabel("Number of Features")
	plt.ylabel("Accuracy")
	plt.legend(["Training Accuracy", "Testing Accuracy"])
	plt.show()
