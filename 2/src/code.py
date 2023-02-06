from scipy.io.arff import loadarff
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

if __name__ == "__main__":
	data = loadarff("pd_speech.arff")
	df = pd.DataFrame(data[0])
	df["class"] = df["class"].astype(int)
	X, y = df.drop("class", axis=1), df["class"]

	classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB()]
	matrices = [np.zeros((2, 2)), np.zeros((2, 2))]
	skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
	acc = [[], []]

	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]

		for i in range(len(classifiers)):
			classifiers[i].fit(X_train, y_train)
			y_pred = classifiers[i].predict(X_test)
			matrices[i] += confusion_matrix(y_test, y_pred)
			acc[i].append(classifiers[i].score(X_test, y_test))

	for matrix in matrices:
		matrix = pd.DataFrame(matrix,
		                      index=["0", "1"],
		                      columns=["Predicted 0", "Predicted 1"])
		sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues")
		plt.show()

	print(
	    "kNN is statistically superior to Naive Bayes regarding accuracy is suported by a p-value of {}."
	    .format(stats.ttest_rel(acc[0], acc[1], alternative="greater").pvalue))
