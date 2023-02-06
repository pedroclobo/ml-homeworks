from scipy.io.arff import loadarff
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def purity_score(y_true, y_pred):
	confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
	return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


if __name__ == "__main__":
	data = loadarff("pd_speech.arff")
	df = pd.DataFrame(data[0])
	df["class"] = df["class"].astype(int)
	X, y = df.drop("class", axis=1), df["class"]
	X = MinMaxScaler().fit_transform(X)

	LABELS = None
	for seed in (0, 1, 2):
		kmeans = cluster.KMeans(n_clusters=3, random_state=seed).fit(X)
		print("Seed {} Silhouette: {}".format(
		    seed, metrics.silhouette_score(X, kmeans.labels_)))
		print("Seed {} Purity: {}".format(seed,
		                                  purity_score(y, kmeans.labels_)))
		if (seed == 0):
			LABELS = kmeans.labels_

	X_variance = X[:, np.argsort(np.var(X, axis=0))[::-1][:2]]
	f1, f2 = X_variance[:, 0], X_variance[:, 1]

	_, ax = plt.subplots(1, 2)
	sns.scatterplot(x=f1, y=f2, hue=y, ax=ax[0])
	ax[0].set_title("Original Parkinson diagnoses")
	ax[0].legend(loc="center right", title="Class")
	ax[0].set_xlabel("Feature 1")
	ax[0].set_ylabel("Feature 2")

	sns.scatterplot(x=f1, y=f2, hue=LABELS, ax=ax[1])
	ax[1].set_title("KMeans Clusters")
	ax[1].legend(loc="center right", title="Cluster")
	ax[1].set_xlabel("Feature 1")
	ax[1].set_ylabel("Feature 2")

	plt.show()

	pca = PCA(n_components=0.8, svd_solver='full')
	pca.fit(X)
	print("{} principal components are necessary.".format(pca.n_components_))
