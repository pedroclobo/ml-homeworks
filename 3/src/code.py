from scipy.io.arff import loadarff
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

if __name__ == "__main__":
	data = loadarff("kin8nm.arff")
	df = pd.DataFrame(data[0])
	X, y = df.drop("y", axis=1), df["y"]

	X_train, X_test, y_train, y_test = train_test_split(X,
	                                                    y,
	                                                    train_size=0.7,
	                                                    random_state=0)

	regressors = [
	    Ridge(alpha=0.1),
	    MLPRegressor(hidden_layer_sizes=(10, 10),
	                 max_iter=500,
	                 random_state=0,
	                 activation="tanh",
	                 early_stopping=True),
	    MLPRegressor(hidden_layer_sizes=(10, 10),
	                 max_iter=500,
	                 random_state=0,
	                 activation="tanh",
	                 early_stopping=False)
	]

	mae = []
	for regressor in regressors:
		regressor.fit(X_train, y_train)
		y_pred = regressor.predict(X_test)
		mae.append(metrics.mean_absolute_error(y_test, y_pred))

	print("Ridge MAE: {}.".format(mae[0]))
	print("MLP1 MAE: {}.".format(mae[1]))
	print("MLP2 MAE: {}.".format(mae[2]))

	print("MLP1 converged in {} iterations.".format(regressors[1].n_iter_))
	print("MLP2 converged in {} iterations.".format(regressors[2].n_iter_))

	plt.boxplot([
	    abs(y_test - y_pred)
	    for y_pred in [regressor.predict(X_test) for regressor in regressors]
	])
	plt.xticks([1, 2, 3], ["Ridge", "MLP1", "MLP2"])
	plt.ylabel("Residues")
	plt.show()

	plt.hist([
	    abs(y_test - y_pred)
	    for y_pred in [regressor.predict(X_test) for regressor in regressors]
	])
	plt.legend(["Ridge", "MLP1", "MLP2"])
	plt.xlabel("Residues")
	plt.ylabel("Frequency")
	plt.show()
