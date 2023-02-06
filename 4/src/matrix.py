import numpy as np
import math

X_1 = np.array([[1.0], [2.0]])
X_2 = np.array([[-1.0], [1.0]])
X_3 = np.array([[1.0], [0.0]])

MU_1 = np.array([[2.0], [2.0]])
MU_2 = np.array([[0.0], [0.0]])
SIGMA_1 = np.array([[2.0, 1.0], [1.0, 2.0]])
SIGMA_2 = np.array([[2.0, 0.0], [0.0, 2.0]])
PI_1 = 0.5
PI_2 = 0.5

DECIMALS = 3


def x_mid_c(x, sigma, mu):
	return (1 / ((2 * math.pi) * np.linalg.det(sigma)**(0.5)) *
	        math.e**(-0.5 * (
	            (x - mu).transpose().dot(np.linalg.inv(sigma)).dot(x - mu))))


if __name__ == "__main__":
	print("1.")
	x1_mid_c1 = x_mid_c(X_1, SIGMA_1, MU_1)
	print("\tx1 | c1 = {}".format(x1_mid_c1.round(DECIMALS)))
	x1_mid_c2 = x_mid_c(X_1, SIGMA_2, MU_2)
	print("\tx1 | c2 = {}".format(x1_mid_c2.round(DECIMALS)))
	x2_mid_c1 = x_mid_c(X_2, SIGMA_1, MU_1)
	print("\tx2 | c1 = {}".format(x2_mid_c1.round(DECIMALS)))
	x2_mid_c2 = x_mid_c(X_2, SIGMA_2, MU_2)
	print("\tx2 | c2 = {}".format(x2_mid_c2.round(DECIMALS)))
	x3_mid_c1 = x_mid_c(X_3, SIGMA_1, MU_1)
	print("\tx3 | c1 = {}".format(x3_mid_c1.round(DECIMALS)))
	x3_mid_c2 = x_mid_c(X_3, SIGMA_2, MU_2)
	print("\tx3 | c2 = {}".format(x3_mid_c2.round(DECIMALS)))

	print()

	x1_c1 = PI_1 * x1_mid_c1
	print("\tx1, c1 = {}".format(x1_c1.round(DECIMALS)))
	x1_c2 = PI_2 * x1_mid_c2
	print("\tx1, c2 = {}".format(x1_c2.round(DECIMALS)))
	x2_c1 = PI_1 * x2_mid_c1
	print("\tx2, c1 = {}".format(x2_c1.round(DECIMALS)))
	x2_c2 = PI_2 * x2_mid_c2
	print("\tx2, c2 = {}".format(x2_c2.round(DECIMALS)))
	x3_c1 = PI_1 * x3_mid_c1
	print("\tx1, c1 = {}".format(x3_c1.round(DECIMALS)))
	x3_c2 = PI_2 * x3_mid_c2
	print("\tx1, c2 = {}".format(x3_c2.round(DECIMALS)))

	print()

	x1 = x1_c1 + x1_c2
	print("\tx1 = {}".format(x1.round(DECIMALS)))
	x2 = x2_c1 + x2_c2
	print("\tx2 = {}".format(x2.round(DECIMALS)))
	x3 = x3_c1 + x3_c2
	print("\tx3 = {}".format(x3.round(DECIMALS)))

	print()

	gamma_11 = x1_c1 / x1
	print("\tγ11 = {}".format(gamma_11.round(DECIMALS)))
	gamma_21 = x1_c2 / x1
	print("\tγ21 = {}".format(gamma_21.round(DECIMALS)))
	gamma_12 = x2_c1 / x2
	print("\tγ12 = {}".format(gamma_12.round(DECIMALS)))
	gamma_22 = x2_c2 / x2
	print("\tγ22 = {}".format(gamma_22.round(DECIMALS)))
	gamma_13 = x3_c1 / x3
	print("\tγ13 = {}".format(gamma_13.round(DECIMALS)))
	gamma_23 = x3_c2 / x3
	print("\tγ23 = {}".format(gamma_23.round(DECIMALS)))

	print()

	n1 = gamma_11 + gamma_12 + gamma_13
	print("\tn1 = {}".format(n1.round(DECIMALS)))
	n2 = gamma_21 + gamma_22 + gamma_23
	print("\tn2 = {}".format(n2.round(DECIMALS)))

	print()

	mu_1_new = (1 / n1) * (gamma_11 * X_1 + gamma_12 * X_2 + gamma_13 * X_3)
	print("\tmu_1 = {}".format(mu_1_new.round(DECIMALS)))
	mu_2_new = (1 / n2) * (gamma_21 * X_1 + gamma_22 * X_2 + gamma_23 * X_3)
	print("\tmu_2 = {}".format(mu_2_new.round(DECIMALS)))

	print()

	sigma_1_new = (1 / n1) * (gamma_11 * ((X_1 - mu_1_new).dot(
	    (X_1 - mu_1_new).transpose())) + gamma_12 * ((X_2 - mu_1_new).dot(
	        (X_2 - mu_1_new).transpose())) + gamma_13 * ((X_3 - mu_1_new).dot(
	            (X_3 - mu_1_new).transpose())))
	print("\tsigma_1 = {}".format(sigma_1_new.round(DECIMALS)))
	sigma_2_new = (1 / n2) * (gamma_21 * ((X_1 - mu_2_new).dot(
	    (X_1 - mu_2_new).transpose())) + gamma_22 * ((X_2 - mu_2_new).dot(
	        (X_2 - mu_2_new).transpose())) + gamma_23 * ((X_3 - mu_2_new).dot(
	            (X_3 - mu_2_new).transpose())))
	print("\tsigma_2 = {}".format(sigma_2_new.round(DECIMALS)))

	print()

	pi_1_new = (n1 / 3)
	print("\tpi_1 = {}".format(pi_1_new.round(DECIMALS)))
	pi_2_new = (n2 / 3)
	print("\tpi_2 = {}".format(pi_2_new.round(DECIMALS)))

	print("2. a)")

	x1_mid_c1_new = x_mid_c(X_1, sigma_1_new, mu_1_new)
	print("\tx1 | c1 = {}".format(x1_mid_c1_new.round(DECIMALS)))
	x1_mid_c2_new = x_mid_c(X_1, sigma_2_new, mu_2_new)
	print("\tx1 | c2 = {}".format(x1_mid_c2_new.round(DECIMALS)))
	x2_mid_c1_new = x_mid_c(X_2, sigma_1_new, mu_1_new)
	print("\tx2 | c1 = {}".format(x2_mid_c1_new.round(DECIMALS)))
	x2_mid_c2_new = x_mid_c(X_2, sigma_2_new, mu_2_new)
	print("\tx2 | c2 = {}".format(x2_mid_c2_new.round(DECIMALS)))
	x3_mid_c1_new = x_mid_c(X_3, sigma_1_new, mu_1_new)
	print("\tx3 | c1 = {}".format(x3_mid_c1_new.round(DECIMALS)))
	x3_mid_c2_new = x_mid_c(X_3, sigma_2_new, mu_2_new)
	print("\tx3 | c2 = {}".format(x3_mid_c2_new.round(DECIMALS)))

	print("\tx1: max({}, {}) = {}".format(
	    (pi_1_new * x1_mid_c1_new).round(DECIMALS),
	    (pi_2_new * x1_mid_c2_new).round(DECIMALS),
	    max(pi_1_new * x1_mid_c1_new,
	        pi_2_new * x1_mid_c2_new).round(DECIMALS)))
	print("\tx2: max({}, {}) = {}".format(
	    (pi_1_new * x2_mid_c1_new).round(DECIMALS),
	    (pi_2_new * x2_mid_c2_new).round(DECIMALS),
	    max(pi_1_new * x2_mid_c1_new,
	        pi_2_new * x2_mid_c2_new).round(DECIMALS)))
	print("\tx3: max({}, {}) = {}".format(
	    (pi_1_new * x3_mid_c1_new).round(DECIMALS),
	    (pi_2_new * x3_mid_c2_new).round(DECIMALS),
	    max(pi_1_new * x3_mid_c1_new,
	        pi_2_new * x3_mid_c2_new).round(DECIMALS)))

	print("2. b)")
	silh_x2 = (np.linalg.norm(X_2 - X_1) - np.linalg.norm(X_2 - X_3)) / max(
	    np.linalg.norm(X_2 - X_3), np.linalg.norm(X_2 - X_1))
	print("\tsilhouette(x2) = {}".format(silh_x2.round(DECIMALS)))

	silh_x3 = (np.linalg.norm(X_3 - X_1) - np.linalg.norm(X_3 - X_2)) / max(
	    np.linalg.norm(X_3 - X_2), np.linalg.norm(X_3 - X_1))
	print("\tsilhouette(x3) = {}".format(silh_x3.round(DECIMALS)))

	print("\tsilhouette(c2) = {}".format(
	    (0.5 * silh_x2 + 0.5 * silh_x3).round(DECIMALS)))
