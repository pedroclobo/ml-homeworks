import numpy as np
import math

X = np.array([[0.8**0, 0.8**1, 0.8**2, 0.8**3], [1**0, 1**1, 1**2, 1**3],
              [1.2**0, 1.2**1, 1.2**2,
               1.2**3], [1.4**0, 1.4**1, 1.4**2, 1.4**3],
              [1.6**0, 1.6**1, 1.6**2, 1.6**3]])

w = None
z = np.array([24, 20, 10, 13, 12])


def pergunta_1():
	global w
	global z
	l_indentity = 2 * np.identity(4)
	w = np.linalg.inv(X.transpose().dot(X) + l_indentity).dot(
	    X.transpose()).dot(z)


def pergunta_2():
	global w
	global z
	sum = 0
	for i in range(5):
		sum += (z[i] - w.dot(X[i]))**2

	print("Error = {}".format((1/5) * sum))
	return np.sqrt((1/5) * sum)


def pergunta_3():
	w_1 = np.array([1.0, 1.0])
	b_1 = np.array([1.0, 1.0])
	w_2 = np.array([1.0, 1.0])
	b_2 = 1

	l_b1s = np.array([0.0, 0.0])
	l_b2s = 0.0
	l_w1s = np.array([0.0, 0.0])
	l_w2s = np.array([0.0, 0.0])

	# Forward
	for x, z in ((0.8, 24.0), (1.0, 20.0), (1.2, 10.0)):
		print("x = {}\n".format(x))
		a_0 = float(x)

		net_1 = w_1 * a_0 + b_1
		print("net1 = {}.".format(net_1))

		a_1 = np.array(net_1)
		a_1[0] = math.e**(0.1 * a_1[0])
		a_1[1] = math.e**(0.1 * a_1[1])
		print("a1 = {}.".format(a_1))

		net_2 = w_2.dot(a_1) + b_2
		print("net2 = {}.".format(net_2))

		a_2 = math.e**(0.1 * net_2)
		print("a2 = {}.".format(a_2))

		print()

		# Backwards
		l_a2 = a_2 - z
		print("l_a2 = {}.".format(l_a2))

		a2_net2 = 0.1 * math.e**(0.1 * net_2)
		print("a2_net2 = {}.".format(a2_net2))

		net2_a1 = w_2.transpose()
		print("net2_a1 = {}.".format(net2_a1))

		a1_net1 = np.array([[0.1 * math.e**(0.1 * net_1[0]), 0.0],
		                    [0.0, 0.1 * math.e**(0.1 * net_1[1])]])
		print("a1_net1 = \n{}.".format(a1_net1))

		delta_2 = a2_net2 * l_a2
		print("delta_2 = {}".format(delta_2))

		delta_1 = a1_net1.dot(net2_a1) * delta_2
		print("delta_1 = {}".format(delta_1))

		l_b2 = delta_2
		l_b2s += l_b2
		print("l_b2 = {}".format(l_b2))

		l_b1 = delta_1
		l_b1s += l_b1
		print("l_b1 = {}".format(l_b1))

		l_w2 = delta_2 * a_1.transpose()
		l_w2s += l_w2
		print("l_w2 = {}".format(l_w2))

		l_w1 = delta_1 * x
		l_w1s += l_w1
		print("l_w1 = {}".format(l_w1))

		print()

	print("l_b2s = {}".format(l_b2s))
	print("l_b1s = {}".format(l_b1s))
	print("l_w2s = {}".format(l_w2s))
	print("l_w1s = {}".format(l_w1s))

	print()

	print("b2 = {}".format(1 - 0.1 * l_b2s))
	print("b1 = {}".format(b_1 - 0.1 * l_b1s))
	print("w2 = {}".format(w_2 - 0.1 * l_w2s))
	print("w1 = {}".format(w_1 - 0.1 * l_w1s))


if __name__ == "__main__":
	pergunta_1()
	print(w)
	pergunta_1()
	print(pergunta_2())
	pergunta_3()
