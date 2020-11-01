import numpy as np
import matplotlib.pyplot as plt


def f(z):
    return max(z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t1, c_t1, x_t, b_f, b_i, b_o, b_c):
    def simple_sigmoid(x):
        if x >= 1:
            return 1
        elif x <= -1:
            return 0
        else:
            return sigmoid(x)

    def simple_tanh(x):
        if x >= 1:
            return 1
        elif x <= -1:
            return -1
        else:
            return np.tanh(x)

    if h_t1 == 0.5 or h_t1 == -0.5:
        h_t1 = 0

    f_t = simple_sigmoid(Wfh * h_t1 + Wfx * x_t + b_f)
    i_t = simple_sigmoid(Wih * h_t1 + Wix * x_t + b_i)
    o_t = simple_sigmoid(Woh * h_t1 + Wox * x_t + b_o)
    c_t = f_t * c_t1 + i_t * simple_tanh(Wch * h_t1 + Wcx * x_t + b_c)
    h_t = o_t * simple_tanh(c_t)
    return (c_t, h_t)


"""
Part 1
Consider the neural network given in the figure below, with ReLU activation functions
(denoted by  f ) on all neurons, and a softmax activation function in the output layer

Given an input x = [x_1, x_2] and the hidden units:
z1 = x_1 W_11 + x_2_W_21 + W_01    f(z1) = max{z1,0}
z2 = x_1 W_12 + x_2 W_22 + W_02    f(z2) = max{z2,0}
z3 = x_1 W_13 + x_2 W_23 + W_03    f(z3) = max{z3,0}
z4 = x_1 W_14 + x_2 W_24 + W_04    f(z4) = max{z4,0}

u1 = f(z1) V11 + f(z2) V_21 + f(z3) V31 + f(z4) V41 + V01    f(u1) = max{u1,0}
u2 = f(z1) V12 + f(z2) V_22 + f(z3) V32 + f(z4) V42 + V02    f(u2) = max{u2,0}

and output layer uses the softmax function:

o1 = e^f(u1) / (e^f(u1) + e^f(u2))
o2 = e^f(u2) / (e^f(u1) + e^f(u2))
"""

if __name__ == "__main__":

    W = np.transpose(np.array([[-1, 1, 0], [-1, 0, 1], [-1, -1, 0], [-1, 0, -1]]))
    V = np.transpose(np.array([[0, 1, 1, 1, 1], [2, -1, -1, -1, -1]]))

    # 1.1 Feed forward step - What is the final output?
    x_1 = 3
    x_2 = 14

    z1 = x_1 * W[1][0] + x_2 * W[2][0] + W[0][0]
    z2 = x_1 * W[1][1] + x_2 * W[2][1] + W[0][1]
    z3 = x_1 * W[1][2] + x_2 * W[2][2] + W[0][2]
    z4 = x_1 * W[1][3] + x_2 * W[2][3] + W[0][3]

    u1 = f(z1) * V[1][0] + f(z2) * V[2][0] + f(z3) * V[3][0] + f(z4) * V[4][0] + V[0][0]
    u2 = f(z1) * V[1][1] + f(z2) * V[2][1] + f(z3) * V[3][1] + f(z4) * V[4][1] + V[0][1]

    o1 = np.exp(f(u1)) / (np.exp(f(u1)) + np.exp(f(u2)))
    o2 = np.exp(f(u2)) / (np.exp(f(u1)) + np.exp(f(u2)))

    print("o1 = ", o1, " o2 = ", o2)
    print("f(u1) = ", f(u1), " f(u2) = ", f(u2))

    # 1.2. Decision boundaries
    # 0 = x_1 * W[1][0] + x_2 * W[2][0] + W[0][0]
    # x_1 = (- x_2 * W[2][0] - W[0][0])/W[1][0]
    random_vector = np.random.rand(10)
    z1_x1 = (-random_vector * W[2][0] - W[0][0]) / W[1][0]
    plt.plot(z1_x1, random_vector)

    z2_x2 = np.full(10, -W[0][1] / W[2][1])
    plt.plot(random_vector, z2_x2)

    z3_x1 = (-random_vector * W[2][2] - W[0][2]) / W[1][2]
    plt.plot(z3_x1, random_vector)

    z4_x2 = np.full(10, -W[0][3] / W[2][3])
    plt.plot(random_vector, z4_x2)

    # plt.show()

    # 1.3. Output of neural network

    print(
        f"u1 = f(z1) * {V[1][0]} + f(z2) * {V[2][0]} + f(z3) * {V[3][0]} + f(z4) * {V[4][0]} + {V[0][0]}"
    )
    print(
        f"u2 = f(z1) * {V[1][1]} + f(z2) * {V[2][1]} + f(z3) * {V[3][1]} + f(z4) * {V[4][1]} + {V[0][1]}"
    )
    # sum(f(z_i)) = 1
    print("o1 = np.exp(1) / (np.exp(1) + np.exp(1))")

    # sum(f(z_i) = 0)
    print("o1 = np.exp(0) / (np.exp(0) + np.exp(2))")

    # sum(f(z_i) = 3)
    print("o1 = np.exp(3) / (np.exp(3) + np.exp(-1))")

    # 2. LSTM

    # ft = sigmoid(W^(f,h)* h_(t − 1) + W^(f,x) * x_t + b_f)
    # it = sigmoid(W^(i,h)* h_(t − 1) + W^(i,x) * x_t + b_i)
    # ot = sigmoid(W^(o,h)* h_(t − 1) + W^(o,x) * x_t + b_o)
    # ct = f_t ⊙ c_(t−1) + i_t ⊙ tanh(W^(c,h)* h_(t − 1) + W^(c,x) * x_t + b_c)
    # ht = o_t ⊙ tanh(c_t)

    Wfh = 0
    Wih = 0
    Woh = 0
    Wfx = 0
    Wix = 100
    Wox = 100
    b_f = -100
    b_i = 100
    b_o = 0
    Wch = -100
    Wcx = 50
    b_c = 0

    #            ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t1, c_t1, x_t, b_f, b_i, b_o, b_c)
    c_t0, h_t0 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, 0, 0, 0, b_f, b_i, b_o, b_c)
    c_t1, h_t1 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t0, c_t0, 0, b_f, b_i, b_o, b_c)
    c_t2, h_t2 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t1, c_t1, 1, b_f, b_i, b_o, b_c)
    c_t3, h_t3 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t2, c_t2, 1, b_f, b_i, b_o, b_c)
    c_t4, h_t4 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t3, c_t3, 1, b_f, b_i, b_o, b_c)
    c_t5, h_t5 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t4, c_t4, 0, b_f, b_i, b_o, b_c)
    print(f'LSTM = [{h_t0}, {h_t1}, {h_t2}, {h_t3}, {h_t4}, {h_t5}]')

    c_t0, h_t0 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, 0, 0, 1, b_f, b_i, b_o, b_c)
    c_t1, h_t1 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t0, c_t0, 1, b_f, b_i, b_o, b_c)
    c_t2, h_t2 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t1, c_t1, 0, b_f, b_i, b_o, b_c)
    c_t3, h_t3 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t2, c_t2, 1, b_f, b_i, b_o, b_c)
    c_t4, h_t4 = ht(Wfh, Wih, Woh, Wfx, Wix, Wox, Wch, Wcx, h_t3, c_t3, 1, b_f, b_i, b_o, b_c)
    print(f'LSTM = [{h_t0}, {h_t1}, {h_t2}, {h_t3}, {h_t4}]')



    # 3 Backpropagation

    # 3.3 Simple network
    # z1 = w1*x
    # a1 = ReLU(z1)
    # z2 = w2*a + b
    # y = sigmoid(z2)
    # C = (1/2)*(y-t)^2
    t = 1
    x = 3
    w1 = 0.01
    w2 = -5
    b = -1
    z1 = w1 * x
    a1 = max(0, z1)  # ReLU
    z2 = w2 * a1 + b
    y = sigmoid(z2)
    C = (1 / 2) * (y - t) ** 2
    print(C)
    derivative_w1 = (
        np.exp(-b - w2 * w1 * x) * w2 * (1 / (1 + np.exp(-b - w2 * w1 * x)) - t) * x
    ) / (1 + np.exp(-b - w2 * w1 * x)) ** 2
    print(derivative_w1)
    derivative_w2 = (
        np.exp(-b - w2 * w1 * x) * w1 * (1 / (1 + np.exp(-b - w2 * w1 * x)) - t) * x
    ) / (1 + np.exp(-b - w2 * w1 * x)) ** 2
    print(derivative_w2)
    derivative_b = (
        np.exp(-b - w1 * w2 * x) * (1 / (1 + np.exp(-b - w1 * w2 * x)) - t)
    ) / (1 + np.exp(-b - w1 * w2 * x)) ** 2
    print(derivative_b)
