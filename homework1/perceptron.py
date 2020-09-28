import numpy as np


def perceptron(x, y):
    """
    Preceptron algorithm through the origin

    Perceptron ({(x(i),y(i)),i=1,...,n},T):
        initialize  θ=0 (vector);
        for  t=1,...,T  do
            for  i=1,...,n  do
                if  y(i)(θ⋅x(i))≤0  then
            update  θ=θ+y(i)x(i)
    """
    theta = np.zeros(x[0].shape)
    T = 10
    number_of_mistakes = 0
    progression_of_thetas = []

    for t in range(T):
        for i, _ in enumerate(x):
            # i means the i-th sample in the training set. Is another vector
            # print("dot", np.dot(theta, x[i])," y=", y[i])
            if y[i] * np.dot(theta, x[i]) <= 0:
                theta = theta + y[i] * x[i]
                number_of_mistakes = number_of_mistakes + 1
                progression_of_thetas.append(theta)

    print("Perceptron", 5 * "#")
    print([str(t) for t in progression_of_thetas])
    print(number_of_mistakes)
    print(10 * "#")
    return number_of_mistakes, progression_of_thetas


if __name__ == "__main__":
    x_1 = np.array([-1, -1], dtype=float)
    x_2 = np.array([1, 0], dtype=float)
    x_3 = np.array([-1, 1.5], dtype=float)

    # ---- Exercise a ---- #
    # Starts with x_1
    X = np.array([x_1, x_2, x_3], dtype=float)
    Y = np.array([1, -1, 1])
    perceptron(X, Y)

    # Starts with x_2
    X2 = X = np.array([x_2, x_3, x_1], dtype=float)
    Y2 = np.array([-1, 1, 1])
    perceptron(X2, Y2)

    # ---- Exercise c ---- #
    new_x_3 = np.array([-1, 10], dtype=float)

    # Starts with x_1
    X3 = np.array([x_1, x_2, new_x_3], dtype=float)
    Y3 = np.array([1, -1, 1])
    perceptron(X3, Y3)

    # Starts with x_2
    X4 = X = np.array([x_2, new_x_3, x_1], dtype=float)
    Y4 = np.array([-1, 1, 1])
    perceptron(X4, Y4)
