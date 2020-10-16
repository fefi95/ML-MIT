import numpy as np


def square_loss(feature_matrix, labels, theta):
    def loss_single(feature_vector, label, theta):
        return label - np.dot(theta, feature_vector)

    loss = [
        loss_single(x, labels[i], theta) ** 2 / 2 for i, x in enumerate(feature_matrix)
    ]
    return sum(loss) / feature_matrix.shape[0]


def hinge_loss(feature_matrix, labels, theta):
    def loss_single(feature_vector, label, theta):
        z = label - np.dot(theta, feature_vector)
        return 1 - z if z < 1 else 0

    loss = [loss_single(x, labels[i], theta) for i, x in enumerate(feature_matrix)]
    return sum(loss) / feature_matrix.shape[0]


if __name__ == "__main__":
    X = [[1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1]]
    Y = [2, 2.7, -0.7, 2]
    theta = [0, 1, 2]
    hinge_loss = hinge_loss(
        np.array(X, dtype=float),
        np.array(Y, dtype=float),
        np.array(theta, dtype=float),
    )
    print(hinge_loss)
    square_loss = square_loss(
        np.array(X, dtype=float), np.array(Y, dtype=float), np.array(theta, dtype=float)
    )
    print(square_loss)
