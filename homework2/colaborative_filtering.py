import numpy as np


def square_error(Y, U, V):
    X = U * V
    return (
        sum(
            [
                sum(
                    [
                        (Y[a][i] - X_a[i]) ** 2 if Y[a][i] != 0 else 0
                        for i, _ in enumerate(X_a)
                    ]
                )
                for a, X_a in enumerate(X)
            ]
        )
        / 2
    )


def regularization(U, V, lamb):
    return (np.linalg.norm(U) ** 2 + np.linalg.norm(V) ** 2) * (lamb / 2)


if __name__ == "__main__":
    U = np.array([[6], [0], [3], [6]])
    V = np.array([4, 2, 1])
    X = U * V
    print(X)

    Y = np.array([[5, 0, 7], [0, 2, 0], [4, 0, 0], [0, 3, 6]])
    print(square_error(Y, U, V))
    print(regularization(U, V, 1))
