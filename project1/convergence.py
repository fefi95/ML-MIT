import matplotlib.pyplot as plt
from project1 import hinge_loss_full


def convergence(algorithm, feature_matrix, labels, **kwargs):
    conv = False
    costs = []
    max_iter = 200
    cost = 0

    i = 10  # Initialize the counter of iterations.
    while not (conv) and (i <= max_iter):
        theta, theta_0 = algorithm(feature_matrix, labels, i, **kwargs)
        current_cost = 1 / hinge_loss_full(feature_matrix, labels, theta, theta_0)
        costs.append(current_cost)

        i = i + 1  # Update the current number of iterations.
        if abs(current_cost - cost) <= 0.001:
            conv = True
        cost = current_cost

    return (costs, i)


def plot_convergence(costs, i):
    iterations = range(1, len(costs) + 1)
    plt.plot(iterations, costs, c="red", linewidth=1.5)
    plt.xlabel("Iteration numbers")
    plt.ylabel("theta")
    plt.title("convergence")
    plt.grid(True)
    plt.show()
