import matplotlib.pyplot as plt
import numpy as np

ls1 = [(1.67, 1.4), (1.65, 2.8), (1.34, 2.8), (1.67, 1.8), (1.54, 3.2), (104.0, 1.4)]
dict_debug = {'du/dt = k * du/dx , k = c[0]': (30922.0, 1.0),
              'du/dt = c[0] * u * du/dx': (1.67, 1.4),
              'du/dt = c[0] * u * du/dx + c[1] * t * du/dx': (1.65, 2.8),
              'du/dt = c[0] * u * du/dx + c[1] * x * du/dx': (1.34, 2.8),
              'du/dt = c[0] * du/dx + c[1] * t': (28595.0, 1.4),
              'du/dt = c[0] * u * du/dx + c[1] * x * t * du/dx': (100, 1.4)}

if __name__ == "__main__":
    vals = list(dict_debug.values())
    vals=ls1
    # x = [lambda i: val[0] for val in vals]
    scores = np.array([val[0] for val in vals])
    complexities = np.array([val[1] for val in vals])
    plt.scatter(scores, complexities, s=200)
    plt.grid()
    plt.xlabel("score")
    plt.ylabel("complexity")
    plt.show()

    ls2 = np.sqrt(scores*scores + complexities*complexities)
    print()