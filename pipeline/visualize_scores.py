import matplotlib.pyplot as plt
import numpy as np

ls1 = [(0.00002, 4.4), (0.0002, 3.4), (0.018, 1.0), (0.019, 1.8), (0.019, 2.2), (0.016, 3.4), (0.018, 2.2), (0.017, 2.9), (0.017, 4.4), (0.017, 1.4), (0.016, 4.2), (0.018, 1.8), (0.017, 1.8), (0.017, 2.8), (0.016, 8.2), (0.016, 3.9), (0.016, 7.2), (0.019, 2.8), (0.017, 2.8), (0.016, 4.9), (0.017, 4.4)]
# ls1 = [(0.0002, 4.4), (0.02, 1.), (0.018, 1.), (0.019, 1.8), (0.017, 2.9), (0.017, 4.4)]
# dict_debug = {'du/dt = k * du/dx , k = c[0]': (30922.0, 1.0),
#               'du/dt = c[0] * u * du/dx': (1.67, 1.4),
#               'du/dt = c[0] * u * du/dx + c[1] * t * du/dx': (1.65, 2.8),
#               'du/dt = c[0] * u * du/dx + c[1] * x * du/dx': (1.34, 2.8),
#               'du/dt = c[0] * du/dx + c[1] * t': (28595.0, 1.4),
#               'du/dt = c[0] * u * du/dx + c[1] * x * t * du/dx': (100, 1.4)}

# dict_debug = {'d^2u/dt^2 = c[0] * d^2u/dx^2': {282.0: 0.75},
#               'd^2u/dt^2 = c[0] * (du/dx)**2 + c[1] * d^2u/dx^2': {284.0: 1.75},
#               'd^2u/dt^2 = c[0] * du/dx + c[1] * d^2u/dx^2': {282.0: 1.25},
#               'd^2u/dt^2 = c[0] * (du/dx)^2 + c[1] * d^2u/dx^2 + c[2] * (t + x)': {280.0: 2.15},
#               'd^2u/dt^2 = c[0] * du/dx': {1000.0: 0.5}}


# wave eq: d^2u/dt^2 =
# dict_debug = {'c[0] * du/dx': {1000.0: 0.5},
#               'c[0] * (du/dx)**2 + c[1] * d^2u/dx^2': {284.0: 1.75},
#               'c[0] * d^2u/dx^2': {282.0: 0.75},
#               'c[0] * du/dx + c[1] * d^2u/dx^2': {282.0: 1.25},
#               'c[0] * t * du/dx': {1000.0: 0.7},
#               'c[0] * du/dx + c[1] * t * (du/dx)**2': {982.0: 1.7},
#               'c[0] * (du/dx)^2 + c[1] * d^2u/dx^2 + c[2] * (t + x)': {280.0: 2.15},
#               'c[0] * du/dx + c[1] * t ** 2': {915.0: 0.9},}


dict_debug = {'1000.0; 0.5': {1000.0: 0.5},
              '284.0; 1.75': {284.0: 1.75},
              '282.0; 0.75': {282.0: 0.75},
              '282.0; 1.25': {282.0: 1.25},
              '1000.0; 0.7': {1000.0: 0.7},
              '982.0; 1.7': {982.0: 1.7},
              '285.0; 2.15': {285.0: 2.15},
              '915.0; 0.9': {915.0: 0.9},
              '354.0; 2.4': {354.0: 2.4},
              '400.0; 2.4': {400.0: 2.4},
              '430.0; 2.25': {430.0: 2.25},
              '950.0: 0.4': {950.0: 0.4},
              '700.0: 0.4': {700.0: 0.4}}


# dict_debug = {'du/dt = c[0] * du/dx': {537.0: 0.5},
#               'du/dt = c[0] * du/dx + c[1] * du^2/dx^2': {532.0: 0.5},
#               'du/dt = c[0] * du/dx + c[1] * u * du/dx': {438.0: 1.2},
#               'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': {37.1: 1.95},
#               'du/dt = c[0] * u * du/dx + c[2] * d^2u/dx^2': {38.9: 1.45},
#               'du/dt = c[0] * u * du/dx + c[2] * d^2u/dx^2 + c[3] * du/dx * t': {37.1: 2.15},
#               'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': {542.0: 1.75},
#               'du/dt = c[0] * du/dx + c[1] * t * du/dx': {442.0: 1.2}}



def plot_track(opt_track):
    plt.grid()
    for point, name in zip(opt_track.values(), opt_track.keys()):
        x, y = point[0], point[1]
        mod = np.round(2.4*(1 - y / 1000) + x * 1.1, 2)
        plt.scatter(x, y, s=200, edgecolors='k')
        plt.text(x + .03, y + .015, f'{name}: {mod}', fontsize=9)

    plt.legend(opt_track.keys())
    plt.show()


if __name__ == "__main__":
    opt_track = {'c[0] * du/dx': (0.5, 1000.0),
                  'c[0] * (du/dx)**2 + c[1] * d^2u/dx^2': (1.75, 284.0),
                  'c[0] * d^2u/dx^2': (0.75, 282.0),
                  'c[0] * du/dx + c[1] * d^2u/dx^2': (1.25, 282.0),
                  'c[0] * t * du/dx': (0.7, 1000.0),
                  'c[0] * du/dx + c[1] * t * (du/dx)**2': (1.7, 982.0),
                  'c[0] * (du/dx)^2 + c[1] * d^2u/dx^2 + c[2] * (t + x)': (2.15, 285.0),
                  'c[0] * du/dx + c[1] * t ** 2': (0.9, 915.0),
                  '354.0; 2.4': (2.4, 354.0),
                  '400.0; 2.4': (2.4, 400.0),
                  '430.0; 2.25': (2.25, 430.0),
                  '950.0: 0.4': (0.4, 950.0),
                  '700.0: 0.4': (0.4, 700.0)
                  }
    plot_track(opt_track)
    # d = dict_debug
    # plt.grid()
    # for data_dict, name in zip(d.values(), d.keys()):
    #     x = float(list(data_dict.keys())[0])
    #     y = float(list(data_dict.values())[0])
    #     mod = np.round(2.4*(1 - x / 1000) + y * 1.1, 2)
    #     plt.scatter(y, x, s=200, edgecolors='k') # color=colors.pop()
    #     plt.text(y + .015, x + .03, f'{name}: {mod}', fontsize=9)
    #
    # plt.legend(d.keys())
    # plt.show()




    # scores = np.array([val[0] for val in vals])
    # complexities = np.array([val[1] for val in vals])
    # plt.scatter(scores, complexities, s=200)
    # plt.grid()
    # plt.xlabel("score")
    # plt.ylabel("complexity")
    # plt.show()

    # ls2 = np.sqrt(scores*scores + complexities*complexities)
    print()