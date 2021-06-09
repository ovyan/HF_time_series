import numpy as np
import matplotlib.pyplot as plt


def get_csv_data():
    from pandas import read_csv
    series = read_csv('LSTM/daily-min-temperatures.csv', header=0,
                      index_col=0, parse_dates=True, squeeze=True)

    return series.values


def get_rand_train():
    np.random.seed(0)

    EPISODES = 10
    TRAJ = 200
    TRAJ_SIZE = 20
    SIZE_T = 500

    Xdata, Ydata = [], []
    for episode in range(EPISODES):
        # t = np.linspace(.1, 8 * np.pi, SIZE_T)
        # t = np.sin(t) + np.log(t)
        # t += np.random.normal(0, 0.03, SIZE_T)
        t = get_csv_data()[:SIZE_T]
        y = np.sqrt(t) * 0
        arr = []
        arr_y = []
        for i in range(0, TRAJ, TRAJ_SIZE):
            arr.append(t[i: i + TRAJ_SIZE])
            if i != TRAJ - TRAJ_SIZE:
                arr_y.append([y[i]])
        # arr.append(t[0:TRAJ])
        # arr.append(t[0:TRAJ] * 0)
        # arr_y.append([y[0:2]])
        Xdata.append(arr)
        Ydata.append(arr_y)

    print(np.dstack(Ydata).shape)
    Xdata = np.stack(np.array(Xdata), axis=2)
    # Ydata = np.dstack(Ydata).transpose((1, 0, 2))
    Ydata = np.dstack(Ydata)
    print("TRAIN", Xdata.shape, Ydata.shape)
    return Xdata, Ydata


def get_rand_test():
    np.random.seed(1)

    EPISODES = 2
    TRAJ = 400
    TRAJ_SIZE = 20
    SIZE_T = 500

    Xdata, Ydata = [], []
    for episode in range(EPISODES):
        # t = np.linspace(.1, 8 * np.pi, SIZE_T)
        # t = np.sin(t) + np.log(t)
        # t += np.random.normal(0, 0.03, SIZE_T)
        t = get_csv_data()[SIZE_T:]
        y = np.sqrt(t) * 0
        arr = []
        arr_y = []
        for i in range(0, TRAJ, TRAJ_SIZE):
            arr.append(t[i: i + TRAJ_SIZE])
            if i != TRAJ - TRAJ_SIZE:
                arr_y.append([y[i]])
        # arr.append(t[0:TRAJ])
        # arr.append(t[0:TRAJ] * 0)
        # arr_y.append([y[0:TRAJ-1]])
        Xdata.append(arr)
        Ydata.append(arr_y)

    print(np.dstack(Ydata).shape)
    Xdata = np.stack(np.array(Xdata), axis=2)
    # Ydata = np.dstack(Ydata).transpose((1, 0, 2))
    Ydata = np.dstack(Ydata)
    print("TEST", Xdata.shape, Ydata.shape)
    # print(Xdata[0, :, :])
    # exit(0)
    return Xdata, Ydata


if __name__ == '__main__':
    pass


# import numpy as np
# import matplotlib.pyplot as plt


# def get_csv_data():
#     from pandas import read_csv
#     series = read_csv('LSTM/daily-min-temperatures.csv', header=0,
#                       index_col=0, parse_dates=True, squeeze=True)

#     return series.values


# def get_rand_train():
#     np.random.seed(0)

#     EPISODES = 20
#     TRAJ = 500
#     TRAJ_SIZE = 5
#     SIZE_T = 500

#     Xdata, Ydata = [], []
#     for episode in range(EPISODES):
#         # t = np.linspace(.1, 8 * np.pi, SIZE_T)
#         # t = np.sin(t) + np.log(t)
#         # t += np.random.normal(0, 0.03, SIZE_T)
#         t = get_csv_data()[:SIZE_T]
#         y = t
#         arr = []
#         arr_y = []
#         for i in range(0, TRAJ, TRAJ_SIZE):
#             arr.append(t[i: i + TRAJ_SIZE])
#             if i != TRAJ - TRAJ_SIZE:
#                 arr_y.append([y[i]])
#         Xdata.append(arr)
#         # Ydata.append(y[episode: episode + 19])
#         Ydata.append(arr_y)

#     print(np.dstack(Ydata).shape)
#     Xdata = np.stack(np.array(Xdata), axis=2)
#     # Ydata = np.dstack(Ydata).transpose((1, 0, 2))
#     Ydata = np.dstack(Ydata)
#     print("TRAIN", Xdata.shape, Ydata.shape)
#     return Xdata, Ydata


# def get_rand_test():
#     np.random.seed(1)

#     EPISODES = 1
#     TRAJ = 400
#     TRAJ_SIZE = 5
#     SIZE_T = 500

#     Xdata, Ydata = [], []
#     for episode in range(EPISODES):
#         t = np.linspace(.1, 8 * np.pi, SIZE_T)
#         t = np.sin(t) + np.log(t)
#         t += np.random.normal(0, 0.03, SIZE_T)
#         # t = get_csv_data()[SIZE_T:]
#         # t[20:] = 0
#         y = t
#         arr = []
#         arr_y = []
#         for i in range(0, TRAJ, TRAJ_SIZE):
#             arr.append(t[i: i + TRAJ_SIZE])
#             if i != TRAJ - TRAJ_SIZE:
#                 arr_y.append([y[i]])
#         Xdata.append(arr)
#         # Ydata.append(y[episode: episode + 19])
#         Ydata.append(arr_y)

#     print(np.dstack(Ydata).shape)
#     Xdata = np.stack(np.array(Xdata), axis=2)
#     # Ydata = np.dstack(Ydata).transpose((1, 0, 2))
#     Ydata = np.dstack(Ydata)
#     print("TEST", Xdata.shape, Ydata.shape)
#     print(Xdata[0, :, :])
#     exit(0)
#     return Xdata, Ydata


# if __name__ == '__main__':
#     pass
