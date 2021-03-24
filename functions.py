import typing

import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def rs(array: np.array, step: int) -> float:
    def compose(array: np.array, step: int) -> np.array:
        segments = array.size // step
        return array[: segments * step].reshape(step, segments)

    log_growth = np.diff(np.log(array))
    composed = compose(log_growth, step)
    mean = composed.mean(axis=0)
    mean_reshaped = np.tile(mean.reshape(mean.size, 1), composed.shape[0]).T
    cumsum = composed.cumsum(axis=0) - mean_reshaped.cumsum(axis=0)
    range_ = np.amax(cumsum, axis=0) - np.amin(cumsum, axis=0)
    std = composed.std(axis=0)
    return (range_ / std).mean()


def window_means(array: np.array, length: int = 12) -> np.array:
    return np.array([np.mean(array[i : i + length]) for i in np.arange(array.size - length)])


def stability(trades: np.array, value: np.array) -> np.array:
    def ratio_difference(array: np.array, first_value: typing.Any = 1) -> np.array:
        return np.r_[(first_value,), array[1:] / array[:-1]]

    result = ratio_difference(trades) / ratio_difference(value)
    return np.cumprod(result)


def window_slopes(array: np.array) -> np.array:
    step_range = np.arange(start=5, stop=array.size // 2, step=1)
    slopes = []
    for month in np.arange(array.size - 12):
        year = np.concatenate(array[month : month + 12])
        results = np.array([(step, rs(year, step)) for step in step_range])
        log_x = np.log(results[:, 0])
        log_y = np.log(results[:, 1])
        slopes += [linregress(log_x, log_y).slope]
    return np.array(slopes)


def medium(data):
    median = [np.median(data[i]) for i in range(len(data))]
    result = []
    for i in range(len(median)):
        result += [[n / median[i] for n in data[i]]]
    return result


def clusters(hurst, stability, lyapunov):
    vector = np.array(list(zip(hurst, stability, lyapunov)))
    nn = NearestNeighbors(n_neighbors=5).fit(vector)
    distances, idx = nn.kneighbors(vector)
    distances = np.sort(distances, axis=0)[:, 1]
    sns.lineplot(x=range(len(distances)), y=distances)
    plot.show()
    eps = float(input("Пожалуйста введите расстояние между точками"))
    return DBSCAN(eps=eps, min_samples=5).fit(vector).labels_


def attractor(data, clusters, a=1.4, b=0.3, step=0):
    def henon(data, a, b):
        x, y = [], []
        for i in range(len(data)):
            x += [[1 - a * data[i][:-1][j] ** 2 + data[i][1:][j] for j in range(len(data[i]) - 1)]]
            y += [[b * data[i][:-1][j] for j in range(len(data[i]) - 1)]]
        return x, y

    def clear(xx, yy, a=a, b=b, step=step):
        x = np.copy(xx)
        y = np.copy(yy)
        for i in range(20):
            x_, y_ = [], []
            for j in range(len(x)):
                x_ += [[1 - a * xx[j][k] ** 2 + yy[j][k] for k in range(len(x[j]) - 1)]]
                y_ += [[b * xx[j][k] for k in range(len(x[j]) - 1)]]
            xx = x_
            yy = y_
        xx = [np.where(not np.isfinite(xx[i])) for i in range(len(xx))]
        yy = [np.where(not np.isfinite(yy[i])) for i in range(len(yy))]
        x = [np.delete(x[i], j) for i in range(len(x)) for j in xx[i]]
        y = [np.delete(y[i], j) for i in range(len(x)) for j in yy[i]]
        return x, y

    classification_x = [[] for _ in range(len(np.unique(clusters)) - 1)]
    classification_y = [[] for _ in range(len(np.unique(clusters)) - 1)]
    year = [np.concatenate(data[month : month + 12]) for month in np.arange(data.size - 12)]
    medium_ = medium([year[i] for i in range(len(year))])
    attractorization_x, attractorization_y = henon(medium_, a=a, b=b)
    attractorization_x, attractorization_y = clear(attractorization_x, attractorization_y)
    for i in range(step):
        x, y = [], []
        for j in range(len(attractorization_x)):
            x += [
                [
                    1 - a * attractorization_x[j][k] ** 2 + attractorization_y[j][k]
                    for k in range(len(attractorization_x[j]) - 1)
                ]
            ]
            y += [[b * attractorization_x[j][k] for k in range(len(attractorization_x[j]) - 1)]]
        attractorization_x = x
        attractorization_y = y
    for i in range(len(attractorization_x)):
        if clusters[i] != -1:
            classification_x[clusters[i]] += [attractorization_x[i]]
            classification_y[clusters[i]] += [attractorization_y[i]]
    return classification_x, classification_y


def meaning(data, clusters, step):
    m_y = [[] for _ in range(len(np.unique(clusters)) - 1)]
    m_x = [[] for _ in range(len(np.unique(clusters)) - 1)]
    for i in range(step):
        x, y = attractor(data=data, clusters=clusters, step=i)
        for j in range(len(x)):
            m_x[j] += [np.mean(np.concatenate(x[j]))]
            m_y[j] += [np.var(np.concatenate(x[j]))]
    return m_x, m_y


def recurrence_plot(data: np.array) -> np.array:
    time = np.arange(data.size)
    epsilon = data.size * 1e-2
    step = 1
    return np.array(
        [
            [np.heaviside(epsilon - np.linalg.norm(data[i] - data[j]), 0) for i in time[step:]]
            for j in time[:-step]
        ]
    )


def lyapunov(matrix: np.array) -> float:
    max_length = 0
    size = matrix.shape[0]
    for diagonal in range(1, size):
        diagonal_lengths = []
        length = 0
        for element in range(size - diagonal):
            i = element
            j = matrix.shape[0] - diagonal - element - 1
            if matrix[i][j]:
                length += 1
            else:
                diagonal_lengths.append(length)
                length = 0
        diagonal_lengths.append(length)
        max_length = max(max(diagonal_lengths), max_length)
    return 1 / max_length


def normalize(data: np.array) -> np.array:
    mean, max_, min_ = data.mean(), data.max(), data.min()
    function = np.vectorize(lambda x: (x - mean) / (max_ - min_))
    return function(data)


def return_(data: np.array) -> np.array:
    return np.diff(np.log(data))


def chaos(data: np.array) -> np.array:
    years = [np.concatenate(data[month : month + 12]) for month in np.arange(data.size - 12)]
    return np.array([lyapunov(recurrence_plot(year)) for year in years])
