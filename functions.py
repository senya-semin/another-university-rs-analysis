import typing

import numpy as np
from scipy.stats import linregress
from sklearn import metrics
from sklearn.cluster import KMeans


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


def clusters(hurst, stability):
    vector = np.array(list(zip(hurst, stability)))
    k = (
        np.argmax(
            [
                metrics.calinski_harabasz_score(vector, KMeans(n_clusters=n).fit(vector).labels_)
                for n in np.arange(2, 9)
            ]
        )
        + 1
    )
    return KMeans(n_clusters=k).fit(vector).labels_


def attractor(data, clusters, a, b):
    def henon(data, a, b):
        x, y = [], []
        for i in range(len(data)):
            x += [[1 - a * data[i][:-1][j] ** 2 + data[i][1:][j] for j in range(len(data[i]) - 1)]]
            y += [[b * data[i][:-1][j] for j in range(len(data[i]) - 1)]]
        return x, y

    classification_x = [[] for _ in range(len(np.unique(clusters)))]
    classification_y = [[] for _ in range(len(np.unique(clusters)))]
    year = [np.concatenate(data[month : month + 12]) for month in np.arange(data.size - 12)]
    medium_ = medium([year[i] for i in range(len(year))])
    attractorization_x, attractorization_y = henon(medium_, a=a, b=b)
    for i in range(len(attractorization_x)):
        classification_x[clusters[i]] += [attractorization_x[i]]
        classification_y[clusters[i]] += [attractorization_y[i]]
    return classification_x, classification_y


def mean(data, size):
    result = []
    for i in range(len(data)):
        if i + 1 < size and i != 0:
            result += [np.mean(data[0:i])]
        elif i != 0:
            result += [np.mean(data[i - size : i])]
        if i == 0:
            result += [data[i]]
    return result


def recurrence_plot(data: np.array) -> np.array:
    print(data)
    print(data.size)
    time = np.arange(data.size)
    epsilon = data.size * 1e-2
    step = 1
    return [
        [np.heaviside(epsilon - np.linalg.norm(data[i] - data[j]), 0) for i in time[step:]]
        for j in time[:-step]
    ]
