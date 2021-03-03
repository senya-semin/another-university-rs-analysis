from typing import Any

import numpy
import scipy.stats as stats
from sklearn import metrics
from sklearn.cluster import KMeans


def rs(array: numpy.array, step: int) -> float:
    def compose(array: numpy.array, step: int) -> numpy.array:
        segments = array.size // step
        return array[: segments * step].reshape(step, segments)

    log_growth = numpy.diff(numpy.log(array))
    composed = compose(log_growth, step)
    mean = composed.mean(axis=0)
    mean_reshaped = numpy.tile(mean.reshape(mean.size, 1), composed.shape[0]).T
    cumsum = composed.cumsum(axis=0) - mean_reshaped.cumsum(axis=0)
    range_ = numpy.amax(cumsum, axis=0) - numpy.amin(cumsum, axis=0)
    std = composed.std(axis=0)
    return (range_ / std).mean()


def window_means(array: numpy.array, length: int = 12) -> numpy.array:
    return numpy.array(
        [numpy.mean(array[i : i + length]) for i in numpy.arange(array.size - length)]
    )


def stability(trades: numpy.array, value: numpy.array) -> numpy.array:
    def ratio_difference(array: numpy.array, first_value: Any = 1) -> numpy.array:
        return numpy.r_[[first_value], array[1:] / array[:-1]]

    result = ratio_difference(trades) / ratio_difference(value)
    return numpy.cumprod(result)


def window_slopes(array: numpy.array) -> numpy.array:
    step_range = numpy.arange(start=5, stop=array.size // 2, step=1)
    slopes = []
    for month in numpy.arange(array.size - 12):
        year = numpy.concatenate(array[month : month + 12])
        results = numpy.array([(step, rs(year, step)) for step in step_range])
        log_x = numpy.log(results[:, 0])
        log_y = numpy.log(results[:, 1])
        slopes += [stats.linregress(log_x, log_y).slope]
    return numpy.array(slopes)


def medium(data):
    print(len(data))
    median = [numpy.median(data[i]) for i in range(len(data))]
    result = []
    for i in range(len(median)):
        result += [[n / median[i] for n in data[i]]]
    return result


def clusters(hurst, stability):
    vektor = numpy.array(list(zip(hurst, stability)))
    k = (
        numpy.argmax(
            [
                metrics.calinski_harabasz_score(vektor, KMeans(n_clusters=n).fit(vektor).labels_)
                for n in numpy.arange(2, 9)
            ]
        )
        + 1
    )
    print(k)
    return KMeans(n_clusters=k).fit(vektor).labels_


def attractor(data, clusters, a, b):
    def henon(data, a, b):
        x, y = [], []
        for i in range(len(data)):
            x += [[1 - a * data[i][:-1][j] ** 2 + data[i][1:][j] for j in range(len(data[i]) - 1)]]
            y += [[b * data[i][:-1][j] for j in range(len(data[i]) - 1)]]
        return x, y

    classification_x = [[] for unique in range(len(numpy.unique(clusters)))]
    classification_y = [[] for unique in range(len(numpy.unique(clusters)))]
    year = [numpy.concatenate(data[month : month + 12]) for month in numpy.arange(data.size - 12)]
    mediam = medium([year[i] for i in range(len(year))])
    attractorization_x, attractorization_y = henon(mediam, a=a, b=b)
    for i in range(len(attractorization_x)):
        classification_x[clusters[i]] += [attractorization_x[i]]
        classification_y[clusters[i]] += [attractorization_y[i]]
    return classification_x, classification_y


def mean(data, size):
    result = []
    for i in range(len(data)):
        if i + 1 < size and i != 0:
            result += [numpy.mean(data[0:i])]
        elif i != 0:
            result += [numpy.mean(data[i - size : i])]
        if i == 0:
            result += [data[i]]
    return result


def Henon_Heiles(data, max_energy, step):
    def function(x, y, axis):
        if axis == 0:
            value = -x - 2 * x * y
        elif axis == 1:
            value = -y - x ** 2 + y ** 2
        return value

    def Runge_Kutta(x, y, step, axis):
        for i in numpy.arange(0, 8, step):
            k_1 = function(x[i], y[i], axis=axis)
            k_2 = function(x[i] + step / 2, y[i] + step * k_1 / 2, axis=axis)
            k_3 = function(x[i] + step / 2, y[i] + step * k_2 / 2, axis=axis)
            k_4 = function(x[i] + step, y[i] + step * k_3, axis=axis)
            if axis == 1:
                y[i + step] = y[i] + 1 / 6 * step * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            elif axis == 0:
                x[i + step] = x[i] + 1 / 6 * step * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        if axis == 0:
            return x
        elif axis == 1:
            return y

    def Hamilton(x, y):
        return [
            1 / 2 * (function(x[n], y[n], axis=0) + function(x[n], y[n], axis=1))
            + 1 / 2 * (x[n] ** 2 + y[n] ** 2)
            + x[n] ** 2 * y
            - y[n] ** 3 / 3
            for n in range(len(x))
        ]

    def energy(y):
        return [1 / 2 * y ** 2 * (1 - 2 / 3 * y)]

    def foundation(y_original, y_integration):
        result_x, result_y = [], []
        for i in range(len(y_integration) - 1):
            if (
                y_integration[i] < 0
                and y_integration[i + 1] > 0
                or y_integration[i] > 0
                and y_integration[i + 1] < 0
            ):
                result_y += [y_integration[i] + y_integration[i + 1] / 2]
                result_x += [y_original[i] + y_original[i + 1] / 2]
        return result_x, result_y

    year = [numpy.concatenate(data[month : month + 12]) for month in numpy.arange(data.size - 12)]
    mediam = medium([year[i] for i in range(len(year))])
    answer_x, answer_y, length = [], [], 0
    for i in range(len(mediam)):
        x, y = mediam[i][:-1], mediam[i][1:]
        energy = [energy(y[i]) for i in range(len(y))]
        for j in numpy.concatenate(energy):
            length += 1
            if j <= max_energy:
                answer_x += foundation(y[length:], Runge_Kutta(x[length:], y[length:], step, 1))[0]
                answer_y += foundation(y[length:], Runge_Kutta(x[length:], y[length:], step, 1))[1]
                break
        length = 0
    return answer_x, answer_y
