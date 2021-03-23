from typing import Any, List, Tuple

import numpy as np
import scipy.stats as stats

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import seaborn as sns
import matplotlib.pyplot as plot

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
    def ratio_difference(array: np.array, first_value: Any = 1) -> np.array:
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
        slopes += [stats.linregress(log_x, log_y).slope]
    return np.array(slopes)


def medium(data):
    median = [np.median(data[i]) for i in range(len(data))]
    result = []
    for i in range(len(median)):
        result += [[n / median[i] for n in data[i]]]
    return result


def clusters(hurst, stability):
    vector = np.array(list(zip(hurst, stability)))
    nn = NearestNeighbors(n_neighbors=5).fit(vector)
    distances, idx = nn.kneighbors(vector)
    distances = np.sort(distances, axis=0)[:, 1]
#    sns.lineplot(x= range(len(distances)), y =distances)
#    plot.show()
#    eps = float(input("Пожалуйста введите расстояние между точками"))
    return DBSCAN(eps=0.01, min_samples= 5).fit(vector).labels_

def attractor_(data, clusters, a = 1.4, b = 0.3, step = 0):
    def henon(data, a, b):
        x, y = [], []
        for i in range(len(data)):
            x += [[1 - a * data[i][:-1][j] ** 2 + data[i][1:][j] for j in range(len(data[i]) - 1)]]
            y += [[b * data[i][:-1][j] for j in range(len(data[i]) - 1)]]
        return x, y
    
    def clear(xx, yy,a = a, b=b, step = step):
        x = np.copy(xx)
        y = np.copy(yy)
        for i in range(20):
            x_, y_ = [], []
            for j in range(len(x)):
                x_ += [[1 - a * xx[j][k] ** 2 + yy[j][k] for k in range(len(x[j]) - 1)]]
                y_ += [[b * xx[j][k] for k in range(len(x[j]) - 1)]]
            xx = x_
            yy= y_
        xx = [np.where(np.isfinite(xx[i]) == False) for i in range(len(xx))]
        yy = [np.where(np.isfinite(yy[i]) == False) for i in range(len(yy))]
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
            x += [[1 - a * attractorization_x[j][k] ** 2 + attractorization_y[j][k] for k in range(len(attractorization_x[j]) - 1)]]
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
        x,y = attractor_(data = data, clusters = clusters, step = i)
        for j in range(len(x)):
            m_x[j] += [np.mean(np.concatenate(x[j]))]
            m_y[j] += [np.var(np.concatenate(x[j]))]
    for i in range(len(m_x)):
    #    sns.lineplot(x = range(len(m_x[i])), y = m_x[i])
        sns.scatterplot(x = m_x[i], y = m_y[i])
    plot.show()


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


def clear(
    lx: List[List[List[float]]], ly: List[List[List[float]]], precision=3
) -> Tuple[Tuple[float, float]]:
    clusters = []
    for cx, cy in zip(lx, ly):
        years = []
        for yx, yy in zip(cx, cy):
            days = []
            for dx, dy in zip(yx, yy):
                dx = round(dx, precision)
                dy = round(dy, precision)
                days += [(dx, dy)]
            years += days
        clusters += [tuple(years)]
    clusters = tuple(clusters)

    pairs = []
    for cluster in clusters:
        for pair in cluster:
            if pair not in pairs:
                pairs += [pair]
    pairs = tuple(pairs)

    weights = {}
    for cluster in clusters:
        if cluster not in weights:
            weights[cluster] = {}
        for pair in pairs:
            if pair not in weights[cluster]:
                weights[cluster][pair] = 0
            if pair in cluster:
                weights[cluster][pair] += 1
    for pair in pairs:
        maximum = 0
        for cluster in clusters:
            maximum = max(maximum, weights[cluster][pair])
        for cluster in clusters:
            if weights[cluster][pair] < maximum:
                del weights[cluster][pair]

    result = []
    for value in weights.values():
        values = []
        for key in value.keys():
            values += [key]
        result += [tuple(values)]
    result = tuple(result)
    return result

def henon_heiles(data, max_energy, step):
    def function(x, y, axis):
        if axis == 0:
            value = -x - 2*x*y
        elif axis == 1:
            value = -y -x**2 + y**2
        return value

    def Runge_Kutta(x, y, step, axis):
        for i in np.arange(0, 8, step):
            k_1 = function(x[i], y[i], axis = axis)
            k_2 = function(x[i] + step/2, y[i] + step * k_1 / 2, axis = axis)
            k_3 = function(x[i] + step/2, y[i] + step * k_2 / 2, axis = axis)
            k_4 = function(x[i] + step, y[i] + step*k_3, axis = axis)
            if axis == 1:
                y[i+step] = y[i] + 1/6 * step*(k_1 + 2*k_2 + 2*k_3 + k_4)
            elif axis == 0:
                x[i+step] = x[i] + 1/6 * step*(k_1 + 2*k_2 + 2*k_3 + k_4)
        if axis == 0:
            return x
        elif axis == 1:
            return y

    def energy(y):
        return [1/2*y[p]**2*(1-2/3*y[p]) for p in range(len(y))]
    
    def foundation(y_original, y_integration):
        result_x, result_y = [], []
        for i in range(len(y_integration)-1):
            if y_integration[i] < 0 and y_integration[i+1] > 0 or y_integration[i] > 0 and y_integration[i+1] < 0:
                result_y += [y_integration[i] + y_integration[i+1]/2] 
                result_x += [y_original[i]+y_original[i+1]/2] 
        return result_x, result_y

    year = [np.concatenate(data[month : month + 12]) for month in np.arange(data.size - 12)]
    mediam = medium([year[i] for i in range(len(year))])
    answer_x, answer_y, l = [], [], 0
    for i in range(len(mediam)):
        x, y = mediam[i][:-1], mediam[i][1:]
        print(len(y))
        energy = energy(y)
        #print(energy)
        for j in energy:
            l += 1
            if j <= max_energy:
                answer_x += foundation(y[l:], Runge_Kutta(x[l:], y[l:], step, 1))[0]
                answer_y += foundation(y[l:], Runge_Kutta(x[l:], y[l:], step, 1))[1]
                print(answer_x)
        l = 0
    return answer_x, answer_y
