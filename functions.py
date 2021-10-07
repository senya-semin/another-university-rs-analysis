import typing

import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


def rs(array: np.array, step: int) -> float:
    def compose(array: np.ndarray, step: int) -> np.ndarray:
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
        step_range = np.arange(start=5, stop=year.size // 2, step=1)
        results = np.array([(step, rs(year, step)) for step in step_range])
        log_x = np.log(results[:, 0])
        log_y = np.log(results[:, 1])
        slopes += [linregress(log_x, log_y).slope]
    return np.array(slopes)

def h(data):
    step_range = np.arange(start=5, stop= len(data) // 2, step=1)
    results = np.array([(step, rs(np.array(data), step)) for step in step_range])
    log_x = np.log(results[:, 0])
    log_y = np.log(results[:, 1])
    return linregress(log_x, log_y).slope

def clusters(hurst, stability, lyapunov, h):
    vector = np.array(list(zip(stability, hurst,  lyapunov)))
    vector = TSNE(n_components=2).fit_transform(vector)
    nn = NearestNeighbors(n_neighbors=5).fit(vector)
    distances, idx = nn.kneighbors(vector)
    distances = np.sort(distances, axis=0)[:, 1]
    if h == 1:
        eps = 1.25
    elif h == 2:
        eps = 0.8
    elif h == 3:
        eps = 1.25
    elif h == 0:
        sns.lineplot(x=range(len(distances)), y=distances)
        plot.xlabel("количество")
        plot.ylabel("дистанция")
        plot.show()
        plot.clf()
        eps = float(input("введите дистанцию: "))
    print("eps = ", eps)
    return DBSCAN(eps= eps, min_samples=4).fit(vector).labels_


def recurrence_plot(data: np.array) -> np.array:
    time = np.arange(data.size)
    epsilon = data.size * 1e-4
    step = 1
    return np.array(np.rot90(
        [
            [np.heaviside(epsilon - np.linalg.norm(data[i] - data[j]), 0) for i in time[step:]]
            for j in time[:-step]
        ]
    ))

def lyapunov(matrix: np.array) -> float:
    max_length = 0
    size = matrix.shape[0]
    for diagonal in range(1, size ):
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
#    print(max_length)
    return 1 / max_length

def normalize(data: np.array) -> np.array:
    mean, max_, min_ = data.mean(), data.max(), data.min()
    function = np.vectorize(lambda x: (x - mean) / (max_ - min_))
    return function(data)

def return_(data: np.array) -> np.array:
    return np.cumsum(np.diff(np.log(data)))

def chaos(data: np.array) -> np.array:
    years = [np.concatenate(data[month : month + 12]) for month in np.arange(data.size - 12)]
    return np.array([lyapunov(recurrence_plot(return_(year))) for year in years])
