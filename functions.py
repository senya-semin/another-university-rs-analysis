import numpy
import scipy.stats as stats


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
    def ratio_difference(array: numpy.array, first_value: any = 1) -> numpy.array:
        return numpy.r_[[first_value], array[1:] / array[:-1]]

    return ratio_difference(trades) / ratio_difference(value)


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
