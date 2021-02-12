# %%

import matplotlib.pyplot as plot
import numpy
import scipy.stats as stats

# %%


def RS(array: numpy.ndarray, step: int) -> float:
    def compose(array: numpy.ndarray, step: int) -> numpy.ndarray:
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


# %%

sample = numpy.genfromtxt("data.csv", delimiter=";", skip_header=1, usecols=7)

plot.plot(sample)
plot.show()

# %%

step_range = numpy.arange(start=5, stop=sample.size // 2, step=1)
results = numpy.array([(step, RS(sample, step)) for step in step_range])

log_x = numpy.log(results[:, 0])
log_y = numpy.log(results[:, 1])

plot.plot(log_x, log_y)
plot.show()

# %%

result = stats.linregress(log_x, log_y)

f"slope={result.slope}, rvalue={result.rvalue}"
