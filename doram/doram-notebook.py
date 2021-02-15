# %%

import datetime

import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn
from itertools import chain
import scipy.stats as stats


seaborn.set_theme(color_codes=True)
plot.rcParams["figure.figsize"] = (16, 9)


# %%

orders = pandas.read_excel(
    "order-book-trading.xlsx",
    sheet_name="Daily Order Book Trading",
    header=9,
    usecols=["Trade Date", "Number of Trades", "Value Traded"],
    thousands=",",
    skipfooter=2,
)
orders = orders.loc[
    (orders["Trade Date"] > datetime.datetime(1997, 12, 31)) & (orders["Trade Date"] < datetime.datetime(2021, 1, 1))
]
orders = orders.iloc[::-1].reset_index(drop=True)

orders


# %%


def calculate_ratio_diff(array: numpy.ndarray) -> numpy.ndarray:
    array = array.copy()
    array = array[:-1] / array[1:]
    numpy.insert(array, 0, 1)
    return array


# %%

orders["Value Traded Ratio Diff"] = numpy.insert(calculate_ratio_diff(orders["Value Traded"].to_numpy()), 0, 1)
orders.loc[orders["Value Traded Ratio Diff"] > 10.0, "Value Traded Ratio Diff"] = 1.0

seaborn.lineplot(x="Trade Date", y="Value Traded", data=orders)
axis = plot.twinx()
seaborn.lineplot(x="Trade Date", y="Value Traded Ratio Diff", data=orders, color="r", ax=axis)
plot.show()

# %%

orders["Number of Trades Ratio Diff"] = numpy.insert(calculate_ratio_diff(orders["Number of Trades"].to_numpy()), 0, 1)
orders.loc[orders["Number of Trades Ratio Diff"] > 10.0, "Number of Trades Ratio Diff"] = 1.0

seaborn.lineplot(x="Trade Date", y="Number of Trades", data=orders)
axis = plot.twinx()
seaborn.lineplot(x="Trade Date", y="Number of Trades Ratio Diff", data=orders, color="r", ax=axis)
plot.show()

# %%

orders.loc[:, "Stability"] = orders["Number of Trades Ratio Diff"] / orders["Value Traded Ratio Diff"]
diffs_ratio_mean = orders["Stability"].to_numpy().mean()

seaborn.lineplot(x="Trade Date", y="Stability", data=orders, estimator="median")
plot.axhline(diffs_ratio_mean, color="r")
plot.show()

# %%

diffs_ratio_mean

# %%

orders.loc[:, "Year"] = orders["Trade Date"].dt.year
orders.loc[:, "Month"] = orders["Trade Date"].dt.month
stability_by_months = orders.groupby(["Year", "Month"]).agg({"Stability": "mean"})

stability_by_months

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

lse = pandas.read_csv(
    "lse-100.csv",
    sep=",",
    header=0,
    usecols=[0, 4],
    names=["date", "close"],
)
lse = lse.iloc[::-1].reset_index(drop=True)

lse.loc[:, "year"] = lse["date"].str[6:8]
lse.loc[:, "month"] = lse["date"].str[:2]
lse_by_months = lse.groupby(["year", "month"])['close'].apply(list).to_list()

stability_mean_by_months = [numpy.mean(stability_by_months[i : i + 12]) for i in range(len(stability_by_months) - 12)]

lse_slopes_by_months = []
for i in range(len(lse_by_months) - 12):
    sample = numpy.array(list(chain.from_iterable(lse_by_months[i : i + 12])))
    step_range = numpy.arange(start=5, stop=sample.size // 2, step=1)
    results = numpy.array([(step, RS(sample, step)) for step in step_range])
    log_x = numpy.log(results[:, 0])
    log_y = numpy.log(results[:, 1])
    lse_slopes_by_months += [stats.linregress(log_x, log_y).slope]

# %%

# color = numpy.array([i for i, v in sorted(enumerate(lse_slopes_by_months), key=lambda iv: iv[1])])

seaborn.regplot(x=stability_mean_by_months, y=lse_slopes_by_months)
# seaborn.scatterplot(x=stability_mean_by_months, y=lse_slopes_by_months, hue=color)
plot.show()

# %%
