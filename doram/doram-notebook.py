# %%

import datetime

import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn

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
    (orders["Trade Date"] > datetime.datetime(1997, 12, 31))
    & (orders["Trade Date"] < datetime.datetime(2021, 1, 1))
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

orders["Value Traded Ratio Diff"] = numpy.insert(
    calculate_ratio_diff(orders["Value Traded"].to_numpy()), 0, 1
)
orders.loc[orders["Value Traded Ratio Diff"] > 10.0, "Value Traded Ratio Diff"] = 1.0

seaborn.lineplot(x="Trade Date", y="Value Traded", data=orders)
axis = plot.twinx()
seaborn.lineplot(x="Trade Date", y="Value Traded Ratio Diff", data=orders, color="r", ax=axis)
plot.show()

# %%

orders["Number of Trades Ratio Diff"] = numpy.insert(
    calculate_ratio_diff(orders["Number of Trades"].to_numpy()), 0, 1
)
orders.loc[orders["Number of Trades Ratio Diff"] > 10.0, "Number of Trades Ratio Diff"] = 1.0

seaborn.lineplot(x="Trade Date", y="Number of Trades", data=orders)
axis = plot.twinx()
seaborn.lineplot(x="Trade Date", y="Number of Trades Ratio Diff", data=orders, color="r", ax=axis)
plot.show()

# %%

orders.loc[:, "Stability"] = (
    orders["Number of Trades Ratio Diff"] / orders["Value Traded Ratio Diff"]
)
diffs_ratio_mean = orders["Stability"].to_numpy().mean()

seaborn.lineplot(x="Trade Date", y="Stability", data=orders, estimator="median")
plot.axhline(diffs_ratio_mean, color="r")
plot.show()

# %%

diffs_ratio_mean

# %%

orders.loc[:, "Year"] = orders["Trade Date"].dt.year
orders.loc[:, "Month"] = orders["Trade Date"].dt.month
stability_mean_by_months = orders.groupby(["Year", "Month"]).agg({"Stability": "mean"})

stability_mean_by_months
