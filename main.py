import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn

import functions

seaborn.set_theme(color_codes=True)
plot.rcParams["figure.figsize"] = (16, 9)

markets = (
    "data/korea",
    "data/london",
)

_, axes = plot.subplots(1, len(markets))

for index, market in enumerate(markets):
    clients = pandas.read_csv(f"{market}/clients.csv", header=0, parse_dates=["date"]).dropna()
    clients["stability"] = functions.stability(
        trades=clients["trades"].to_numpy(), value=clients["value"].to_numpy()
    )
    stability_by_month = (
        clients.groupby([clients["date"].dt.year, clients["date"].dt.month])["stability"]
        .mean()
        .to_numpy()
    )
    stability_window_means = functions.window_means(stability_by_month)
    indexes = pandas.read_csv(f"{market}/indexes.csv", header=0, parse_dates=["date"]).dropna()
    indexes_by_month = (
        indexes.groupby([clients["date"].dt.year, clients["date"].dt.month])["close"]
        .apply(numpy.array)
        .to_numpy()
    )
    indexes_window_slopes = functions.window_slopes(indexes_by_month)

    axis = axes[index]
    seaborn.scatterplot(
        x=stability_window_means,
        y=indexes_window_slopes,
        c=numpy.arange(len(stability_window_means)),
        ax=axis,
    )
    seaborn.regplot(
        x=stability_window_means,
        y=indexes_window_slopes,
        order=3,
        scatter=False,
        ci=0,
        ax=axis,
    )
    axis.set_xlabel("stability")
    axis.set_ylabel("slopes")
    axis.set_title(market)

plot.show()
