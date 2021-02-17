import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn

import functions

seaborn.set_theme(color_codes=True)
plot.rcParams["figure.figsize"] = (16, 9)

clients = pandas.read_csv("data/london/clients.csv", header=0, parse_dates=["date"])
stability = functions.stability(
    trades=clients["trades"].to_numpy(), value=clients["value"].to_numpy()
)
clients["stability"] = stability
stability_by_month = (
    clients.groupby([clients["date"].dt.year, clients["date"].dt.month])["stability"]
    .mean()
    .to_numpy()
)
stability_window_means = functions.window_means(stability_by_month)

indexes = pandas.read_csv("data/london/indexes.csv", header=0, parse_dates=["date"])
indexes_by_month = (
    indexes.groupby([clients["date"].dt.year, clients["date"].dt.month])["close"]
    .apply(numpy.array)
    .to_numpy()
)
indexes_window_slopes = functions.window_slopes(indexes_by_month)

seaborn.regplot(
    x="stability",
    y="slopes",
    data={"stability": stability_window_means, "slopes": indexes_window_slopes},
)
plot.show()
