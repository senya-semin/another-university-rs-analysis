import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

import functions

markets = (
    "data/korea",
    "data/london",
    "data/taiwan",
)

figure = sp.make_subplots(rows=1, cols=len(markets))

for index, market in enumerate(markets):
    clients = pd.read_csv(f"{market}/clients.csv", header=0, parse_dates=["date"]).dropna()
    clients["stability"] = functions.stability(
        trades=clients["trades"].to_numpy(), value=clients["value"].to_numpy()
    )
    stability_by_month = (
        clients.groupby([clients["date"].dt.year, clients["date"].dt.month])["stability"]
        .std()
        .to_numpy()
    )
    stability_window_means = functions.window_means(stability_by_month)
    indexes = pd.read_csv(f"{market}/indexes.csv", header=0, parse_dates=["date"]).dropna()

    indexes_by_month = (
        indexes.groupby([clients["date"].dt.year, clients["date"].dt.month])["close"]
        .apply(np.array)
        .to_numpy()
    )
    indexes_window_slopes = functions.window_slopes(indexes_by_month)
    clusters_ = functions.clusters(indexes_window_slopes, stability_window_means)
    print(np.unique(clusters_))
    # clusters_ = clusters_[0]
    # print(len(clusters_))
    print(len(clusters_))
    print(len(indexes_by_month))
    #    x, y = functions.henon_heiles(indexes_by_month, 0.125, 1)

    x, y = functions.attractor(indexes_by_month, clusters_, a=1.4, b=0.3)
    for i in range(len(x)):
        figure.add_trace(
            go.Scatter(x=np.concatenate(x[i]), y=np.concatenate(y[i]), mode="markers"),
            row=1,
            col=index + 1,
        )

# figure.show()
# plot.show()
