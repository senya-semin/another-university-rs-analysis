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

figure = sp.make_subplots(rows=1, cols=len(markets), specs=[[dict(type="scene")] * len(markets)])

for index, market in enumerate(markets):
    clients = pd.read_csv(f"{market}/clients.csv", header=0, parse_dates=["date"]).dropna()
    clients["stability"] = functions.stability(
        trades=clients["trades"].to_numpy(),
        value=clients["value"].to_numpy(),
    )
    indexes = pd.read_csv(f"{market}/indexes.csv", header=0, parse_dates=["date"]).dropna()

    stability_by_month = (
        clients.groupby([clients["date"].dt.year, clients["date"].dt.month])["stability"]
        .std()
        .to_numpy()
    )
    indexes_by_month = (
        indexes.groupby([clients["date"].dt.year, clients["date"].dt.month])["close"]
        .apply(np.array)
        .to_numpy()
    )

    hurst = functions.window_slopes(indexes_by_month)
    stability = functions.window_means(stability_by_month)
    lyapunov = functions.chaos(indexes_by_month)
    clusters = functions.clusters(hurst, stability, lyapunov)

    figure.add_trace(
        go.Scatter3d(x=stability, y=hurst, z=lyapunov, mode="markers"),
        row=1,
        col=index + 1,
    )

figure.show()
