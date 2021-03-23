import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
import matplotlib.pyplot as plot
import plotly.express as px

import functions

markets = (
#    "G:\Games\data\korea",
#   "G:\Games\data\london",
    "G:\Games\data\iwan",
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
    #clusters_ = clusters_[0]
    #print(len(clusters_))
    print(len(clusters_))
    print(len(indexes_by_month))
#    x, y = functions.henon_heiles(indexes_by_month, 0.125, 1)

#    x,y = functions.attractor_(indexes_by_month, clusters_, step = 4)
    functions.meaning(indexes_by_month, clusters_, step = 20)
#    for cluster in functions.clear(*attractor, 2):
#        x = [i[0] for i in cluster]
#        print("среднее:", np.mean(x))
#        print(len(x))
#        y = [i[1] for i in cluster]
#        figure.add_trace(go.Scatter(x=x, y=y, mode="markers"), row=1, col=index + 1)
#    for i in range(len(x)):
#        print(np.mean(np.concatenate(x[i])))
#        figure.add_trace(go.Scatter(x= np.concatenate(x[i]), y=np.concatenate(y[i]), mode="markers"), row=1, col=index + 1)
    #figure.add_trace(go.Scatter(x= np.concatenate(x[1]), y=np.concatenate(y[1]), mode="markers"), row=1, col=index + 1)
    #sns.scatterplot(x= stability_window_means, y = indexes_window_slopes, palette = clusters_)
    #figure = px.scatter(x= stability_window_means, y = indexes_window_slopes, color=clusters_)
    #figure.add_trace(go.Scatter(x= stability_window_means, y = indexes_window_slopes, mode="markers"), row = 1, col = index +1)
#    figure = px.scatter(x= stability_window_means, y = indexes_window_slopes, color = clusters_)

#figure.show()
#plot.show()
