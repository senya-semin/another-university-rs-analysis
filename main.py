import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import matplotlib.pyplot as plt

import functions

import seaborn as sns
import matplotlib.pyplot as plt

markets = (
    "data/korea",
    "data/london",
    "data/iwan",
)
h = 0
sns.set(style = "darkgrid")

figure = sp.make_subplots(rows=1, cols=len(markets), specs=[[dict(type="scene")] * len(markets)])

for index, market in enumerate(markets):
    h += 1
    clients = pd.read_csv(f"{market}/clients.csv", header=0, parse_dates=["date"]).dropna()
    clients["stability"] = functions.stability(
        trades=clients["trades"].to_numpy(),
        value=clients["value"].to_numpy(),
    )
    indexes = pd.read_csv(f"{market}/indexes.csv", header=0, parse_dates=["date"]).dropna()


    stability_by_month = (
        clients.groupby([clients["date"].dt.year, clients["date"].dt.month])["stability"]
        .var()
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
    clusters = functions.clusters(hurst, stability, lyapunov, h = h)

    sns.lineplot(data = indexes, x = "date", y = "close")
    plt.xlabel("Период")
    plt.ylabel("Значение индекса")
    plt.show()
    plt.clf()

    sns.lineplot(data = clients, x = "date", y = "trades")
    plt.xlabel("Период")
    plt.ylabel("Количество совершенных сделок")
    plt.show()
    plt.clf()

    sns.lineplot(x = stability, y = hurst, hue = range(len(hurst)))
    plt.xlabel("Период")
    plt.ylabel("Коэффициента Херста")
    plt.show()
    plt.clf()


    data = {"Показатель Херста": hurst, 
    "Стабильность": stability, 
    "Кластер": clusters, 
    "Период": range(len(stability)), 
    "Показатель Ляпунова": lyapunov,
    }
    df = pd.DataFrame(data=data)

    fig = px.scatter(df, x = "Стабильность", y = "Показатель Херста", 
    color = "Период", color_continuous_scale = "Bluered")
    fig.show()

    fig = px.scatter(df, x = "Стабильность", y = "Показатель Херста", 
    color = "Кластер", color_continuous_scale = "Portland", template = "plotly_dark")
    fig.show()

    fig = px.scatter_3d(df, x = "Стабильность", y = "Показатель Херста", z="Показатель Ляпунова", 
    color="Период", color_continuous_scale = "Bluered")
    fig.show()

    fig = px.scatter_3d(df, x = "Стабильность", y = "Показатель Херста", z="Показатель Ляпунова", 
    color="Кластер", color_continuous_scale = "Portland", template = "plotly_dark")
    fig.show()

    if h == 1:
        print("Корейский показатель Ляпунова", functions.lyapunov(
            functions.recurrence_plot(
                functions.return_(
                    np.concatenate(indexes_by_month)))))
        print("Корейский среднее показателя Ляпунова", np.mean(lyapunov))
        print("Корейская диспекрсия Ляпунова ", np.std(lyapunov))
        print("Корейский размах Ляпунова ", np.max(lyapunov)-np.min(lyapunov))
        print("Корейский коэффициет H", functions.h(np.concatenate(indexes_by_month)))
        print("Корейский среднее коэффициента Херста ", np.mean(hurst))
        print("Корейская диспекрсия H ", np.std(hurst))
        print("Корейский размах H ", np.max(hurst)-np.min(hurst))
    elif h == 2:
        print("Лондонский показатель Ляпунова", functions.lyapunov(
            functions.recurrence_plot(
                functions.return_(
                    np.concatenate(indexes_by_month)))))
        print("Лондонский среднее показателя Ляпунова", np.mean(lyapunov))
        print("Лондонский диспекрсия Ляпунова ", np.std(lyapunov))
        print("Лондонский размах Ляпунова ", np.max(lyapunov)-np.min(lyapunov))
        print("Лондонский коэффициет H", functions.h(np.concatenate(indexes_by_month)))
        print("Лондонский среднее коэффициента Херста ", np.mean(hurst))
        print("Лондонский диспекрсия H ", np.std(hurst))
        print("Лондонский размах H ", np.max(hurst)-np.min(hurst))
    elif h == 3:
        print("Тайваньский показатель Ляпунова", functions.lyapunov(
            functions.recurrence_plot(
                functions.return_(
                    np.concatenate(indexes_by_month)))))
        print("Тайваньский среднее показателя Ляпунова", np.mean(lyapunov))
        print("Тайваньский диспекрсия Ляпунова ", np.std(lyapunov))
        print("Тайваньский размах Ляпунова ", np.max(lyapunov)-np.min(lyapunov))
        print("Тайваньский коэффициет H", functions.h(np.concatenate(indexes_by_month)))
        print("Тайваньский среднее коэффициента Херста ", np.mean(hurst))
        print("Тайваньский диспекрсия H ", np.std(hurst))
        print("Тайваньский размах H ", np.max(hurst)-np.min(hurst))