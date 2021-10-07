import pathlib
import re
import typing

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
import functions
import plotly.express as px

sns.set(style = "darkgrid")

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

months_translation = {
    "January": "Январь",
    "February": "Февраль",
    "March": "Март",
    "April": "Апрель",
    "May": "Май",
    "June": "Июнь",
    "July": "Июль",
    "August": "Август",
    "September": "Сентябрь",
    "October": "Октябрь",
    "November": "Ноябрь",
    "December": "Декабрь",
}
users: typing.Dict[int, typing.Dict[str, str]] = {}
for file_ in pathlib.Path("imoex/clients").iterdir():
    sheets = pandas.read_excel(file_, sheet_name=None, header=None)
    year = int(re.findall(r"clients\-(\d{4})", str(file_))[0])
    for name, sheet in sheets.items():
        if year < 2016:
            month, year = re.findall(r"([А-Я][а-я]{2,7})(\d{4})", name)[0]
            year = int(year)
        else:
            month = months_translation[name]
        if year not in users:
            users[year] = {}
        users[year][month] = sheet[5][39 if year > 2011 else 33]

months_order = (
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
)
users_by_years = list(map(int, users.keys()))
years_range = range(min(users_by_years), max(users_by_years) + 1)
users_by_months = []
for year in years_range:
    users_by_months += [users[year][month] for month in months_order[5 if year == 2007 else 0 :]]

imoex = pandas.read_csv(
    "imoex.csv",
    sep=";",
    header=0,
    usecols=[2, 7],
    names=["date", "close"],
    dtype={"date": str, "close": float},
)

imoex.loc[:, "year-month"] = imoex["date"].str[:-2]
imoex_by_months = imoex.groupby("year-month")["close"].apply(list).to_numpy()

hurst = functions.window_slopes(imoex_by_months)
stability = functions.window_means(numpy.array(users_by_months))
lyapunov = functions.chaos(imoex_by_months)
clusters = functions.clusters(hurst, stability, lyapunov, 0)

sns.lineplot(data = imoex, x = range(len(imoex["close"])), y = "close")
plt.xlabel("Период")
plt.ylabel("Значение индекса")
plt.show()
plt.clf()

sns.lineplot(x = range(len(users_by_months)), y = users_by_months)
plt.xlabel("Период")
plt.ylabel("Количество пользователей")
plt.show()
plt.clf()

sns.lineplot(x = stability, y = hurst, hue = range(len(hurst)))
plt.xlabel("Период")
plt.ylabel("Коэффициента Херста")
plt.show()
plt.clf()


data = {"Показатель Херста": hurst, 
    "Количество пользователей": stability, 
    "Кластер": clusters, 
    "Период": range(len(stability)), 
    "Показатель Ляпунова": lyapunov,
    }
df = pandas.DataFrame(data=data)

fig = px.scatter(df, x = "Количество пользователей", y = "Показатель Херста", 
color = "Период", color_continuous_scale = "Bluered")
fig.show()

fig = px.scatter(df, x = "Количество пользователей", y = "Показатель Херста", 
color = "Кластер", color_continuous_scale = "Portland", template = "plotly_dark")
fig.show()

fig = px.scatter_3d(df, x = "Количество пользователей", y = "Показатель Херста", z="Показатель Ляпунова", 
color="Период", color_continuous_scale = "Bluered")
fig.show()

fig = px.scatter_3d(df, x = "Количество пользователей", y = "Показатель Херста", z="Показатель Ляпунова", 
color="Кластер", color_continuous_scale = "Portland", template = "plotly_dark")
fig.show()

print("Корейский показатель Ляпунова", functions.lyapunov(
    functions.recurrence_plot(
        functions.return_(
            numpy.concatenate(imoex_by_months)))))
print("Корейский среднее показателя Ляпунова", numpy.mean(lyapunov))
print("Корейская диспекрсия Ляпунова ", numpy.std(lyapunov))
print("Корейский размах Ляпунова ", numpy.max(lyapunov)-numpy.min(lyapunov))
print("Корейский коэффициет H", functions.h(numpy.concatenate(imoex_by_months)))
print("Корейский среднее коэффициента Херста ", numpy.mean(hurst))
print("Корейская диспекрсия H ", numpy.std(hurst))
print("Корейский размах H ", numpy.max(hurst)-numpy.min(hurst))
