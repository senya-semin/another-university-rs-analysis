# %%

import pathlib
import re
import typing
from itertools import chain

import matplotlib.pyplot as plot
import numpy
import pandas
import scipy.stats as stats
import seaborn

seaborn.set_theme(color_codes=True)
plot.rcParams["figure.figsize"] = (16, 9)

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
for file_ in pathlib.Path("clients").iterdir():
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
    "data.csv",
    sep=";",
    header=0,
    usecols=[2, 7],
    names=["date", "close"],
    dtype={"date": str, "close": float},
)

imoex.loc[:, "year-month"] = imoex["date"].str[:-2]
imoex_by_months = imoex.groupby("year-month")["close"].apply(list).to_list()

users_mean_by_months = [
    numpy.mean(users_by_months[i : i + 12]) for i in range(len(users_by_months) - 12)
]

imoex_slopes_by_months = []
for i in range(len(imoex_by_months) - 12):
    sample = numpy.array(list(chain.from_iterable(imoex_by_months[i : i + 12])))
    step_range = numpy.arange(start=5, stop=sample.size // 2, step=1)
    results = numpy.array([(step, RS(sample, step)) for step in step_range])
    log_x = numpy.log(results[:, 0])
    log_y = numpy.log(results[:, 1])
    imoex_slopes_by_months += [stats.linregress(log_x, log_y).slope]

# %%

color = numpy.array([i for i, v in sorted(enumerate(imoex_slopes_by_months), key=lambda iv: iv[1])])

seaborn.regplot(x=users_mean_by_months, y=imoex_slopes_by_months)
seaborn.scatterplot(x=users_mean_by_months, y=imoex_slopes_by_months, hue=color)
plot.show()
