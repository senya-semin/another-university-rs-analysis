import logging
from itertools import product
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas

logging.basicConfig(level=logging.DEBUG)


def collect(collection_name: str, url_name: str) -> None:
    def download() -> List[Path]:
        # dateformat: YYYYMMDD
        template_url = "https://www.twse.com.tw/en/{url_name}?response=csv&date={date}"
        logging.debug(f"Template URL: '{template_url}'.")
        dates = tuple(
            f"{date[0]}{date[1]:02}{date[2]:02}"
            for date in product(range(1999, 2020 + 1), range(1, 12 + 1), (1,))
        )
        logging.debug(f"Dates to be downloaded: {dates}.")

        filenames = []
        for i, date in enumerate(dates):
            url = template_url.format(url_name=url_name, date=date)
            logging.info(f"Downloading {date} ({i + 1} of {len(dates)}) from '{url}'...")
            download_dir = Path(f".downloaded_{collection_name}")
            download_dir.mkdir(exist_ok=True)
            filename = download_dir / f"{date}.csv"
            urlretrieve(url, filename)
            logging.info(f"Downloaded as '{filename}'.")
            filenames += [filename]
        return filenames

    def fix_files(filenames: List[Path]) -> None:
        """The footer lines count is variable from file to file, so we can't use the default
        Pandas' solution to skip these lines. But we know all the footers starts with the "Remarks:"
        string. This fix method is intended to remove this line and all the ones after it."""
        logging.info("Fixing files...")
        for filename in filenames:
            logging.debug(f"Fixing file '{filename}'...")
            with open(filename, "r") as file:
                content = file.read()
            content = content.split('"Remarks:"')[0]
            with open(filename, "w") as file:
                file.write(content)

    def combine(filenames: List[Path]) -> None:
        logging.info(f"Combining {collection_name}...")
        combined: pandas.DataFrame = None
        for filename in filenames:
            logging.debug(f"Combining '{filename}'...")
            csv = pandas.read_csv(
                filename,
                header=1,
                parse_dates=["Date"],
                thousands=",",
            )
            combined = csv if combined is None else combined.append(csv)
        # the line fixes bad dataframe columns parsing
        combined = combined.loc[:, ~combined.columns.str.contains("^Unnamed")]
        # rename the header to ease its parsing in the other modules
        combined.columns = (c.lower().replace(" ", "-") for c in combined.columns)
        filename_ = f"{collection_name}.csv"  # the underscore is used because of mypy
        with open(filename_, "w") as file:
            logging.info(f"Saving the combined to '{filename_}'...")
            combined.to_csv(file, index=False)

    logging.info(f"Collecting {collection_name}...")
    filenames = download()
    fix_files(filenames)
    combine(filenames)


collect(collection_name="clients", url_name="exchangeReport/FMTQIK")
collect(collection_name="indexes", url_name="indicesReport/MI_5MINS_HIST")
