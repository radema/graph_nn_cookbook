import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import os
    import sys

    import marimo as mo
    import requests
    from config import REGION_LINKS
    from extract_data import process_zip_and_save_filtered_csvs

    sys.path.append(os.path.abspath("../../src"))  # noqa: E402
    from logging_utils import setup_logger  # noqa: E402

    return REGION_LINKS, mo, process_zip_and_save_filtered_csvs, setup_logger


@app.cell
def _(setup_logger):
    log = setup_logger("marimo_ingestion_notebook")
    return (log,)


@app.cell
def _(mo):
    FILTER_COLUMN = mo.ui.text(label="Enter numeric column to prune links:", value="")
    FILTER_COLUMN
    return (FILTER_COLUMN,)


@app.cell
def _(FILTER_COLUMN):
    FILTER_COLUMN.value
    return


@app.cell
def _(mo):
    OUTPUT_FOLDER = mo.ui.text(
        label="Enter folder path", value="", placeholder="/path/to/folder"
    )

    OUTPUT_FOLDER
    return (OUTPUT_FOLDER,)


@app.cell
def _(mo):
    THRESHOLD = mo.ui.number(
        label="Enter a thresholder number to prune links:",
        value=99,  # default value
        start=0,  # optional: minimum allowed value
        step=1,  # optional: step size for up/down arrows
    )

    THRESHOLD
    return (THRESHOLD,)


@app.cell
def _(
    FILTER_COLUMN,
    OUTPUT_FOLDER,
    REGION_LINKS,
    THRESHOLD,
    log,
    process_zip_and_save_filtered_csvs,
):
    log.info("Inizio del download e filtraggio dei file CSV.")
    for region, url in REGION_LINKS.items():
        log.info(f"Elaborando regione: {region}")
        process_zip_and_save_filtered_csvs(
            url=url,
            filter_column=FILTER_COLUMN.value,
            threshold=THRESHOLD.value,
            output_folder=OUTPUT_FOLDER.value + "/Italia/" + region,
        )
    log.info("Completato il download e filtraggio dei file CSV.")
    return


if __name__ == "__main__":
    app.run()
