"""
Questo script scarica file ZIP da URL specificati, estrae i file CSV
e filtra i dati in base a una colonna e soglia specificate.
I file CSV filtrati vengono salvati in una cartella di output.
"""
import io
import os
import sys
import zipfile

import polars as pl
import requests
from config import REGION_LINKS

sys.path.append(os.path.abspath("../src"))  # noqa: E402
from logging_utils import setup_logger  # noqa: E402

log = setup_logger("download_istat_distances")


def process_zip_and_save_filtered_csvs(
    url: str,
    filter_column: str,
    threshold: float,
    output_folder: str = "output_filtrato",
):
    """
    Scarica un file ZIP da un URL, estrae i file CSV
    e filtra i dati in base a una colonna e soglia specificate.
    Args:
        url (str): URL del file ZIP da scaricare.
        filter_column (str): Nome della colonna da filtrare.
        threshold (float): Soglia per il filtro.
        output_folder (str): Cartella di output per i file filtrati.
    """
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Scarica lo ZIP
    log.info(f"Scaricando ZIP da: {url}")
    response = requests.get(url)
    response.raise_for_status()

    # Estrai lo ZIP in memoria
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for filename in z.namelist():
            if filename.endswith(".csv"):
                log.info(f"Elaborando file: {filename}")
                with z.open(filename) as file:
                    try:
                        # Legge il CSV con Polars
                        df = pl.read_csv(file, separator=";")

                        if filter_column in df.columns:
                            # Filtra i dati
                            df = df.with_columns(
                                pl.col(filter_column)
                                .str.replace(",", ".")
                                .cast(pl.Float64, strict=False)
                                .alias(filter_column)
                            )
                            df_filtered = df.filter(pl.col(filter_column) < threshold)

                            # Costruisce il path per il file filtrato
                            output_path = os.path.join(
                                output_folder, f"filtered_{os.path.basename(filename)}"
                            )

                            # Salva il CSV filtrato
                            df_filtered.write_csv(output_path)
                            log.info(
                                f"✔ Salvato: {output_path} ({df_filtered.shape[0]} righe)"
                            )
                        else:
                            log.warning(
                                f"⚠ Colonna '{filter_column}' non trovata in {filename}"
                            )
                    except Exception as e:
                        log.error(
                            f"❌ Errore durante la lettura o scrittura di {filename}: {e}"
                        )


def main(filter_column: str, threshold: float, output_folder: str = "output_filtrato"):
    """
    Funzione principale per scaricare e filtrare i file CSV da URL specificati.
    Args:
        filter_column (str): Nome della colonna da filtrare.
        threshold (float): Soglia per il filtro.
        output_folder (str): Cartella di output per i file filtrati.
    """
    log.info("Inizio del download e filtraggio dei file CSV.")
    for region, url in REGION_LINKS.items():
        log.info(f"Elaborando regione: {region}")
        process_zip_and_save_filtered_csvs(
            url, filter_column, threshold, output_folder + "/Italia/" + region
        )
    log.info("Completato il download e filtraggio dei file CSV.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scarica e filtra file CSV da URL specificati."
    )
    parser.add_argument(
        "--filter_column",
        type=str,
        required=True,
        help="Nome della colonna da filtrare.",
    )
    parser.add_argument(
        "--threshold", type=float, required=True, help="Soglia per il filtro."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output_filtrato",
        help="Cartella di output per i file filtrati.",
    )

    args = parser.parse_args()
    main(args.filter_column, args.threshold, args.output_folder)
