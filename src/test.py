import glob
import pandas as pd
import sys
import datetime
from os.path import join
from core.processor import Processor

# ---------------------------------------------------------
# Declaração de constantes
# ---------------------------------------------------------
BUS_STOPS = "/mnt/work/datalake/trusted/busstops/year={0}/month={1:02d}/day={2:02d}"
VEHICLES = "/mnt/work/datalake/trusted/vehicles/year={0}/month={1}/day={2}"

# ---------------------------------------------------------
# Rotina principal
# ---------------------------------------------------------

# ---------------------------------------------------------
# Carregamento dos arquivos .parquet
# ---------------------------------------------------------

year = 2022
month = 7
day = 11
n_threads = 1

file_bus_stops = BUS_STOPS.format(year, month, day)
file_vehicles = VEHICLES.format(year, month, day)
base_date = datetime.date(year, month, day)

print("Processando a data base '{0}'".format(base_date))

parquet_files = glob.glob(join(file_bus_stops, "*.parquet"))
data = [pd.read_parquet(f) for f in parquet_files]
bus_stops = pd.concat(data, ignore_index = True)

parquet_files = glob.glob(join(file_vehicles, "*.parquet"))
data = [pd.read_parquet(f) for f in parquet_files]
vehicles = pd.concat(data, ignore_index = True)

p = Processor(bus_stops, vehicles, base_date)

p.process_line("216")