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
if (len(sys.argv) != 5):
    print("Parâmetros de entrada: year month day n_threads")
    exit()

year = int(sys.argv[1])
month = int(sys.argv[2])
day = int(sys.argv[3])
n_threads = int(sys.argv[4])

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

p.process_all_lines(n_threads)