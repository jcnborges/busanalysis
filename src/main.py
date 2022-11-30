import glob
import pandas as pd
from os.path import join
from core.processor import Processor

# ---------------------------------------------------------
# Declaração de constantes
# ---------------------------------------------------------
DAY = 11
MONTH = 7
YEAR = 2022
BUS_STOPS = "/mnt/work/datalake/trusted/busstops/year={0}/month={1:02d}/day={2:02d}".format(YEAR, MONTH, DAY)
VEHICLES = "/mnt/work/datalake/trusted/vehicles/year={0}/month={1}/day={2}".format(YEAR, MONTH, DAY)
LINHAS = ['829', '303', '020', '506', '606', '607', '203', '505', '828']

# ---------------------------------------------------------
# Rotina principal
# ---------------------------------------------------------

# ---------------------------------------------------------
# Carregamento dos arquivos .parquet
# ---------------------------------------------------------
parquet_files = glob.glob(join(BUS_STOPS, "*.parquet"))
data = [pd.read_parquet(f) for f in parquet_files]
bus_stops = pd.concat(data, ignore_index = True)

parquet_files = glob.glob(join(VEHICLES, "*.parquet"))
data = [pd.read_parquet(f) for f in parquet_files]
vehicles = pd.concat(data, ignore_index = True)

p = Processor(bus_stops, vehicles)

p.process_line("829") 