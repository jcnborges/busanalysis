from dataprocessing.processors.refined_ingestion import BusItineraryRefinedProcess, BusLineRefinedProcess, BusTrackingRefinedProcess
# ---------------------------------------------------------
# Rotina principal
# ---------------------------------------------------------

# ---------------------------------------------------------
# Carregamento dos arquivos .parquet
# ---------------------------------------------------------

year = '2022'
month = '7'
day = '12'

BusItineraryRefinedProcess(year, month, day)()
BusLineRefinedProcess(year, month, day)()
BusTrackingRefinedProcess(year, month, day, '863')()