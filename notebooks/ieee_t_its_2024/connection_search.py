import pandas as pd
import numpy as np
from haversine import haversine, Unit
from os.path import join
from pathlib import Path
import sys

if (len(sys.argv) != 4):
    print("Parâmetros de entrada: file_name batch_size batch_num")
    exit()

file_name = sys.argv[1].split('.')
batch_size = int(sys.argv[2])
batch_num = int(sys.argv[3])

event = pd.read_parquet('vw_event.parquet')
event
connections = pd.read_parquet(f'{file_name[0]}.{file_name[1]}')
connections = list(connections.filter(['busstop_a', 'busstop_b']).itertuples(index = False, name = None))
start = (batch_num - 1) * batch_size
end = len(connections) if start + batch_size > len(connections) else start + batch_size
connections = connections[start:end]

S = 600 # distância máxima (metros)
T = 1200 # tempo máximo (segundos)
edges = []
total = len(connections)
c = 0

for connection in connections:
    for idx1, row1 in event.query(f"dummy_legacy_id == {connection[0]}").iterrows():
        for idx2, row2 in event.query(f"dummy_legacy_id == {connection[1]} and line_code != '{row1['line_code']}' and event_timestamp > '{row1['event_timestamp']}' and event_timestamp <= '{row1['event_timestamp'] + pd.Timedelta(T, unit = 's')}'").iterrows():
            p0 = (row1['latitude'], row1['longitude'])
            p1 = (row2['latitude'], row2['longitude'])
            t0 = row1['event_timestamp']
            t1 = row2['event_timestamp']
            ds = haversine(p0, p1, unit = Unit.METERS)
            dt = (t1 - t0).total_seconds()
            if (ds < S and dt > 0 and dt <= T):
                edges.append((row1['dummy_legacy_id'], row2['dummy_legacy_id'], row1['line_code'], row2['line_code'], row1['vehicle'], row2['vehicle'], t0, t1))
    c = c + 1
    print("Connection: {0}/{1} [{2:.2f}%]".format(c, total, 100 * c / total))               

positive_edges = pd.DataFrame(edges, columns = ['dummy_legacy_id_u', 'dummy_legacy_id_v', 'line_code_u', 'line_code_v', 'vehicle_u', 'vehicle_v', 'event_timestamp_u', 'event_timestamp_v'])
positive_edges.to_parquet(f'{file_name[0]}_{batch_num}.parquet')