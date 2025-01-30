import os
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from os.path import join
from pathlib import Path
import sys

if (len(sys.argv) != 7):
    print("Parâmetros de entrada: events_file_name connections_file_name start_base_date end_base_date batch_size batch_num")
    exit()

events_file_name = sys.argv[1].split('.')
connections_file_name = sys.argv[2].split('.')
start_base_date = sys.argv[3]
end_base_date = sys.argv[4]
batch_size = int(sys.argv[5])
batch_num = int(sys.argv[6])

T = 1200 # tempo máximo (segundos)
for base_date in pd.date_range(start_base_date, end_base_date, freq='D').to_list():
        
    event = pd.read_parquet(f'{events_file_name[0]}.{events_file_name[1]}')
    event.base_date = pd.to_datetime(event.base_date)#event.base_date.astype(str)
    event.query(f"base_date == '{base_date}'", inplace = True)
    event = event[['vehicle', 'event_timestamp', 'dummy_id', 'line_code']].drop_duplicates()
    connections = pd.read_parquet(f'{connections_file_name[0]}.{connections_file_name[1]}')
    connections = list(connections[['dummy_id_x', 'dummy_id_y', 'dist']].itertuples(index = False, name = None))
    start = (batch_num - 1) * batch_size
    end = len(connections) if start + batch_size > len(connections) else start + batch_size
    connections = connections[start:end]
        
    edges = []
    total = len(connections)
    c = 0

    print(f"Base date: {base_date:%Y-%m-%d} \tBatch: {batch_num} \tTotal: {total} \tStatus: Processing started...")
    
    for connection in connections:
        event_u = event.query(f"dummy_id == {connection[0]}").copy()
        event_v = event.query(f"dummy_id == {connection[1]}").copy()
        for idx1, row1 in event_u.iterrows():
            for idx2, row2 in event_v.query(f"line_code != '{row1['line_code']}' and event_timestamp > '{row1['event_timestamp']}' and event_timestamp <= '{row1['event_timestamp'] + pd.Timedelta(T, unit = 's')}'").iterrows():
                t0 = row1['event_timestamp']
                t1 = row2['event_timestamp']
                dt = (t1 - t0).total_seconds()
                if (dt > 0 and dt <= T):
                    edges.append((row1['dummy_id'], row2['dummy_id'], row1['line_code'], row2['line_code'], row1['vehicle'], row2['vehicle'], t0, t1))
        c = c + 1
        #uncomment for debug
        #print("Connection: {0}/{1} [{2:.2f}%]".format(c, total, 100 * c / total))               
    
    positive_edges = pd.DataFrame(edges, columns = ['dummy_id_u', 'dummy_id_v', 'line_code_u', 'line_code_v', 'vehicle_u', 'vehicle_v', 'event_timestamp_u', 'event_timestamp_v'])

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    positive_edges.to_parquet(f'{output_dir}//{base_date:%Y-%m-%d}_{batch_num}.parquet')

    print(f"Base date: {base_date:%Y-%m-%d} \tBatch: {batch_num} \tTotal: {total} \tStatus: Processing done!")