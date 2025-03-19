import os
import pandas as pd
import sys
import MAG as nxmag
import networkx as nx
from os.path import join
from pathlib import Path

if (len(sys.argv) != 6):
    print("Parâmetros de entrada: mag_file_name mag_aspect start_value start_time[min] cover_pct[0-1]")
    exit()

mag_file_name = sys.argv[1].split('.')
mag_aspect = sys.argv[2].split('.')
start_value = sys.argv[3]
start_time = sys.argv[4]
cover_pct = float(sys.argv[5])

if cover_pct < 0 or cover_pct > 1:
    print("Percentual de cobertura [cover_pct] de estar entre 0 e 1")

mag = pd.read_parquet(f"{mag_file_name[0]}.parquet")
aspect = 'neighborhood'
aspect_u = f"{aspect}_u"
aspect_v = f"{aspect}_v"
neigborhoods = set(mag[aspect_u]).union(mag[aspect_v])

G = nxmag.MultiAspectDiGraph()
for idx, row in mag.iterrows():
    u = (row['dummy_id_u'], row['line_code_u'], row['vehicle_u'], row['neighborhood_u'], row['ts_u'], row['tm_u'], row['th_u'])
    v = (row['dummy_id_v'], row['line_code_v'], row['vehicle_v'], row['neighborhood_v'], row['ts_v'], row['tm_v'], row['th_v'])
    w = row['ds']
    G.add_edge(u, v, weight = w)   

neighborhood_time = []

if start_value == 'ignore':
    starting_nodes = mag.query(f"tm_u == {start_time}")[[
        'dummy_id_u', 'line_code_u', 'vehicle_u', 'neighborhood_u', 'ts_u', 'tm_u', 'th_u'
    ]].drop_duplicates()
else:
    starting_nodes = mag.query(f"tm_u == {start_time} and {aspect_u} == '{start_value}'")[[
        'dummy_id_u', 'line_code_u', 'vehicle_u', 'neighborhood_u', 'ts_u', 'tm_u', 'th_u'
    ]].drop_duplicates()

print(f"Mag: {mag_file_name[0]} Aspect: {aspect} \tStart value: {start_value} \tStart time: {start_time} \tStatus: Processing started...")

i = 1
for row in starting_nodes.itertuples(index = False, name = None):
    dict_nt = {}
    flag = False
    for node in list(nx.dfs_preorder_nodes(G, source = row)):
        if node[3] not in dict_nt.keys():
            dict_nt.update({node[3]: node[4]})
            pct = 1 - len(neigborhoods.difference(set(dict_nt.keys()))) / len(neigborhoods)
            if pct >= cover_pct:
                flag = True
                break
    
    # Create the DataFrame directly from the dictionary
    nt = pd.DataFrame.from_dict(dict_nt, orient = 'index', columns = ['ts'])
    
    # Add a 'neighborhood' column using the index
    nt['neighborhood'] = nt.index
    
    # Reset the index to make 'neighborhood' a regular column
    nt = nt.reset_index(drop=True)
    
    # Reorder columns to match your original intent
    nt = nt[['neighborhood', 'ts']]

    if flag:
        nt['start_value'] = start_value
        nt['start_time'] = start_time
        nt['starting_node_idx'] = i        
        neighborhood_time.append(nt)    
    
    i = i + 1
    
neighborhood_time = pd.concat(
    neighborhood_time, ignore_index = True
)

output_dir = "cover_time_output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

neighborhood_time.to_parquet(f"{output_dir}//{mag_file_name[0]}_{aspect}_{start_value}_{start_time}.parquet")

print(f"Mag: {mag_file_name[0]} Aspect: {aspect} \tStart value: {start_value} \tStart time: {start_time} \tStatus: Processing done!")