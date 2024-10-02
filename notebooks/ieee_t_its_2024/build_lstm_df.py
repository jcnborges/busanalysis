import pandas as pd

day = 11
tw = 5
cd = [30, 1200]
ws = 1
density_choices = ['Low', 'Moderate', 'High']
df_cluster = pd.read_parquet('df_cluster.parquet')
agg_cluster =  pd.read_parquet('agg_cluster.parquet')
df_bus_availability = pd.read_parquet('df_bus_availability.parquet')
positive_edges = pd.read_parquet('positive_edges.parquet')
cluster_high = list(agg_cluster.query(f"density == '{density_choices[2]}'").cluster)
df_cluster.query(f"cluster == {cluster_high}").head()

def build_dataset(group_id):
    edge_info = df_cluster.query(f"group_id == {group_id}")
    if (edge_info['bus_count_u'] < edge_info['bus_count_v']).bool():
        minor_id = list(edge_info[['dummy_legacy_id_u', 'line_code_u']].itertuples(index=False, name=None))[0]
        bigger_id = list(edge_info[['dummy_legacy_id_v', 'line_code_v']].itertuples(index=False, name=None))[0]
    else:
        minor_id = list(edge_info[['dummy_legacy_id_v', 'line_code_v']].itertuples(index=False, name=None))[0]
        bigger_id = list(edge_info[['dummy_legacy_id_u', 'line_code_u']].itertuples(index=False, name=None))[0]
    minor_df = df_bus_availability.query(f"dummy_legacy_id == {minor_id[0]} and line_code == '{minor_id[1]}'").filter(['time_window', 'bus_count']).copy()
    minor_df['series'] = 'minor'
    bigger_df = df_bus_availability.query(f"dummy_legacy_id == {bigger_id[0]} and line_code == '{bigger_id[1]}'").filter(['time_window', 'bus_count']).copy()
    bigger_df['series'] = 'bigger'
    bus_count_df = pd.concat([minor_df, bigger_df], ignore_index = True)
    bus_count_df.rename({'bus_count': 'value'}, axis = 1, inplace = True)
    pe = positive_edges.query(f"group_id == {group_id} and duration >= {cd[0]} and duration <= {cd[1]}").copy()
    pe['time_window'] = (60 * pe["event_timestamp_v"].dt.hour + pe["event_timestamp_v"].dt.minute).floordiv(tw)
    pe_df = pd.merge(bus_count_df.time_window.drop_duplicates(), pe.groupby('time_window').size().reset_index(name='conn_count'), how = 'left').fillna(0)
    pe_df['series'] = 'connection'
    pe_df.rename({'conn_count': 'value'}, axis = 1, inplace = True)
    lstm_df = pd.concat([bus_count_df, pe_df], ignore_index = True)
    lstm_df['group_id'] = group_id
    lstm_df['cluster'] = edge_info.cluster.values[0]
    return lstm_df

import threading
from concurrent.futures import ThreadPoolExecutor

def build_dataset_thread(group_ids, lstm_dfs):
    """Builds the dataset for a list of group IDs, appends them to the lstm_dfs list.
    """
    c = 0
    for group_id in group_ids:
        lstm_dfs.append(build_dataset(group_id))
        c += 1
        if c % 100 == 0:
            print(f"[{threading.current_thread().name}] {c} / {len(group_ids)}")

lstm_dfs = []
group_ids = list(df_cluster.query(f"cluster == {cluster_high}").group_id)

# Divide group_ids into chunks of 1,000
chunk_size = 1000
group_id_chunks = [group_ids[i:i + chunk_size] for i in range(0, len(group_ids), chunk_size)]

with ThreadPoolExecutor(max_workers=8) as executor:
    for chunk in group_id_chunks:
        executor.submit(build_dataset_thread, chunk, lstm_dfs)

# lstm_dfs will contain the datasets for each group

df = pd.concat(lstm_dfs)
df = df.pivot(index = ['group_id', 'cluster', 'time_window'], columns = 'series', values = 'value').reset_index()
df.filter(['group_id', 'cluster', 'time_window', 'minor', 'bigger', 'connection']).to_parquet('lstm_df.parquet', index = None)