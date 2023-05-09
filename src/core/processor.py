import pandas as pd
import numpy as np
import os
import threading
import time
import traceback
from haversine import haversine, Unit
from numpy.random import binomial
from os.path import join

# ---------------------------------------------------------
# Declaração de constantes
# ---------------------------------------------------------
FILE_BUS_STOPS = "bus_stops.csv"
FILE_VEHICLES = "vehicles.csv"
FILE_LINES = "lines.csv"
FILE_ETL_ITINERARY = "etl_itinerary.csv"
FILE_ETL_EVENT = "etl_event.csv"
FILE_ETL_LINE = "etl_line.csv"
FILE_ERROR = "error.log"
ITINERARY_THRESHOLD = 0.8

class Processor:

    def __init__(self, bus_stops, vehicles, lines, base_date):
        
        self.base_date = base_date
        self._lock = threading.Lock()

        # ---------------------------------------------------------
        # Recuperar os data frames iniciais (caso existam)
        # ---------------------------------------------------------
        
        self.base_dir = base_date.strftime("%Y-%m-%d")
        self.file_bus_stops = join(self.base_dir, FILE_BUS_STOPS)
        self.file_vehicles = join(self.base_dir, FILE_VEHICLES)
        self.file_lines = join(self.base_dir, FILE_LINES)
        self.file_etl_itinerary = join(self.base_dir, FILE_ETL_ITINERARY)
        self.file_etl_event = join(self.base_dir, FILE_ETL_EVENT)
        self.FILE_ETL_LINE = join(self.base_dir, FILE_ETL_LINE)
        self.file_error = join(self.base_dir, FILE_ERROR)

        if (not os.path.exists(self.base_dir)):
            os.mkdir(self.base_dir)

        if (os.path.exists(self.file_bus_stops) and os.path.exists(self.file_vehicles) and os.path.exists(self.file_lines)):
            self.bus_stops = pd.read_csv(self.file_bus_stops, dtype = {"line_code": np.str})
            self.vehicles = pd.read_csv(self.file_vehicles, dtype = {"line_code": np.str}, parse_dates = ["event_timestamp"])
            self.lines = pd.read_csv(self.file_lines, dtype = {"line_code": np.str})
        else:
            self.vehicles = vehicles
            self.bus_stops = bus_stops
            self.lines = lines

            # ---------------------------------------------------------
            # Limpar bus stops
            # ---------------------------------------------------------
            self.bus_stops['latitude'] = pd.to_numeric(self.bus_stops['latitude'])
            self.bus_stops['longitude'] = pd.to_numeric(self.bus_stops['longitude'])
            self.bus_stops['seq'] = pd.to_numeric(self.bus_stops['seq'])
            self.bus_stops.sort_values(by = ["line_code", "itinerary_id", "seq"], inplace = True)
            self.bus_stops = self.bus_stops[{"itinerary_id", "line_code", "latitude", "longitude", "name", "number", "line_way", "type"}].drop_duplicates()
            self.bus_stops["id"] = pd.to_numeric(self.bus_stops["number"]) #self.bus_stops["name_norm"].astype('category').cat.codes
            self.bus_stops["next_stop_id"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["id"].shift(-1), None)
            self.bus_stops["next_stop_id"] = self.bus_stops["next_stop_id"].astype('Int64')
            self.bus_stops["next_stop_latitude"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["latitude"].shift(-1), np.nan)
            self.bus_stops["next_stop_longitude"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["longitude"].shift(-1), np.nan)
            self.bus_stops["next_stop_delta_s"] = self.bus_stops.apply(lambda row: haversine((row["latitude"], row["longitude"]), (row["next_stop_latitude"], row["next_stop_longitude"]), unit = Unit.METERS), axis = 1)
            self.bus_stops = self.bus_stops.query("id != next_stop_id or next_stop_id.isnull()")
            self.bus_stops["seq"] = self.bus_stops.groupby(["itinerary_id", "line_code"]).cumcount()
            self.bus_stops["max_seq"] = self.bus_stops.groupby(["itinerary_id", "line_code"])["seq"].transform(max)

            # ---------------------------------------------------------
            # Calcular a velocidade média dos ônibus
            # ---------------------------------------------------------
            self.vehicles["event_timestamp"] = pd.to_datetime(self.vehicles["event_timestamp"])
            self.vehicles.sort_values(by = ["line_code", "vehicle", "event_timestamp"], inplace = True)

            # ---------------------------------------------------------
            # Gravar os data frames iniciais
            # ---------------------------------------------------------
            self.bus_stops.to_csv(self.file_bus_stops, index = False)
            self.vehicles.to_csv(self.file_vehicles, index = False)
            self.lines.to_csv(self.file_lines, index = False)

            # ---------------------------------------------------------
            # Remover arquivos de saída se existirem
            # ---------------------------------------------------------
            if os.path.exists(self.file_etl_itinerary):
                os.remove(self.file_etl_itinerary)
            if os.path.exists(self.file_etl_event):    
                os.remove(self.file_etl_event)

            # ---------------------------------------------------------
            # Selecionar campos utilizados no ETL e exportar
            # ---------------------------------------------------------         
            self.bus_stops.to_csv(self.file_etl_itinerary, header = True, columns = ["line_code", "id", "name", "latitude", "longitude", "type", "itinerary_id", "line_way", "next_stop_id", "next_stop_delta_s", "seq"], index = False)
            self.lines.to_csv(self.FILE_ETL_LINE, header = True, columns = ["line_code", "line_name", "service_category", "color"], index = False)

    # ---------------------------------------------------------
    # Procurar o itinerário do ônibus (sentido da rota)
    # ---------------------------------------------------------
    def search_itinerary(self, line_code, destino, origem_1, origem_2 = np.nan):
        # Destino x Origem - determina o itinerário
        if np.isnan(origem_1) or origem_1 == destino: # TODO: verificar esses casos (onde id == last_id)
            return np.nan
        aux = pd.merge(self.bus_stops.query("line_code == @line_code and id == @destino"), self.bus_stops.query("line_code == @line_code and id == @origem_1"), on = ["itinerary_id", "line_code"]).query("seq_x > seq_y")["itinerary_id"]
        if aux.empty:
            if np.isnan(origem_2) or origem_2 == destino: 
                return np.nan
            aux = pd.merge(self.bus_stops.query("line_code == @line_code and id == @destino"), self.bus_stops.query("line_code == @line_code and id == @origem_2"), on = ["itinerary_id", "line_code"]).query("seq_x > seq_y")["itinerary_id"]
        if aux.empty:
            return np.nan
        return aux.iloc[0]

    # ---------------------------------------------------------
    # Procurar o ponto de ônibus mais próximo
    # ---------------------------------------------------------
    def search_bus_stop(self, line_code, latitude, longitude):
        bs = self.bus_stops.query("line_code == @line_code")
        bs["distance"] = bs.apply(lambda row: haversine((latitude, longitude), (row["latitude"], row["longitude"]), unit = Unit.METERS), axis = 1)
        a = bs.sort_values("distance", ascending = True).iloc[0]
        return a["id"]

    # ---------------------------------------------------------
    # Passar filtro de itinerario para atenuar ruído
    # ---------------------------------------------------------
    def filter_itinerary(self, line_code, vehicle, actual_value, threshold, itinerary_probability):
        a = itinerary_probability.query("line_code == @line_code and vehicle == @vehicle")
        if (a.empty):
            return actual_value
        a = a.sort_values("%", ascending = False).iloc[0]
        if (a["%"] >= threshold):
            return a["itinerary_id"]
        else:
            return actual_value

    # ---------------------------------------------------------
    # Recuperar descontinuidades de tempo (pontos sem parada)
    # ---------------------------------------------------------
    def recover_data(self, n, events):
        
        while n >= 0:

            events = events.sort_values(by = ["line_code", "vehicle", "event_timestamp"])
            events["next_stop_seq"] = np.where((events["line_code"] == events["line_code"].shift(-1)) & (events["itinerary_id"] == events["itinerary_id"].shift(-1)) & (events["vehicle"] == events["vehicle"].shift(-1)), events["seq"].shift(-1), np.nan)
            events["next_event_timestamp"] = np.where((events["line_code"] == events["line_code"].shift(-1)) & (events["itinerary_id"] == events["itinerary_id"].shift(-1)) & (events["vehicle"] == events["vehicle"].shift(-1)), events["event_timestamp"].shift(-1), np.datetime64('NaT'))
            events["lost_positions"] = np.where(events["next_stop_id"].isna() == False, events["next_stop_seq"] - events["seq"] - 1, np.nan)

            if (n == 0):
                break
            
            if (len(events.query("lost_positions == @n")) > 0):

                # ---------------------------------------------------------
                # Recuperar as paradas perdidas de 1 em 1
                # ---------------------------------------------------------
                aux = pd.DataFrame(columns = events.columns)
                aux = pd.concat([aux, events.query("lost_positions == @n")])
                aux["last_eventtimestamp"] = aux["event_timestamp"]
                aux["estimated_next_delta_t"] = aux.apply(lambda row: (row["next_event_timestamp"] - row["last_eventtimestamp"]) / (n + 1), axis = 1) #aux.apply(lambda row: self.estimate_next_delta_t(row["next_event_timestamp"], row["lost_positions"]), axis = 1)
                aux["event_timestamp"] = aux.apply(lambda row: row["last_eventtimestamp"] + row["estimated_next_delta_t"], axis = 1)
                aux["last_id"] = aux["id"]
                aux["id"] = aux["next_stop_id"]
                aux["generated"] = True
                aux.drop(["latitude", "longitude", "seq", "next_stop_id", "next_stop_delta_s"], axis = 1, inplace = True)

                # --------------------------------------------------------------------------
                # Validar o timestamp gerado (maior que o anterior e menor que o proximo)
                # --------------------------------------------------------------------------
                aux.query("event_timestamp > last_eventtimestamp & event_timestamp < next_event_timestamp", inplace = True)
                aux = pd.merge(aux, self.bus_stops[{"line_code", "id", "itinerary_id", "seq", "next_stop_id", "next_stop_delta_s", "latitude", "longitude"}], on = ["line_code", "id", "itinerary_id"], how = "inner")            

                # ---------------------------------------------------------
                # Adicionar os registros selecionados
                # ---------------------------------------------------------
                events = pd.concat([events, aux], ignore_index = True)

            n = n - 1

        return events

    # ---------------------------------------------------------
    # Processar todas as linhas
    # ---------------------------------------------------------
    def process_all_lines(self, n_threads):
        self.linhas = self.bus_stops["line_code"].drop_duplicates()
        linhas_processadas = []
        if os.path.exists(self.file_etl_event):
            events = pd.read_csv(self.file_etl_event, header = 0, names = ["line_code", "vehicle", "id", "itinerary_id", "event_timestamp", "seq"], dtype = {"line_code": np.str})
            linhas_processadas = events["line_code"].astype(str).drop_duplicates().tolist()
        linhas_nao_processadas = list(set(self.linhas) - set(linhas_processadas))
        self.c = len(linhas_processadas)
        while len(linhas_nao_processadas):
            if (threading.active_count() <= n_threads):
                threading.Thread(
                    target = self.process_line,
                    args = (linhas_nao_processadas.pop(), 10,)
                ).start()
            time.sleep(1)           

    # ---------------------------------------------------------
    # Processar uma linha individualmente
    # ---------------------------------------------------------
    def process_line(self, linha, tentativas):
        success = False
        ctentativa = 0
        while not success and ctentativa < tentativas:
            try:
                print("Processando linha '{0}'...".format(linha))

                # -----------------------------------------------------------------------------------------
                # Cruzar a posição do ônibus com a localização dos pontos de ônibus e calcular a distância
                # -----------------------------------------------------------------------------------------
                dim_bus_stops = self.bus_stops.groupby(by = ["id"]).agg({"latitude": "mean", "longitude": "mean"}).reset_index()
                aux = self.vehicles.query("line_code == @linha")

                if (aux.empty):
                    print("Não há logs de veículos... Encerrando o processamento da linha!")
                    with self._lock:
                        self.c = self.c + 1
                        print("Linha '{0}' processada com sucesso! [{1} de {2} ({3:.2f}%)]".format(linha, self.c, len(self.linhas), 100 * self.c / len(self.linhas)))
                        return

                aux["id"] = aux.apply(lambda row: self.search_bus_stop(row["line_code"], row["latitude"], row["longitude"]), axis = 1)
                aux = pd.merge(aux, dim_bus_stops, on = ["id"])
                aux["distance"] = aux.apply(lambda row: haversine((row["latitude_x"], row["longitude_x"]), (row["latitude_y"], row["longitude_y"]), unit = Unit.METERS), axis = 1)
                aux.rename(columns = {"latitude_y": "latitude", "longitude_y": "longitude"}, inplace = True)   

                # ------------------------------------------------------------------
                # Computar passagem de ônibus próximo aos pontos (Eventos)
                # ------------------------------------------------------------------
                time_window = 10
                events = aux.query("distance <= 50")[{"line_code", "vehicle", "event_timestamp", "id", "latitude", "longitude"}]
                events["year"] = events["event_timestamp"].dt.year
                events["month"] = events["event_timestamp"].dt.month
                events["day"] = events["event_timestamp"].dt.day
                events["hour"] = events["event_timestamp"].dt.hour
                events["minute"] = time_window * events["event_timestamp"].dt.minute.floordiv(time_window)
                events = events.groupby(by = ["line_code", "vehicle", "id", "year", "month", "day", "hour", "minute", "latitude", "longitude"]).agg({"event_timestamp": "mean"}).reset_index()
                events.drop(["year", "month", "day", "hour", "minute"], axis = 1, inplace = True)
                events = events.sort_values(by = ["line_code", "vehicle", "event_timestamp"])
                events["last_1_id"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["id"].shift(1), np.nan)
                events["last_2_id"] = np.where((events["line_code"] == events["line_code"].shift(2)) & (events["vehicle"] == events["vehicle"].shift(2)), events["id"].shift(2), np.nan)
                events.query("id != last_1_id and id != last_2_id", inplace = True)
                events["itinerary_id"] = events.apply(lambda row: self.search_itinerary(row["line_code"], row["id"], row["last_1_id"], row["last_2_id"]), axis = 1)
                events["itinerary_id"] = events["itinerary_id"].fillna(method = "bfill")
                
                # ---------------------------------------------------------
                # Passar filtro de itinerario para atenuar ruído
                # ---------------------------------------------------------
                itinerary_probability = events.groupby(by = ["line_code", "vehicle", "itinerary_id"]).agg({"id": "count"}).reset_index()
                itinerary_probability["%"] = itinerary_probability["id"] / itinerary_probability.groupby(by = ["line_code", "vehicle"])["id"].transform("sum")
                counts, bins = np.histogram(itinerary_probability["%"])
                
                events["itinerary_id"] = events.apply(lambda row: self.filter_itinerary(row["line_code"], row["vehicle"], row["itinerary_id"], ITINERARY_THRESHOLD, itinerary_probability), axis = 1)                        
                events = pd.merge(events, self.bus_stops[{"line_code", "id", "itinerary_id", "seq", "max_seq"}], on = ["line_code", "id", "itinerary_id"], how = "left")         
                events.query("not (itinerary_id.notnull() and seq.isnull())", inplace = True, engine = "python")
                events.drop(["seq", "max_seq", "last_1_id", "last_2_id"], axis = 1, inplace = True)

                # ---------------------------------------------------------
                # Reordenar logs válidos
                # ---------------------------------------------------------
                events = events.sort_values(by = ["line_code", "vehicle", "event_timestamp"])
                events["last_eventtimestamp"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["event_timestamp"].shift(1), np.datetime64('NaT'))
                events["last_id"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["id"].shift(1), np.nan)                
                events.query("id != last_id", inplace = True)
                events = pd.merge(events, self.bus_stops[{"line_code", "id", "itinerary_id", "seq", "next_stop_id", "next_stop_delta_s"}], on = ["line_code", "id", "itinerary_id"], how = "left")

                if (events.empty):
                    print("Não há logs válidos de veículos. Encerrando o processamento da linha!")
                    return

                events["generated"] = False

                # ---------------------------
                # Descomentar para testes
                # --------------------------                
                #events = self.recover_data(0, events) # somente para marcar lost_position
                #events.to_csv("csv_sem_interpolacao/events_finished_{0}.csv".format(linha))

                events = self.recover_data(7, events) # no máximo 7 interpolações
                events["last_stop_seq"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["seq"].shift(1), np.nan)
                events.query("seq == last_stop_seq + 1 or next_stop_seq == seq + 1 or seq == 0", inplace = True) # tratamento caso Nádia
                
                # ---------------------------
                # Descomentar para testes
                # ---------------------------
                #events.to_csv("csv_com_interpolacao/events_finished_{0}.csv".format(linha))         

                # ---------------------------------------------------------
                # Selecionar campos utilizados no ETL e exportar
                # ---------------------------------------------------------        
                with self._lock:
                    events.to_csv(self.file_etl_event, header = False, columns = ["line_code", "vehicle", "id", "itinerary_id", "event_timestamp", "seq"], index = False, mode = "a")
                    self.c = self.c + 1
                    print("Linha '{0}' processada com sucesso! [{1} de {2} ({3:.2f}%)]".format(linha, self.c, len(self.linhas), 100 * self.c / len(self.linhas)))                
                success = True
            except Exception as e:
                ctentativa = ctentativa + 1
                with self._lock:
                    with open(self.file_error, "a") as f:
                        f.write("Linha: {0} [{1}]\nTraceback:{2}".format(linha, ctentativa, traceback.format_exc()))
                    print(traceback.format_exc())