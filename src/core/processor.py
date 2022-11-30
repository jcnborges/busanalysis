import pandas as pd
import numpy as np
from haversine import haversine, Unit
from distfit import distfit
from numpy.random import binomial

class Processor:

    def __init__(self, bus_stops, vehicles):
        self.vehicles = vehicles
        self.bus_stops = bus_stops

        # ---------------------------------------------------------
        # Limpar bus stops
        # ---------------------------------------------------------
        self.bus_stops['latitude'] = pd.to_numeric(self.bus_stops['latitude'])
        self.bus_stops['longitude'] = pd.to_numeric(self.bus_stops['longitude'])
        self.bus_stops['seq'] = pd.to_numeric(self.bus_stops['seq'])
        self.bus_stops.sort_values(by = ["line_code", "itinerary_id", "seq"], inplace = True)
        self.bus_stops = self.bus_stops[{"itinerary_id", "line_code", "latitude", "longitude", "name", "number", "line_way", "type"}].drop_duplicates()
        self.bus_stops["name_norm"] = self.bus_stops["name"].str.replace("Rua", "R.", regex = False)
        self.bus_stops["name_norm"] = self.bus_stops["name_norm"].str.extract("([\w\s.,\d]+)").apply(lambda x: x.str.strip())
        self.bus_stops["name_norm"] = self.bus_stops["name_norm"].str.normalize("NFKD").str.encode("ascii", errors = "ignore").str.decode("utf-8")
        self.bus_stops["id"] = self.bus_stops["name_norm"].astype('category').cat.codes
        self.bus_stops["next_stop_id"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["id"].shift(-1), None)
        self.bus_stops["next_stop_name_norm"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["name_norm"].shift(-1), None)
        self.bus_stops["next_stop_latitude"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["latitude"].shift(-1), np.nan)
        self.bus_stops["next_stop_longitude"] = np.where((self.bus_stops["line_code"] == self.bus_stops["line_code"].shift(-1)) & (self.bus_stops["itinerary_id"] == self.bus_stops["itinerary_id"].shift(-1)), self.bus_stops["longitude"].shift(-1), np.nan)
        self.bus_stops["next_stop_delta_s"] = self.bus_stops.apply(lambda row: haversine((row["latitude"], row["longitude"]), (row["next_stop_latitude"], row["next_stop_longitude"]), unit = Unit.METERS), axis = 1)
        self.bus_stops = self.bus_stops.query("id != next_stop_id")
        self.bus_stops["seq"] = self.bus_stops.groupby(["itinerary_id", "line_code"]).cumcount()
        self.bus_stops["max_seq"] = self.bus_stops.groupby(["itinerary_id", "line_code"])["seq"].transform(max)

        # ---------------------------------------------------------
        # Calcular a velocidade média dos ônibus
        # ---------------------------------------------------------
        self.vehicles["event_timestamp"] = pd.to_datetime(self.vehicles["event_timestamp"])
        self.vehicles.sort_values(by = ["line_code", "vehicle", "event_timestamp"], inplace = True)
        self.vehicles["last_eventtimestamp"] = np.where((self.vehicles["line_code"] == self.vehicles["line_code"].shift(1)) & (self.vehicles["vehicle"] == self.vehicles["vehicle"].shift(1)), self.vehicles["event_timestamp"].shift(1), np.datetime64('NaT'))
        self.vehicles["last_latitude"] = np.where((self.vehicles["line_code"] == self.vehicles["line_code"].shift(1)) & (self.vehicles["vehicle"] == self.vehicles["vehicle"].shift(1)), self.vehicles["latitude"].shift(1), np.nan)
        self.vehicles["last_longitude"] = np.where((self.vehicles["line_code"] == self.vehicles["line_code"].shift(1)) & (self.vehicles["vehicle"] == self.vehicles["vehicle"].shift(1)), self.vehicles["longitude"].shift(1), np.nan)
        self.vehicles["delta_t"] = (self.vehicles["event_timestamp"] - self.vehicles["last_eventtimestamp"]).astype('timedelta64[s]')
        self.vehicles["delta_s"] = self.vehicles.apply(lambda row: haversine((row["latitude"], row["longitude"]), (row["last_latitude"], row["last_longitude"]), unit = Unit.METERS), axis = 1)
        self.vehicles["speed"] = 3.6 * (self.vehicles["delta_s"] / self.vehicles["delta_t"])
        self.vehicles = self.vehicles.query("delta_t <= 120 and speed >= 0 and speed <= 62.54")
        group = self.vehicles.groupby(by = ["line_code", "vehicle"])
        speed_windowed = group.rolling(window = "10s", min_periods = 1, on = "event_timestamp", closed = "both").agg({"speed": "mean"})
        self.vehicles.set_index(["line_code", "vehicle", "event_timestamp"], inplace = True)
        self.vehicles["speed_windowed"] = speed_windowed["speed"]
        self.vehicles["status"] = np.where(self.vehicles["speed_windowed"] <= 12.5, "STOPPED", "MOVING")
        self.vehicles.reset_index(inplace = True)

    # ---------------------------------------------------------
    # Encontrar a PDF que melhor se ajusta a velocidade
    # ---------------------------------------------------------
    def fit_speed_pdf(self, line_code, vehicle):
        a = self.vehicles.query("status == 'MOVING' and line_code == @line_code and vehicle == @vehicle")["speed"]
        a = a[np.isfinite(a)]
        if a.empty:
            return np.nan
        dist = distfit()
        dist.fit_transform(a)
        return dist

    # ---------------------------------------------------------
    # Encontrar a PDF que melhor se ajusta ao tempo de espera
    # ---------------------------------------------------------
    def fit_waiting_time_pdf(self, line_code, vehicle, waiting_time):
        a = waiting_time.query("last_status == 'MOVING' and status == 'STOPPED' and next_status == 'MOVING' and id == last_id and id == next_id and delta_t and line_code == @line_code and vehicle == @vehicle")["delta_t"]
        a = a[np.isfinite(a)]
        if a.empty:
            return np.nan    
        dist = distfit()
        dist.fit_transform(a)
        return dist

    # ---------------------------------------------------------
    # Calcular a probabilidade a priori de haver paradas
    # ---------------------------------------------------------
    def calculate_apriori_bus_stop(self, line_code, vehicle, events):
        a = events.query("line_code == @line_code and vehicle == @vehicle")
        if a.empty:
            return np.nan
        a = a.groupby(by = ["line_code", "vehicle", "status"]).agg({"id": "count"}).reset_index()
        a["%"] = a["id"] / a.groupby(by = ["line_code", "vehicle"])["id"].transform("sum")
        return a.query("status == 'STOPPED'")["%"].iloc[0]

    # ---------------------------------------------------------
    # Estimar o intervalo de tempo para a próxima parada
    # ---------------------------------------------------------
    def estimate_next_delta_t(self, next_stop_delta_s, speed_pdf, waiting_time_pdf, apriori_bus_stop, status = None):
        if status == "STOPPED":
            b = 1
        else:
            b = binomial(1, apriori_bus_stop)
        return 3.6 * next_stop_delta_s / speed_pdf.generate(1)[0] + b * waiting_time_pdf.generate(1)[0]

    # ---------------------------------------------------------
    # Procurar o itinerário do ônibus (sentido da rota)
    # ---------------------------------------------------------
    def search_itinerary(self, destino, origem):
        # Destino x Origem - determina o itinerário
        if np.isnan(origem) or origem == destino: # TODO: verificar esses casos (onde id == last_id)
            return np.nan
        aux = pd.merge(self.bus_stops.query("id == @destino"), self.bus_stops.query("id == @origem"), on = ["itinerary_id", "line_code"]).query("seq_x > seq_y")["itinerary_id"]
        if aux.empty:
            return np.nan
        return aux.iloc[0]

    # ---------------------------------------------------------
    # Passar filtro de itinerario para atenuar ruído
    # ---------------------------------------------------------
    def filter_itinerary(self, line_code, vehicle, actual_value, threshold, itinerary_probability):
        a = itinerary_probability.query("line_code == @line_code and vehicle == @vehicle").sort_values("%", ascending = False).iloc[0]
        if (a["%"] >= threshold):
            return a["itinerary_id"]
        else:
            return actual_value

    # ---------------------------------------------------------
    # Recuperar descontinuidades de tempo (pontos sem parada)
    # ---------------------------------------------------------
    def recover_data(self, n, max_iter, events):
        
        i = 0
        while True:
            events = events.sort_values(by = ["line_code", "vehicle", "event_timestamp"])
            events["next_stop_seq"] = np.where((events["line_code"] == events["line_code"].shift(-1)) & (events["itinerary_id"] == events["itinerary_id"].shift(-1)) & (events["vehicle"] == events["vehicle"].shift(-1)), events["seq"].shift(-1), np.nan)
            events["next_event_timestamp"] = np.where((events["line_code"] == events["line_code"].shift(-1)) & (events["itinerary_id"] == events["itinerary_id"].shift(-1)) & (events["vehicle"] == events["vehicle"].shift(-1)), events["event_timestamp"].shift(-1), np.datetime64('NaT'))
            events["lost_positions"] = np.where(events["next_stop_id"].isna() == False, events["next_stop_seq"] - events["seq"] - 1, np.nan)
            
            i = i + 1
            if (len(events.query("lost_positions == @n")) == 0 or i > max_iter):
                break

            # ---------------------------------------------------------
            # Recuperar as paradas perdidas de 1 em 1
            # ---------------------------------------------------------
            aux = pd.DataFrame(columns = events.columns)
            aux = pd.concat([aux, events.query("lost_positions == @n")])
            aux["last_name_norm"] = aux["name_norm"]
            aux["last_eventtimestamp"] = aux["event_timestamp"]
            aux["status"] = np.where(binomial(1, aux["apriori_bus_stop"]) == 1, "STOPPED", "MOVING")
            aux["estimated_next_delta_t"] = aux.apply(lambda row: self.estimate_next_delta_t(row["next_stop_delta_s"], row["speed_pdf"], row["waiting_time_pdf"], row["apriori_bus_stop"], row["status"]), axis = 1)
            aux["event_timestamp"] = aux["last_eventtimestamp"] + pd.to_timedelta(aux["estimated_next_delta_t"], unit = 's')
            aux["last_id"] = aux["id"]
            aux["id"] = aux["next_stop_id"]
            aux["generated"] = True
            aux.drop(["name_norm", "latitude", "longitude", "seq", "next_stop_id", "next_stop_delta_s"], axis = 1, inplace = True)

            # --------------------------------------------------------------------------
            # Validar o timestamp gerado (maior que o anterior e menor que o proximo)
            # --------------------------------------------------------------------------
            aux = pd.merge(aux, self.bus_stops[{"line_code", "id", "itinerary_id", "seq", "next_stop_id", "next_stop_delta_s", "name_norm", "latitude", "longitude"}], on = ["line_code", "id", "itinerary_id"], how = "inner")
            aux = aux.query("event_timestamp > last_eventtimestamp & event_timestamp < next_event_timestamp")

            # ---------------------------------------------------------
            # Adicionar os registros selecionados
            # ---------------------------------------------------------
            events = pd.concat([events, aux], ignore_index = True)

        return events

    # ---------------------------------------------------------
    # Processar uma linha individualmente
    # ---------------------------------------------------------
    def process_line(self, linha):

        # -----------------------------------------------------------------------------------------
        # Cruzar a posição do ônibus com a localização dos pontos de ônibus e calcular a distância
        # -----------------------------------------------------------------------------------------
        dim_bus_stops = self.bus_stops.groupby(by = ["id", "name_norm"]).agg({"latitude": "mean", "longitude": "mean"}).reset_index()
        aux = pd.merge(self.vehicles.query("line_code == @linha"), self.bus_stops.query("line_code == @linha")[{"line_code", "id"}].drop_duplicates(), on = ["line_code"])
        aux = pd.merge(aux, dim_bus_stops, on = ["id"])
        aux["distance"] = aux.apply(lambda row: haversine((row["latitude_x"], row["longitude_x"]), (row["latitude_y"], row["longitude_y"]), unit = Unit.METERS), axis = 1)

        # ---------------------------------------------------------
        # Computar passagem de ônibus próximo aos pontos
        # ---------------------------------------------------------
        time_window = 10
        events = aux.query("distance <= 50")[{"line_code", "vehicle", "event_timestamp", "id", "name_norm", "status", "latitude_y", "longitude_y"}]
        events["year"] = events["event_timestamp"].dt.year
        events["month"] = events["event_timestamp"].dt.month
        events["day"] = events["event_timestamp"].dt.day
        events["hour"] = events["event_timestamp"].dt.hour
        events["minute"] = time_window * events["event_timestamp"].dt.minute.floordiv(time_window)
        events = events.groupby(by = ["line_code", "vehicle", "id", "name_norm", "year", "month", "day", "hour", "minute", "latitude_y", "longitude_y"]).agg({"event_timestamp": "mean", "status": "max"}).reset_index()
        events.drop(["year", "month", "day", "hour", "minute"], axis = 1, inplace = True)
        events = events.sort_values(by = ["line_code", "vehicle", "event_timestamp"])
        events["last_eventtimestamp"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["event_timestamp"].shift(1), np.datetime64('NaT'))
        events["last_id"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["id"].shift(1), np.nan)
        events["last_name_norm"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["name_norm"].shift(1), np.nan)
        events.query("id != last_id", inplace = True)
        events["itinerary_id"] = events.apply(lambda row: self.search_itinerary(row["id"], row["last_id"]), axis = 1)
        events.rename(columns = {"latitude_y": "latitude", "longitude_y": "longitude"}, inplace = True)

        waiting_time = aux.query("distance <= 250").sort_values(by = ["line_code", "vehicle", "event_timestamp"])
        waiting_time["year"] = waiting_time["event_timestamp"].dt.year
        waiting_time["month"] = waiting_time["event_timestamp"].dt.month
        waiting_time["day"] = waiting_time["event_timestamp"].dt.day
        waiting_time["hour"] = waiting_time["event_timestamp"].dt.hour
        waiting_time["minute"] = time_window * waiting_time["event_timestamp"].dt.minute.floordiv(time_window)
        waiting_time = waiting_time.groupby(by = ["line_code", "vehicle", "id", "name_norm", "year", "month", "day", "hour", "minute", "latitude_y", "longitude_y", "status"]).agg({"event_timestamp": "mean"}).reset_index()
        waiting_time.drop(["year", "month", "day", "hour", "minute"], axis = 1, inplace = True)
        waiting_time.sort_values(by = ["line_code", "vehicle", "event_timestamp"], inplace = True)
        waiting_time["last_id"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(1)), waiting_time["id"].shift(1), np.nan)
        waiting_time["last_name_norm"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(1)), waiting_time["name_norm"].shift(1), np.nan)
        waiting_time["last_status"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(1)), waiting_time["status"].shift(1), np.nan)
        waiting_time["last_eventtimestamp"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(1)), waiting_time["event_timestamp"].shift(1), np.datetime64('NaT'))
        waiting_time["next_id"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(-1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(-1)), waiting_time["id"].shift(-1), np.nan)
        waiting_time["next_name_norm"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(-1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(-1)), waiting_time["name_norm"].shift(-1), np.nan)
        waiting_time["next_status"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(-1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(-1)), waiting_time["status"].shift(-1), np.nan)
        waiting_time["next_eventtimestamp"] = np.where((waiting_time["line_code"] == waiting_time["line_code"].shift(-1)) & (waiting_time["vehicle"] == waiting_time["vehicle"].shift(-1)), waiting_time["event_timestamp"].shift(-1), np.datetime64('NaT'))
        waiting_time["delta_t"] = (waiting_time["next_eventtimestamp"] - waiting_time["last_eventtimestamp"]).astype('timedelta64[s]')

        # ---------------------------------------------------------
        # Passar filtro de itinerario para atenuar ruído
        # ---------------------------------------------------------
        itinerary_probability = events.groupby(by = ["line_code", "vehicle", "itinerary_id"]).agg({"id": "count"}).reset_index()
        itinerary_probability["%"] = itinerary_probability["id"] / itinerary_probability.groupby(by = ["line_code", "vehicle"])["id"].transform("sum")
        counts, bins = np.histogram(itinerary_probability["%"])
        threshold = bins[len(bins) - 2]

        # ---------------------------------------------------------
        # Estimar os parâmetros das variáveis aleatórias
        # ---------------------------------------------------------
        vehicle_pdfs = self.vehicles.query("line_code == @linha")[{"line_code", "vehicle"}].drop_duplicates()
        vehicle_pdfs["speed_pdf"] = vehicle_pdfs.apply(lambda row: self.fit_speed_pdf(row["line_code"], row["vehicle"]), axis = 1)
        vehicle_pdfs["waiting_time_pdf"] = vehicle_pdfs.apply(lambda row: self.fit_waiting_time_pdf(row["line_code"], row["vehicle"], waiting_time), axis = 1)
        vehicle_pdfs["apriori_bus_stop"] = vehicle_pdfs.apply(lambda row: self.calculate_apriori_bus_stop(row["line_code"], row["vehicle"], events), axis = 1)

        events["itinerary_id"] = events.apply(lambda row: self.filter_itinerary(row["line_code"], row["vehicle"], row["itinerary_id"], threshold, itinerary_probability), axis = 1)
        events = pd.merge(events, self.bus_stops[{"line_code", "id", "itinerary_id", "seq", "max_seq"}], on = ["line_code", "id", "itinerary_id"], how = "left")
        events = pd.merge(events, vehicle_pdfs[{"line_code", "vehicle", "speed_pdf", "waiting_time_pdf", "apriori_bus_stop"}], on = ["line_code", "vehicle"], how = "inner")

        # ---------------------------------------------------------
        # Filtros de logs válidos
        # ---------------------------------------------------------
        events["last_seq"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["seq"].shift(1), np.nan)
        events.query("not (itinerary_id.notnull() and seq.isnull())", inplace = True, engine = "python")
        events.query("not (seq.notnull() and last_seq.isnull())", inplace = True, engine = "python")
        events.query("speed_pdf.notna() & waiting_time_pdf.notna()", inplace = True)
        events.drop(["last_seq", "seq", "max_seq"], axis = 1, inplace = True)

        # ---------------------------------------------------------
        # Reordenar logs válidos
        # ---------------------------------------------------------
        events = events.sort_values(by = ["line_code", "vehicle", "event_timestamp"])
        events["last_eventtimestamp"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["event_timestamp"].shift(1), np.datetime64('NaT'))
        events["last_id"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["id"].shift(1), np.nan)
        events["last_name_norm"] = np.where((events["line_code"] == events["line_code"].shift(1)) & (events["vehicle"] == events["vehicle"].shift(1)), events["name_norm"].shift(1), np.nan)
        events.query("id != last_id", inplace = True)
        events["itinerary_id"] = events.apply(lambda row: self.search_itinerary(row["id"], row["last_id"]), axis = 1)
        events["itinerary_id"] = events.apply(lambda row: self.filter_itinerary(row["line_code"], row["vehicle"], row["itinerary_id"], threshold, itinerary_probability), axis = 1)
        events = pd.merge(events, self.bus_stops[{"line_code", "id", "itinerary_id", "seq", "next_stop_id", "next_stop_delta_s"}], on = ["line_code", "id", "itinerary_id"], how = "left")

        # ---------------------------------------------------------
        # Estimar o timestamp da próxima parada
        # ---------------------------------------------------------
        events["estimated_next_delta_t"] = events.apply(lambda row: self.estimate_next_delta_t(row["next_stop_delta_s"], row["speed_pdf"], row["waiting_time_pdf"], row["apriori_bus_stop"]), axis = 1)
        events["estimated_next_eventtimestamp"] = events["event_timestamp"] + pd.to_timedelta(events["estimated_next_delta_t"], unit = 's')

        events["generated"] = False
        events = self.recover_data(1, 100, events)

        events.sort_values(by = ["line_code", "vehicle", "event_timestamp"]).to_csv("out.csv")
        self.bus_stops.to_csv("bus_stops.csv")