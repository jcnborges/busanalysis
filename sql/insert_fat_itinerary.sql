USE busanalysis_etl;
TRUNCATE etl_fat_itinerary;
INSERT INTO etl_fat_itinerary
SELECT
	line_code
	,id
    ,next_stop_id
    ,next_stop_delta_s
    ,itinerary_id
    ,seq
    ,line_way
FROM etl_itinerary;

INSERT INTO busanalysis_dw.fat_itinerary(line_code, bus_stop_id, next_bus_stop_id, next_bus_stop_delta_s, itinerary_id, seq, line_way)
SELECT
	line_code
    ,dim.id AS bus_stop_id
    ,dim_next.id AS next_bus_stop_id
    ,next_stop_delta_s
    ,itinerary_id
    ,seq
    ,line_way
FROM etl_fat_itinerary AS etl_fat
INNER JOIN etl_dim_bus_stop AS etl_dim ON etl_fat.id = etl_dim.id
INNER JOIN busanalysis_dw.dim_bus_stop AS dim ON dim.name_norm = etl_dim.name_norm
LEFT JOIN etl_dim_bus_stop AS etl_dim_next ON etl_fat.next_stop_id = etl_dim_next.id
LEFT JOIN busanalysis_dw.dim_bus_stop AS dim_next ON dim_next.name_norm = etl_dim_next.name_norm