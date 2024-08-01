USE busanalysis_etl;
TRUNCATE etl_dim_bus_stop;
INSERT INTO etl_dim_bus_stop
SELECT
	id
	,name_norm
    ,AVG(latitude) AS 'latitude'
    ,AVG(longitude) AS 'longitude'
    ,MAX(type) AS type
FROM etl_itinerary
GROUP BY id, name_norm;

INSERT INTO busanalysis_dw.dim_bus_stop(name_norm, latitude, longitude, type)
SELECT
	etl_dim.name_norm
    ,etl_dim.latitude
    ,etl_dim.longitude
    ,etl_dim.type
FROM etl_dim_bus_stop AS etl_dim
LEFT JOIN busanalysis_dw.dim_bus_stop AS dim ON etl_dim.name_norm = dim.name_norm
WHERE dim.name_norm IS NULL;

UPDATE busanalysis_dw.dim_bus_stop AS dim
INNER JOIN etl_dim_bus_stop AS etl_dim ON dim.name_norm = etl_dim.name_norm
SET
	dim.latitude = etl_dim.latitude
    ,dim.longitude = etl_dim.longitude
    ,dim.type = etl_dim.type
    ,dim.last_update = current_timestamp()
WHERE
	dim.latitude <> etl_dim.latitude
    OR dim.longitude <> etl_dim.longitude
    OR dim.type <> etl_dim.type