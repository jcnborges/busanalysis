SELECT * FROM etl_dim_bus_stop
WHERE id IN
(
	SELECT id AS qtd FROM etl_dim_bus_stop
	GROUP BY id
	HAVING count(id) > 1
);

SELECT * FROM busanalysis_dw.dim_bus_stop
ORDER BY last_update DESC
