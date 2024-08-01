USE busanalysis_etl;

TRUNCATE TABLE etl_event;

LOAD DATA INFILE '/var/lib/mysql-files/etl_event.csv'
INTO TABLE etl_event
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

TRUNCATE TABLE etl_itinerary;

LOAD DATA INFILE '/var/lib/mysql-files/etl_itinerary.csv'
INTO TABLE etl_itinerary
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(line_code,id,name,latitude,longitude,type,itinerary_id,line_way,@next_stop_id,@next_stop_delta_s,seq)
SET 
	next_stop_id = NULLIF(@next_stop_id,''),
    next_stop_delta_s = NULLIF(@next_stop_delta_s,'');
    
TRUNCATE TABLE etl_line;

LOAD DATA INFILE '/var/lib/mysql-files/etl_line.csv'
INTO TABLE etl_line
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;