USE busanalysis_etl;

TRUNCATE TABLE etl_event;

LOAD DATA INFILE '/var/lib/mysql-files/etl_event.csv'
INTO TABLE etl_event
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
#IGNORE 1 ROWS;

TRUNCATE TABLE etl_itinerary;

LOAD DATA INFILE '/var/lib/mysql-files/etl_itinerary.csv'
INTO TABLE etl_itinerary
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;