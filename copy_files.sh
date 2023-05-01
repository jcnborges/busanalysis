#!/bin/bash

sudo rm /var/lib/mysql-files/etl_event.csv
sudo rm /var/lib/mysql-files/etl_itinerary.csv
sudo rm /var/lib/mysql-files/etl_line.csv


sudo cp $1/etl_event.csv /var/lib/mysql-files/
sudo cp $1/etl_itinerary.csv /var/lib/mysql-files/
sudo cp $1/etl_line.csv /var/lib/mysql-files/

pwd=!jfzrpg93!

echo mysql -uroot --password=$pwd -D busanalysis_dw -e "source ./src/sql/bulk_insert.sql"
mysql -uroot --password=$pwd -D busanalysis_dw -e "source ./src/sql/bulk_insert.sql"

echo mysql -uroot --password=$pwd -D busanalysis_dw -e "call busanalysis_dw.sp_load_all('$1');"
mysql -uroot --password=$pwd -D busanalysis_dw -e "call busanalysis_dw.sp_load_all('$1');"
