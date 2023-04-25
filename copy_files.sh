#!/bin/bash

sudo rm /var/lib/mysql-files/etl_event.csv
sudo rm /var/lib/mysql-files/etl_itinerary.csv

sudo cp $1/etl_event.csv /var/lib/mysql-files/
sudo cp $1/etl_itinerary.csv /var/lib/mysql-files/
