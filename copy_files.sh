#!/bin/bash

sudo rm /var/lib/mysql-files/etl_event.csv
sudo rm /var/lib/mysql-files/etl_itinerary.csv

sudo cp etl_event.csv /var/lib/mysql-files/
sudo cp etl_itinerary.csv /var/lib/mysql-files/
