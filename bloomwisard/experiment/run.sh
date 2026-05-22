#!/bin/bash
echo "Fridge"
cd fridge
python overall_info.py

echo "Garage door"
cd ../garage_door
python overall_info.py

echo "GPS tracker"
cd ../gps_tracker
python overall_info.py

echo "Motion light"
cd ../motion_light
python overall_info.py

echo "Modbus"
cd ../modbus
python overall_info.py

echo "Thermostat"
cd ../thermostat
python overall_info.py

echo "Weather"
cd ../weather
python overall_info.py  