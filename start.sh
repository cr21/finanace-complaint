#!bin/sh
nohup airflow scheduler &
airflow webserver -p 8081