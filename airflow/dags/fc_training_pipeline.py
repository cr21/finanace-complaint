from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os

from airflow import DAG
training_pipeline=None

from airflow.operators.python import PythonOperator

with DAG(dag_id='finance_complaint',
        default_args={'retiers':2},
        description="Spark ML pipeline Scheduler",
        schedule_interval='@weekly',
        start_date=pendulum.datetime(2023,1, 5, tz="UTC"),
        catchup=False,
        tags=['finance_complaint'],
) as dag:
    dag.doc_md = __doc__
