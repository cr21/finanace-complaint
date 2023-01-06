from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os

from airflow import DAG
training_pipeline=None

from airflow.operators.python import PythonOperator
   # You can override them on a per-task basis during operator initialization
with DAG('finance_complaint',
    default_args={'retries': 2},
    # [END default_args]
    description='Machine learning Spark Project',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2023,1, 6, tz="UTC"),
    catchup=False,
    tags=['finance_complaint'],
) as dag:
    # [END instantiate_dag]
    # [START documentation]
        dag.doc_md = __doc__
    # [END documentation]

    # [START extract_function]


        from src.pipeline.training import  TrainingPipeline
        from src.config.pipeline.training import FinanceConfig
        training_pipeline= TrainingPipeline(FinanceConfig())
        print(training_pipeline)

        def data_ingestion(**kwargs):
                print("data_ingestion run")
                from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,\
                ModelTrainingArtifact,ModelEvaluationArtifact,ModelPusherArtifact,PartialModelTrainerRefArtifact,PartialModelTrainerMetricArtifact
                
                ti = kwargs['ti']
                data_ingestion_artifact = training_pipeline.initiate_data_ingestion()
                print(data_ingestion_artifact)
                ti.xcom_push('data_ingestion_artifact',data_ingestion_artifact)


        def data_validation(**kwargs):
                from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,\
                ModelTrainingArtifact,ModelEvaluationArtifact,ModelPusherArtifact,PartialModelTrainerRefArtifact,PartialModelTrainerMetricArtifact
                
                ti = kwargs['ti']
                data_ingestion_artifact =ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")
                print(data_ingestion_artifact)
                data_ingestion_artifact = DataIngestionArtifact(*(data_ingestion_artifact.__dict__))
                data_valid_artifact = training_pipeline.initiate_data_validation(dataingest_artifact=data_ingestion_artifact)
                ti.xcom_push('data_validation_artifact',data_valid_artifact)

        def data_transformation(**kwargs):
                from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,\
                ModelTrainingArtifact,ModelEvaluationArtifact,ModelPusherArtifact,PartialModelTrainerRefArtifact,PartialModelTrainerMetricArtifact
                ti  = kwargs['ti']

                data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")
                data_ingestion_artifact=DataIngestionArtifact(*(data_ingestion_artifact.__dict__))

                data_validation_artifact = ti.xcom_pull(task_ids="data_validation",key="data_validation_artifact")
                data_validation_artifact=DataValidationArtifact(*(data_validation_artifact.__dict__))
                data_transformation_artifact=training_pipeline.initiate_data_transformation(
                data_validation_artifact=data_validation_artifact
                )
                ti.xcom_push('data_transformation_artifact', data_transformation_artifact)

        def model_trainer(**kwargs):
                from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,\
                ModelTrainingArtifact,ModelEvaluationArtifact,ModelPusherArtifact,PartialModelTrainerRefArtifact,PartialModelTrainerMetricArtifact
                
                ti  = kwargs['ti']

                data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation",key="data_transformation_artifact")
                data_transformation_artifact=DataTransformationArtifact(*(data_transformation_artifact.__dict__))

                model_trainer_artifact=training_pipeline.initiate_model_training(data_transformation_artifact=data_transformation_artifact)

                ti.xcom_push('model_trainer_artifact', model_trainer_artifact.__dict__)

        def model_evaluation(**kwargs):
                from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,\
                ModelTrainingArtifact,ModelEvaluationArtifact,ModelPusherArtifact,PartialModelTrainerRefArtifact,PartialModelTrainerMetricArtifact
                
                ti  = kwargs['ti']
                data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")
                data_ingestion_artifact=DataIngestionArtifact(*(data_ingestion_artifact.__dict__))

                data_validation_artifact = ti.xcom_pull(task_ids="data_validation",key="data_validation_artifact")
                data_validation_artifact=DataValidationArtifact(*(data_validation_artifact.__dict__))

                model_trainer_artifact = ti.xcom_pull(task_ids="model_trainer",key="model_trainer_artifact")
                print(model_trainer_artifact)
                model_trainer_artifact=ModelTrainingArtifact(*(model_trainer_artifact.__dict__))

                model_evaluation_artifact = training_pipeline.initiate_model_evaluation(data_validation_artifact=data_validation_artifact,
                                                                        model_trainer_artifact=model_trainer_artifact)

        
                ti.xcom_push('model_evaluation_artifact', model_evaluation_artifact.__dict__)

        def push_model(**kwargs):
                from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,\
                ModelTrainingArtifact,ModelEvaluationArtifact,ModelPusherArtifact,PartialModelTrainerRefArtifact,PartialModelTrainerMetricArtifact
                
                ti  = kwargs['ti']
                model_evaluation_artifact = ti.xcom_pull(task_ids="model_evaluation",key="model_evaluation_artifact")
                model_evaluation_artifact=ModelEvaluationArtifact(*(model_evaluation_artifact.__dict__))
                model_trainer_artifact = ti.xcom_pull(task_ids="model_trainer",key="model_trainer_artifact")
                model_trainer_artifact=ModelTrainingArtifact(*(model_trainer_artifact.__dict__))

                if model_evaluation_artifact.model_accepted:
                        model_pusher_artifact = training_pipeline.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
                        print(f'Model pusher artifact: {model_pusher_artifact}')
                else:
                        print("Trained model rejected.")
                        print("Trained model rejected.")
                        print("Training pipeline completed")




        data_ingestion = PythonOperator(
                task_id='data_ingestion',
                python_callable=data_ingestion,
        )
        data_ingestion.doc_md = dedent(
                """\
                        #### Extract task
                        A simple Extract task to get data ready for the rest of the data pipeline.
                        In this case, getting data is simulated by reading from a hardcoded JSON string.
                        This data is then put into xcom, so that it can be processed by the next task.
                """
        )

        data_validation = PythonOperator(
                task_id="data_validation",
                python_callable=data_validation

        )

        data_transformation = PythonOperator(
                task_id ="data_transformation",
                python_callable=data_transformation
        )

        model_trainer = PythonOperator(
                task_id="model_trainer", 
                python_callable=model_trainer

        )

        model_evaluation = PythonOperator(
                task_id="model_evaluation", python_callable=model_evaluation
        )   

        push_model  =PythonOperator(
            task_id="push_model",
            python_callable=push_model
        )

        data_ingestion >> data_validation >> data_transformation >> model_trainer >> model_evaluation >> push_model
