
from src.constant.training_pipeline_config import * 
from src.constant import TIMESTAMP
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, \
    DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig, ModelPusherConfig
from src.entity.metadata_entity import DataIngestionMetadata
from src.logger import logger
import os, sys
import requests
from src.exception import FinanceException
import json
from src.constant.model import S3_MODEL_BUCKET_NAME,S3_MODEL_DIR_KEY
from datetime import datetime


class FinanceConfig:
    def __init__(self,pipeline_name=PIPELINE_NAME, timestamp=TIMESTAMP) -> None:
        self.pipeline=pipeline_name
        self.config = self.get_config_pipe()
        self.timestamp = timestamp

    def get_config_pipe(self)->None:
        """
        return pipeline config
        PipelineConfig(pipeline_name, artifact_dir)
        
        """
        artifact_dir = PIPELINE_ARTIFACT_DIR
        pipeline_config = TrainingPipelineConfig(pipeline_name=self.pipeline, artifact_dir=artifact_dir)
        logger.info("Training Pipeline config: {pipeline_config}")
        self.pipeline_config=pipeline_config
        return pipeline_config

    def get_data_ingestion_pipeline(self, from_date=DATA_INGESTION_MIN_START_DATE,to_date=None)-> DataIngestionConfig:
        """
        prepare dataingestion configuration options
        """

        # check if metadatafile exists or not 
        min_start_date = datetime.strptime(DATA_INGESTION_MIN_START_DATE, "%Y-%m-%d")
        from_date_obj =  datetime.strptime(from_date,'%Y-%m-%d')
        if from_date_obj < min_start_date:
            from_date = DATA_INGESTION_MIN_START_DATE
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        """
        Master Directory for storing downloaded file, sothat muliple files are not downloaded
        
        """
        data_ingestion_master_dir = os.path.join(self.config.artifact_dir, DATA_INGESTION_DIR)
        # timebased directory
        data_ingestion_dir = os.path.join(data_ingestion_master_dir, self.timestamp)

        metadata_file_path =  os.path.join(data_ingestion_master_dir, DATA_INGESTION_METADATA_FILE_NAME)
        data_ingestion_meta_data = DataIngestionMetadata(metadata_path=metadata_file_path)

        # check if metadata file already exists or not if already exists
        if data_ingestion_meta_data.is_metadata_file_present:
            from_date = data_ingestion_meta_data.get_metadata_info().to_date

        data_ingestion_config = DataIngestionConfig(  
                                                    from_date=from_date, 
                                                    to_date=to_date,
                                                    data_ingestion_dir=data_ingestion_dir,
                                                    download_dir=os.path.join(data_ingestion_dir, DATA_INGESTION_DOWNLOADED_DATA_DIR),
                                                    file_name=DATA_INGESTION_FILE_NAME,
                                                    feature_store_dir=os.path.join(data_ingestion_master_dir,DATA_INGESTION_FEATURE_STORE_DIR),
                                                    failed_dir=os.path.join(data_ingestion_dir, DATA_INGESTION_FAILED_DIR),
                                                    metadata_file_path=metadata_file_path,
                                                    datasource_url=DATA_INGESTION_DATA_SOURCE_URL
                                                )
        logger.info(f"Data ingestion config: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_transformation_config(self)->DataTransformationConfig:
        try:
            data_transformation_dir = os.path.join(self.pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR, self.timestamp)
            transformed_train_dir = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRAIN_DIR)
            transformed_test_dir=os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TEST_DIR)
            exported_pipeline_dir = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_PIPELINE_DIR)
            
            data_transform_config = DataTransformationConfig(file_name=DATA_TRANSFORMATION_FILE_NAME,
                                                            exported_pipeline_dir=exported_pipeline_dir,
                                                            transformed_train_dir=transformed_train_dir,
                                                            transformed_test_dir=transformed_test_dir,
                                                            test_size=DATA_TRANSFORMATION_TEST_SIZE)
            
            logger.info(f"Data_TransformationConfig {data_transform_config}")
            return data_transform_config

        except Exception as exp:
            raise FinanceException(exp, sys)

    def get_data_validation_config(self)-> DataValidationConfig:
        try:
            data_validation_dir = os.path.join(self.pipeline_config.artifact_dir, DATA_VALIDATION_DIR, self.timestamp) 
        
            dv_config = DataValidationConfig(
                                                accepted_data_dir=os.path.join(data_validation_dir,DATA_VALIDATION_ACCEPTED_DATA_DIR),
                                                rejected_data_dir=os.path.join(data_validation_dir,DATA_VALIDATION_REJECTED_DATA_DIR),
                                                file_name=DATA_VALIDATION_FILE_NAME
                                            )
            logger.info(f"Data Validation Config info: {dv_config}")
            return dv_config
        except Exception as exp:
            raise FinanceException(exp, sys)
        


    def get_model_training_config(self)-> ModelTrainerConfig:
        try:
            model_trainer_dir = os.path.join(self.pipeline_config.artifact_dir,MODEL_TRAINER_DIR, self.timestamp)
            trained_model_file_path = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_MODEL_NAME)
            label_indexer_model_dir = os.path.join(model_trainer_dir, MODEL_TRAINER_LABEL_INDEXER_DIR)
            model_trainer_config = ModelTrainerConfig(base_accuracy=MODEL_TRAINER_BASE_ACCURACY,
                                                     label_indexer_model_dir=label_indexer_model_dir,
                                                     metric_list=MODEL_TRAINER_MODEL_METRIC_NAMES,
                                                     trained_model_file_path=trained_model_file_path
                                                    )
            logger.info(f"Model trainer config: {model_trainer_config}")
            return model_trainer_config
            
        except Exception as exp:
            raise FinanceException(exp, sys)

    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        try:
            model_evaluation_dir = os.path.join(self.pipeline_config.artifact_dir,MODEL_EVALUATION_DIR,self.timestamp)
            model_evaluation_report_file_path=os.path.join(model_evaluation_dir,MODEL_EVALUATION_REPORT_DIR, MODEL_EVALUATION_REPORT_FILE_NAME)
            model_eval_config = ModelEvaluationConfig(
                                                        model_evaluation_report_file_path=model_evaluation_report_file_path,
                                                        threshold=MODEL_EVALUATION_THRESHOLD_VALUE,
                                                        metric_list=MODEL_EVALUATION_METRIC_NAMES,
                                                        model_dir=S3_MODEL_DIR_KEY,
                                                        bucket_name=S3_MODEL_BUCKET_NAME
                                                    )
            logger.info(f"Model evaluation config: [{model_eval_config}]")
            return model_eval_config
        except Exception as exp:
            raise FinanceException(exp, sys)

    def get_model_pusher_config(self)-> ModelPusherConfig:
        try:
            model_pusher_config = ModelPusherConfig(
                model_dir=S3_MODEL_DIR_KEY,
                bucket_name=S3_MODEL_BUCKET_NAME
            )
            logger.info(f"Model pusher config: {model_pusher_config}")
            return model_pusher_config
        except Exception as exp:
            raise FinanceException(exp, sys)
        