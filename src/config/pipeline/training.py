
from src.constant.training_pipeline_config import * 
from src.constant import TIMESTAMP
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.entity.metadata_entity import DataIngestionMetadata
from src.logger import logger
import os, sys
import requests
import json
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