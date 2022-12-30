
from src.config.pipeline.training import FinanceConfig
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.component.training.data_ingestion import DataIngestion
from src.component.training.data_validation import DataValidation
from src.exception import FinanceException
import sys
from src.logger import logger,LOG_FILE_PATH

class TrainingPipeline:

    def __init__(self, config:FinanceConfig) -> None:
        self.config = config

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:

            data_ingest_config = self.config.get_data_ingestion_pipeline()
            data_ingest = DataIngestion(dataingest_config=data_ingest_config,n_retry=5)
            data_ingest_artifact = data_ingest.initiate_data_ingestion()
            return data_ingest_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)


    def initiate_data_validation(self, dataingest_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_valid_config:DataValidationConfig = self.config.get_data_validation_config()
            data_validator = DataValidation(valid_config=data_valid_config, ingest_artifact=dataingest_artifact)
            data_validation_artifact:DataValidationArtifact = data_validator.initiate_data_processing()
            return data_validation_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)

    def copy_logger_to_dump(self)->None:
        
        contents = None
        with open(LOG_FILE_PATH,'r') as file1:
            contents = file1.readlines()
            with open(LOG_FILE_PATH+'_dump.txt','w') as f:
                for line in contents:
                    f.write(line)
        


    def start(self):
        try:
            
            data_ingest_artifact = self.initiate_data_ingestion()
            data_validation_artifact=self.initiate_data_validation(data_ingest_artifact)
            self.copy_logger_to_dump()
            return data_validation_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)
