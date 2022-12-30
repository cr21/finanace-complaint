
from src.config.pipeline.training import FinanceConfig
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.component.training.data_ingestion import DataIngestion
from src.exception import FinanceException
import sys

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

    def start(self):
        try:
            
            data_ingest_artifact = self.initiate_data_ingestion()
            return data_ingest_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)
