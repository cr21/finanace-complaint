from dataclasses import dataclass
from typing import List
from src.constant.prediction_pipeline_config.file_config import ARCHIVE_DIR, INPUT_DIR, FAILED_DIR, \
    PREDICTION_DIR, REGION_NAME

@dataclass
class DataTransformationConfig:
    file_name:str
    exported_pipeline_dir:str
    transformed_train_dir:str
    transformed_test_dir:str
    test_size:float


@dataclass
class TrainingPipelineConfig:
    pipeline_name:str
    artifact_dir:str

    
@dataclass
class DataIngestionConfig:
    from_date:str 
    to_date:str
    data_ingestion_dir:str
    download_dir:str
    file_name:str
    feature_store_dir:str
    failed_dir:str
    metadata_file_path:str
    datasource_url:str

@dataclass
class DataIngestionMetaDataInfo:
    from_date:str
    to_date:str
    metadata_file_path:str

@dataclass
class DataValidationConfig:
    accepted_data_dir:str
    rejected_data_dir:str
    file_name:str

@dataclass
class ModelTrainerConfig:
    base_accuracy:float
    label_indexer_model_dir:str
    metric_list:List[str]
    trained_model_file_path:str

@dataclass
class ModelEvaluationConfig:
    model_evaluation_report_file_path:str
    threshold:float
    metric_list:List[str]
    model_dir:str
    bucket_name:str


@dataclass
class ModelPusherConfig:
    model_dir:str
    bucket_name:str



class PredictionPipelineConfig:

    def __init__(self, input_dir=INPUT_DIR,
                 prediction_dir=PREDICTION_DIR,
                 failed_dir=FAILED_DIR,
                 archive_dir=ARCHIVE_DIR,
                 region_name=REGION_NAME
                 ):
        self.input_dir = input_dir
        self.prediction_dir = prediction_dir
        self.failed_dir = failed_dir
        self.archive_dir = archive_dir
        self.region_name = region_name

    def to_dict(self):
        return self.__dict__

