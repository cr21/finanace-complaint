from dataclasses import dataclass


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

