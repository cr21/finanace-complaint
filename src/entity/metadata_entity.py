
import os
from src.entity.config_entity import DataIngestionMetaDataInfo
from src.utils import read_yaml_file, write_yaml_file
from src.logger import logger

class DataIngestionMetadata:

    def __init__(self, metadata_path) -> None:
        self.metadata_file_path=metadata_path

    @property
    def is_metadata_file_present(self)-> bool:
        return os.path.exists(self.metadata_file_path)

    def write_metadata_info(self, from_date:str, to_date:str, data_file_path:str):
        try:
            data_ingestion_meta_data_info = DataIngestionMetaDataInfo(
                                        from_date=from_date,
                                        to_date=to_date,
                                        metadata_file_path=data_file_path
                                    )

            write_yaml_file(data_file_path, data_ingestion_meta_data_info.__dict__)
            
        except Exception as exp:
            raise exp

    def get_metadata_info(self) -> DataIngestionMetaDataInfo :
        try:
            if not self.is_metadata_file_present:
                raise Exception("No metadata file available")

        
            data = read_yaml_file(file_path=self.metadata_file_path)

            metadata_info= DataIngestionMetaDataInfo(**(data))  
            logger.info(data)
            return metadata_info
        except Exception as exp:
            raise exp


