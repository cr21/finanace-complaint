
import os
from dataclasses import dataclass
from src.entity.config_entity import DataIngestionMetaDataInfo
from src.utils import read_yaml_file, write_yaml_file
from  src.exception import FinanceException
import sys
from src.logger import logger
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    metadata_file_path:str
    download_dir:str


class DataIngestionMetadata:
    def __init__(self, meta_data_file_path) -> None:
        self.metadata_path = meta_data_file_path

    @property
    def is_metadata_file_present(self)->bool:
        return os.path.exists(self.metadata_path)

    def write_meta_info(self, from_date:str, to_date:str, file_path:str=None):
        try:
            metadata_info = DataIngestionMetaDataInfo(
                from_date=from_date,
                to_date=to_date,
                metadata_file_path=self.metadata_path
            )
            
            write_yaml_file(file_path=self.metadata_path, data=metadata_info.__dict__)

        except Exception as e:
            raise FinanceException(e, sys)

    def get_meta_data_info(self):
        try:
            if not self.is_metadata_file_present:
                raise Exception("No metadata file available")
            metadata = read_yaml_file(self.metadata_file_path)
            metadata_info = DataIngestionMetaDataInfo(**(metadata))
            logger.info(metadata)
            return metadata_info
        except Exception as e:
            raise FinanceException(e, sys)

