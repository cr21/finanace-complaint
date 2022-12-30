from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from typing import List, Dict
from src.entity.schema import FinanceDataSchema
import os, sys
from src.logger import logger,LOG_FILE_PATH
from src.exception import FinanceException
from src.config.spark_manager import spark_session
from pyspark.sql import DataFrame
from dataclasses import dataclass
from pyspark.sql.functions import lit

COMPLAINT_TABLE = "complaint"
ERROR_MESSAGE = "error_msg"

@dataclass
class MissingReport:
    total_row:int
    missing_row:int
    missing_percent:float

class DataValidation(FinanceDataSchema):

    def __init__(self,
                    valid_config:DataValidationConfig,
                    ingest_artifact:DataIngestionArtifact,
                    table_name:str=COMPLAINT_TABLE,
                    schema=FinanceDataSchema()
    
                ) -> None:
        try:
            super().__init__()
            self.valid_config = valid_config
            self.input_artifact = ingest_artifact
            self.table_name=table_name
            self.schema=schema
        except Exception as exp:
            raise FinanceException(exp, sys)

    def read_data(self)->DataFrame:
        try:
            dataframe = spark_session.read.parquet(self.input_artifact.feature_store_file_path)
                                        # limit(100000)
            logger.info(f"Data frame is created using file: {self.input_artifact.feature_store_file_path}")
            logger.info(f"Number of row: {dataframe.count()} and column: {len(dataframe.columns)}") 
            return dataframe
        except Exception as exp:
            raise FinanceException(exp, sys)

    def get_missing_report_data(self, dataframe:DataFrame) ->  Dict[str, MissingReport]:
        try:
            missing_report:Dict[str:MissingReport] = dict()
            logger.info("Getting Missing Count for all the features in dataframe")
            total_rows = dataframe.count()
            for column in dataframe.columns:
                missing_rows = dataframe.filter(f"{column} is null").count()
                missing_percentage= (missing_rows*100)/total_rows
                missing_report[column] = MissingReport( total_row=total_rows,
                                                        missing_row=missing_rows,
                                                        missing_percent=missing_percentage 
                                                    )
            logger.info(f"Missing Reports for each of the column: {missing_report} ")
            return missing_report
        except Exception as exp:
            raise FinanceException(exp, sys)
            

    def get_unwanted_and_high_missing_columns(self, dataframe:DataFrame, missing_value_threshold:float=0.2)-> List[str]:
        try:
            # 1. get Missing report data
            missing_report_data:Dict[str, MissingReport] = self.get_missing_report_data(dataframe)
            
            #2.  for all columns in missing report see the column statistics and 
            # get all columns violating minimum missing value threshold
            unwanted_cols :List[str] = self.schema.unwanted_columns
            # We have some set of features we don't want to drop even if it violated missing constraint
            for column in missing_report_data:
                if column not in self.skip_validation_features:
                    if missing_report_data[column].missing_percent > (missing_value_threshold*100):
                        unwanted_cols.append(column)
                        logger.info(f"Missing report {column}: [{missing_report_data[column]}]")
            #3. return violated columns
            unwanted_cols = list(set(unwanted_cols))
            return unwanted_cols
        except Exception as exp:
            raise FinanceException(exp, sys)


    def drop_unwanted_columns(self, dataframe:DataFrame)-> DataFrame:
        try:
            #1. get unwanted and high missing value columns
            unwanted_cols:List=self.get_unwanted_and_high_missing_columns(dataframe=dataframe)
            logger.info(f"Dropping feature: {','.join(unwanted_cols)}")
            #2. Write error files / unwanted data files  in rejected dirs
            #2.1 get all columns filtered out from dataframe to created rejected dataframe
            unwanted_df:DataFrame = dataframe.select(unwanted_cols)     
            unwanted_df = unwanted_df.withColumn('ERROR_MESSAGE',lit("Many column contains more than 20% missing values"))

            rejected_dir = os.path.join(self.valid_config.rejected_data_dir,"missing_data")
            os.makedirs(rejected_dir,exist_ok=True)
            file_path = os.path.join(rejected_dir, self.valid_config.file_name)
            logger.info(f"Writing dropped column into file: [{file_path}]")
            unwanted_df.write.mode("append").json(file_path)

            #3. Drop unwanted cols and return resultant dataframe
            dataframe: DataFrame = dataframe.drop(*unwanted_cols)
            logger.info(f"Remaining number of columns: [{dataframe.columns}]")
            return dataframe
        except Exception as exp:
            raise FinanceException(exp, sys)


    def is_required_col_exists(self, dataframe:DataFrame)->None:
        try:
            required_cols :List[str] = self.schema.required_columns
            columns = list(filter(lambda x:x in required_cols,dataframe.columns))

            if columns and required_cols and len(columns) != len(required_cols):
                raise Exception(f"Required column missing\n\
                 Expected columns: {required_cols}\n\
                 Found columns: {columns}\
                 ")
            
        except Exception as exp:
            raise FinanceException(exp, sys)

    def initiate_data_processing(self)->DataValidationArtifact:
        try:
             #1. Read data from feature_store dir in parquet format
            logger.info("Starting Data Processing ")
            dataframe:DataFrame = self.read_data()
            
            #2. Drop unwanted columns
            logger.info("Droppong unwanted columns")
            dataframe:DataFrame = self.drop_unwanted_columns(dataframe = dataframe)

            #3. Validation Check if all requrired columns  are present
            self.is_required_col_exists(dataframe)

            logger.info("Saving preprocessed data.")
            
            #4. prepare validation data artifact
            os.makedirs(os.path.join(self.valid_config.accepted_data_dir), exist_ok=True)
            accepted_file_path = os.path.join(  self.valid_config.accepted_data_dir,
                                                self.valid_config.file_name
                                                )

            dataframe.write.parquet(accepted_file_path)
            #5. return data validation data artifact
            artifact = DataValidationArtifact(accepted_file_path=accepted_file_path,
                                                rejected_dir=self.valid_config.rejected_data_dir
                                            )
            return artifact

        except Exception as exp:
            raise FinanceException(exp, sys)
       