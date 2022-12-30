from dataclasses import dataclass
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataIngestionMetadata
from datetime import datetime
import pandas as pd
import sys
from src.exception import FinanceException
from src.logger import logger
import os
import time
import requests
import uuid
import json
import re
from src.config.spark_manager import spark_session
from typing import List

@dataclass
class DownloadURL:
    url:str
    file_path:str
    n_retry:int

class DataIngestion:

    def __init__(self, dataingest_config:DataIngestionConfig, n_retry:int=5) -> None:
        self.config = dataingest_config
        self.n_retry=n_retry
        self.failed_download_urls: List[DownloadURL] = []

    def get_required_interval(self):
        start_date  =datetime.strptime(self.config.from_date,'%Y-%m-%d')
        end_date = datetime.strptime(self.config.to_date,'%Y-%m-%d')
        n_diff_days  = (end_date-start_date).days
        freq=None

        if n_diff_days > 365:
            freq="Y"
        if n_diff_days > 30:
            freq="M"
        if n_diff_days > 7:
            freq="W"


        if freq is None:
            intervals = pd.date_range(start=self.config.from_date,
                                    end = self.config.to_date,
                                    periods=2).astype('str').tolist()
        else:
            intervals = pd.date_range(start=self.config.from_date,
                                    end = self.config.to_date,
                                    freq=freq).astype('str').tolist()
                                        
            logger.info(f"Prepared interval {intervals}")
        if self.config.to_date not in intervals:
            intervals.append(self.config.to_date)
        return intervals 

    def download_files(self, n_day_interval_url:int=None):
        """
        n_month_interval_url: if not provided then information default value will be set
        =======================================================================================
        returns: List of DownloadUrl = namedtuple("DownloadUrl", ["url", "file_path", "n_retry"])
        """
        try:
            logger.info("Start Downloading files")
            required_intervals = self.get_required_interval()

            for index in range(1, len(required_intervals)):
                from_date  = required_intervals[index-1]
                to_date = required_intervals[index]
                logger.info(f"Generating Data Download url from : {from_date} to  {to_date}")

                datasource_url:str = self.config.datasource_url
                url = datasource_url.replace('<fromdate>', from_date)
                url = url.replace('<todate>', to_date)
                logger.debug(f"Url: {url}")
                # prepare url, file_path, file_dir
                file_name = f'{self.config.file_name}_{from_date}_{to_date}.json'
                file_path = os.path.join(self.config.download_dir, file_name)
                download_url = DownloadURL(url=url, file_path=file_path, n_retry=self.n_retry)
                
                # download data from url
                self.download_data(download_url=download_url)

            logger.info("file download finished")
        
        except Exception as exp:
            raise FinanceException(exp, sys)
            



    def convert_to_parquet_file(self)->str:
        """
        downloaded files will be converted and merged into single parquet file
        json_data_dir: downloaded json file directory
        data_dir: converted and combined file will be generated in data_dir
        output_file_name: output file name 
        =======================================================================================
        returns output_file_path
        """
        try:
            json_data_dir:str = self.config.download_dir
            data_dir:str = self.config.feature_store_dir
            output_file_name:str=self.config.file_name
            os.makedirs(data_dir, exist_ok=True)
            file_path = os.path.join(data_dir, f"{output_file_name}")
            logger.info(f"Parquet file will be created at: {file_path}")
            if not os.path.exists(json_data_dir):
                return file_path

            for file_name in os.listdir(json_data_dir):
                json_file_path = os.path.join(json_data_dir, file_name)
                logger.debug(f"Converting {json_file_path} into parquet format at {file_path}")
                df = spark_session.read.json(json_file_path)
                if df.count()>0:
                    df.write.mode('append').parquet(file_path)

            return file_path

            
        except Exception as exp:
            raise FinanceException(exp, sys)


    def retry_download_data(self, data, download_url:DownloadURL):
        """
        This function help to avoid failure as it help to download failed file again
        
        data:failed response
        download_url: DownloadUrl
        """
        try:
            if download_url.n_retry == 0:
                # we reached download limit not store in failed dir
                self.failed_download_urls.append(download_url)
                logger.info(f"Unable to Download File {download_url}")
                return

            # need to handle throatling requst wait for sometime before hitting request again
            content = data.content.decode('utf-8')
            wait_second=re.findall(r'\d+', content)
            if len(wait_second) > 0:
                time.sleep(int(wait_second[0])+2)
            # write response to keep track of errors and exception for failure
            failed_file_path = os.path.join(self.config.failed_dir, os.path.basename(download_url.file_path))
            os.makedirs(self.config.failed_dir,exist_ok=True)
            with open(failed_file_path,'w') as file_obj:
                file_obj.write(data.content)

            # calling download_url with n_rety changes
            download_url = DownloadURL(download_url.url, file_path=download_url.file_path, n_retry=download_url.n_retry-1)
            self.download_data(download_url=download_url)

        except Exception as exp:
            raise FinanceException(exp, sys)



    def download_data(self,download_url:DownloadURL):
        try:
            logger.info(f"Starting download operation: {download_url}")
            # create all dirs
            file_path = download_url.file_path
            n_retry=download_url.n_retry
            url=download_url.url
            
            download_dir = os.path.dirname(file_path)
            os.makedirs(download_dir, exist_ok=True)
            # download data

            data = requests.get(url,params={'User-agent': f'your bot {uuid.uuid4()}'})

            # writing data to json file
            try:
                logger.info(f"Started writing downloaded data into json file: {download_url.file_path}")
                with open(file_path,'w') as file_obj:
                    # finance_complaint_data = list(  map(lambda x:x["_source"],
                    #                                                 filter(lambda x:"_source" in x.keys(),
                    #                                                 json.loads(data.content)
                    #                                                 )
                    #                                 )
                    #                             )

                    finance_complaint_data = list(map(lambda x: x["_source"],
                                                      filter(lambda x: "_source" in x.keys(),
                                                             json.loads(data.content)))
                                                  )
                    json.dump(finance_complaint_data, file_obj)
                logger.info(f"Downloaded data has been written into file: {download_url.file_path}")
            except Exception as exp:
                logger.info("Failed to download hence retry again.")
                # removing file failed file exist
                if os.path.exists(download_url.file_path):
                    os.remove(download_url.file_path)
                self.retry_download_data(data, download_url=download_url)

        except Exception as exp:
            raise exp

    def write_metadata(self, file_path:str)->None:
        """
        This function help us to update metadata information 
        so that we can avoid redundant download and merging.

        """
        try:
            logger.info(f"Writing metadata info into metadata file.")
            metadata_info = DataIngestionMetadata(meta_data_file_path=self.config.metadata_file_path)
            metadata_info.write_meta_info(
                                        from_date=self.config.from_date,
                                        to_date=self.config.to_date,
                                        file_path=file_path
                                    )
            logger.info(f"Metadata has been written.")


        except Exception as exp:
            raise FinanceException(exp, sys)



    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logger.info("Start DataIngestion")

            #1. check if from date and to date are not same 
            if self.config.from_date != self.config.to_date:
                self.download_files()

            if os.path.exists(self.config.download_dir):
                logger.info(f"converting all downloaded file in download directory {self.config.download_dir} to parquet file")
                file_path = self.convert_to_parquet_file()
                self.write_metadata(file_path=file_path)

            feature_store_file_path = os.path.join(self.config.feature_store_dir,
                                                   self.config.file_name)
            artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
                download_dir=self.config.download_dir,
                metadata_file_path=self.config.metadata_file_path,
            )
            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact


        
        except Exception as exp:
            raise FinanceException(exp, sys)

        