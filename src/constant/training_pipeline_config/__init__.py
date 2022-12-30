import os
PIPELINE_NAME='finance-complaint'
PIPELINE_ARTIFACT_DIR=os.path.join(os.getcwd(),'finance-artifact')
from src.constant.training_pipeline_config.data_ingestion import *