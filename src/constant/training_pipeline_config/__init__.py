import os
PIPELINE_NAME='finance-complaint'
PIPELINE_ARTIFACT_DIR=os.path.join(os.getcwd(),'finance-artifact')
from src.constant.training_pipeline_config.data_ingestion import *
from src.constant.training_pipeline_config.data_validation import *
from src.constant.training_pipeline_config.data_transformation import *
from src.constant.training_pipeline_config.model_trainer import *
from src.constant.training_pipeline_config.model_evaluation import *