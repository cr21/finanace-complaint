import yaml
from src.exception import FinanceException
import os
from pyspark.sql import DataFrame
from src.exception import FinanceException
from src.logger import logger
import sys
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def write_yaml_file(file_path:str, data:dict)-> None:
    """
    Create yaml file
    @params
        file_path : file_path
        data : dictionary
    
    """
    try:
        pass
        # check if file_path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as exp:
        raise FinanceException(exp, sys)

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    
    """

    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as exp:
        raise FinanceException(exp, sys)
   

def get_score(metric_name:str,dataframe:DataFrame,label_col:str,prediction_col:str):

    try:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=prediction_col,
            metricName=metric_name)
        score = evaluator.evaluate(dataframe)
        print(f"{metric_name} score: {score}")
        logger.info(f"{metric_name} score: {score}")
        return score
    except Exception as exp:
        raise FinanceException(exp, sys)