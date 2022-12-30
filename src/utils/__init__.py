import yaml
from src.exception import FinanceException
import os

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
    except FinanceException as exp:
        raise exp

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    
    """

    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except FinanceException as exp:
        raise exp
   
