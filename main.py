import os
from src.logger import logger
from src.exception import FinanceException
from src.pipeline.training import TrainingPipeline
from src.config.pipeline.training import FinanceConfig
import sys

def start_training(start=True):
    try:

        # if not start:
        #     return None
        TrainingPipeline(FinanceConfig()).start()
    except Exception as exp:
        raise FinanceException(exp, sys)


def main(training_status=False, prediction_status=True):
    try:
        start_training(start=training_status)
    except Exception as e:
        raise FinanceException(e, sys)

main()