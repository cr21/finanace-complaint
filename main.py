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
        # print("Training Running")
        TrainingPipeline(FinanceConfig()).start()
    except Exception as exp:
        raise FinanceException(exp, sys)


def main(training_status=False, prediction_status=True):
    try:
        print("Start training")
        start_training(start=training_status)
    except Exception as e:
        raise FinanceException(e, sys)

main()