from src.logger import logger
from src.exception import FinanceException
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelTrainingArtifact, ModelPusherArtifact
from src.ml.estimator import S3FinanceEstimator
import os, sys
class ModelPusher:
    def __init__(self, model_trainer_artifact:ModelTrainingArtifact, model_pusher_config:ModelPusherConfig) -> None:
        self.model_trainer_artifact=model_trainer_artifact
        self.model_pusher_config = model_pusher_config


    def push_model(self)->str:
        try:
            model_registry = S3FinanceEstimator(bucket_name=self.model_pusher_config.bucket_name,s3_key=self.model_pusher_config.model_dir)
            model_file_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
            model_registry.save(model_dir=os.path.dirname(model_file_path),
                                key=self.model_pusher_config.model_dir
                                )
            
            return model_registry.get_latest_model_path()
        except Exception as exp:
            raise FinanceException(exp, sys)

    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            pusher_dir = self.push_model()
            model_pusher_artifact = ModelPusherArtifact(model_pushed_dir=pusher_dir)
            logger.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)
    