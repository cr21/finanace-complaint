
from src.config.pipeline.training import FinanceConfig
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainingArtifact,ModelEvaluationArtifact,PartialModelTrainerMetricArtifact, PartialModelTrainerRefArtifact, ModelPusherArtifact
from src.component.training.data_ingestion import DataIngestion
from src.component.training.data_validation import DataValidation
from src.component.training.data_transformer import DataTransformer
from src.component.training.model_trainer import ModelTrainer
from src.component.training.model_pusher import ModelPusher
# from src.com
from src.component.training.model_evaluation import ModelEvaluation
from src.exception import FinanceException
import sys
from src.logger import logger,LOG_FILE_PATH

class TrainingPipeline:

    def __init__(self, config:FinanceConfig) -> None:
        self.config = config

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:

            data_ingest_config = self.config.get_data_ingestion_pipeline()
            data_ingest = DataIngestion(dataingest_config=data_ingest_config,n_retry=5)
            data_ingest_artifact = data_ingest.initiate_data_ingestion()
            return data_ingest_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)


    def initiate_data_validation(self, dataingest_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_valid_config:DataValidationConfig = self.config.get_data_validation_config()
            data_validator = DataValidation(valid_config=data_valid_config, ingest_artifact=dataingest_artifact)
            data_validation_artifact:DataValidationArtifact = data_validator.initiate_data_processing()
            logger.info(f"Data Validation Artifact {data_validation_artifact}" )
            return data_validation_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)


    def initiate_data_transformation(self, data_valid_artifact:DataValidationArtifact)-> DataTransformationArtifact:
        try:
            data_transformation_config: DataTransformationConfig = self.config.get_data_transformation_config()
            data_transformer = DataTransformer(data_transformation_config, data_valid_artifact)
            data_transform_artifact=data_transformer.initiate_data_transformation()
            return data_transform_artifact

        except Exception as exp:
            raise FinanceException(exp, sys)

    def copy_logger_to_dump(self)->None:
        
        contents = None
        with open(LOG_FILE_PATH,'r') as file1:
            contents = file1.readlines()
            with open(LOG_FILE_PATH+'_dump.txt','w') as f:
                for line in contents:
                    f.write(line)
        
    def initiate_model_training(self,data_transform_artifact:DataTransformationArtifact)->ModelTrainingArtifact:
        try:
            model_trainer_config:ModelTrainerConfig = self.config.get_model_training_config()
            modeltrainer = ModelTrainer(model_trainer_config, data_transform_artifact)
            trainer_artifact = modeltrainer.start_model_training()
            return trainer_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)

    def initiate_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
     model_trainer_artifact:ModelTrainingArtifact)->ModelEvaluationArtifact:
        try:
            model_eval_config = self.config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(model_evaluation_config=model_eval_config, model_trainer_artifact=model_trainer_artifact,data_valid_artifact=data_validation_artifact)
            eval_artifact= model_evaluation.start_model_evaluation()
            return eval_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)

    def start_model_pusher(self,model_trainer_artifact:ModelTrainingArtifact)->ModelPusherArtifact:
        try:
            model_pusher_config = self.config.get_model_pusher_config()
            model_pusher = ModelPusher(model_trainer_artifact=model_trainer_artifact,
                                       model_pusher_config=model_pusher_config
                                       )
            return model_pusher.initiate_model_pusher()
        except Exception as exp:
            raise FinanceException(exp, sys)



    def start(self):
        try:
            
            data_ingest_artifact = self.initiate_data_ingestion()
            data_validation_artifact=self.initiate_data_validation(data_ingest_artifact)
            data_transformation_artifact = self.initiate_data_transformation(data_valid_artifact=data_validation_artifact)
            model_trainer_artifact = self.initiate_model_training(data_transformation_artifact)
            #model_trainer_artifact = ModelTrainingArtifact(model_trainer_ref_artifact=PartialModelTrainerRefArtifact(trained_model_file_path='/Users/chiragtagadiya/datascience_projects/my_projects/finanace-complaint/finance-artifact/model_trainer/20230103_211304/trained_model/finance_estimator', label_indexer_model_file_path='/Users/chiragtagadiya/datascience_projects/my_projects/finanace-complaint/finance-artifact/model_trainer/20230103_211304/label_indexer'), model_trainer_train_metric_artifact=PartialModelTrainerMetricArtifact(f1_score=1.0, precision_score=1.0, recall_score=1.0), model_trainer_test_metric_artifact=PartialModelTrainerMetricArtifact(f1_score=1.0, precision_score=1.0, recall_score=1.0))
            
            #data_validation_artifact = DataValidationArtifact(accepted_file_path='/Users/chiragtagadiya/datascience_projects/my_projects/finanace-complaint/finance-artifact/data_validation/20230103_211304/accepted_data/finanace_complaint', rejected_dir='/Users/chiragtagadiya/datascience_projects/my_projects/finanace-complaint/finance-artifact/data_validation/20230103_211304/rejected_data')
            model_evalution_artifact = self.initiate_model_evaluation(data_validation_artifact, model_trainer_artifact)
            if model_evalution_artifact.model_accepted:
                model_pusher_artifact=self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
            self.copy_logger_to_dump()
            
        except Exception as exp:
            raise FinanceException(exp, sys)
