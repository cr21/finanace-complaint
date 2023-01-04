from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainingArtifact, DataValidationArtifact, ModelEvaluationArtifact
from src.logger import logger
from src.exception import FinanceException
import sys
from pyspark.sql import DataFrame
from pyspark.sql.types import StructField, StructType, StringType, FloatType
from src.entity.schema import FinanceDataSchema
from src.ml.estimator import S3FinanceEstimator
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.pipeline import PipelineModel
from src.config.spark_manager import spark_session
from src.utils import get_score
from src.data_access.model_eval_artifact import ModelEvaluationArtifactData

class ModelEvaluation:
    def __init__(self, model_evaluation_config:ModelEvaluationConfig, model_trainer_artifact:ModelTrainingArtifact,
    data_valid_artifact:DataValidationArtifact, schema=FinanceDataSchema) -> None:
        try:
            self.model_eval_artifact_data = ModelEvaluationArtifactData()
            self.eval_config =   model_evaluation_config
            self.trainer_artifact=model_trainer_artifact
            self.valid_artifact=data_valid_artifact 
            self.schema=FinanceDataSchema()
            self.bucket_name=self.eval_config.bucket_name
            self.s3_model_dir_key=self.eval_config.model_dir
            self.s3_finance_estimator = S3FinanceEstimator(
                bucket_name=self.bucket_name,
                s3_key=self.s3_model_dir_key
            )

            self.metric_report_schema = StructType([StructField("model_accepted", StringType()),
                                                    StructField("changed_accuracy", FloatType()),
                                                    StructField("trained_model_path", StringType()),
                                                    StructField("best_model_path", StringType()),
                                                    StructField("active", StringType())]
                                                   )

        except Exception as exp:
            raise FinanceException(exp, sys)

    def read_data(self)->DataFrame:
        try:
            df:DataFrame = spark_session.read.parquet(self.valid_artifact.accepted_file_path)
            return df
        except Exception as exp:
            raise FinanceException(exp, sys)


    def evaluate_trained_model(self)->ModelEvaluationArtifact:
        try:
            is_model_accepted = False
            is_active=False
            #load train model path
            trained_model_file_path = self.trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
            trained_model = PipelineModel.load(trained_model_file_path)
            # label indexer for target column
            label_indexer_model_path = self.trainer_artifact.model_trainer_ref_artifact.label_indexer_model_file_path            
            label_indexer_model = StringIndexerModel.load(label_indexer_model_path)
            
            # read dataframe and evaluate trained model
            dataframe:DataFrame = self.read_data()
            dataframe = label_indexer_model.transform(dataframe)

            # get best model from s3 cloud
            best_model_path = self.s3_finance_estimator.get_latest_model_path()
            # pass dataframe  to trained model
            trained_model_dataframe = trained_model.transform(dataframe)
            # get best model from s3 cloud
            best_model_dataframe = self.s3_finance_estimator.transform(dataframe)

            trained_model_f1_score = get_score(dataframe=trained_model_dataframe, metric_name="f1",
                                            label_col=self.schema.target_indexed_label,
                                            prediction_col=self.schema.prediction_column_name)
            best_model_f1_score = get_score(dataframe=best_model_dataframe, metric_name="f1",
                                            label_col=self.schema.target_indexed_label,
                                            prediction_col=self.schema.prediction_column_name)

            logger.info(f"Trained_model_f1_score: {trained_model_f1_score}, Best model f1 score: {best_model_f1_score}")
            changed_accuracy = trained_model_f1_score - best_model_f1_score
            if changed_accuracy >= self.eval_config.threshold:
                is_model_accepted, is_active = True, True

            model_evaluation_artifact = ModelEvaluationArtifact(model_accepted=is_model_accepted,
                                                                changed_accuracy=changed_accuracy,
                                                                trained_model_path=trained_model_file_path,
                                                                best_model_path=best_model_path,
                                                                active=is_active
                                                                )
            logger.info(f"Model Evaluation Artifact :{model_evaluation_artifact}")
            return model_evaluation_artifact

            
        except Exception as exp:
            raise FinanceException(exp,sys)
        
    def start_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            model_accepted = True
            is_active=True
            # if no model is saved on cloud save current model as base model
            if not self.s3_finance_estimator.is_model_available(self.s3_finance_estimator.s3_key):
                latest_model_path = None
                trained_model_path = self.trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
                model_evaluation_artifact = ModelEvaluationArtifact(
                                                                    model_accepted=model_accepted,
                                                                    changed_accuracy=0.0,
                                                                    trained_model_path=trained_model_path,
                                                                    best_model_path=latest_model_path,
                                                                    active=is_active
                                                                )
            else:
                model_evaluation_artifact = self.evaluate_trained_model()
            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            # store in mongodb
            self.model_eval_artifact_data.save_eval_artifact(model_eval_artifact=model_evaluation_artifact)
            return model_evaluation_artifact
        except Exception as exp:
            raise FinanceException(exp, sys)


