
from src.entity.artifact_entity import  DataTransformationArtifact,ModelTrainingArtifact,\
    PartialModelTrainerMetricArtifact,PartialModelTrainerRefArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.logger import logger
from src.exception import FinanceException
from src.entity.schema import FinanceDataSchema
import os, sys
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer,StringIndexerModel
from typing import List
from pyspark.ml.feature import IndexToString
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from src.utils import get_score
from src.config.spark_manager import spark_session 
class ModelTrainer:

    def __init__(self, config:ModelTrainerConfig, artifact:DataTransformationArtifact,
                         schema=FinanceDataSchema()) -> None:
        self.trainer_config = config
        self.transform_artifact= artifact
        self.schema = schema

    def get_train_test_dataframe(self)->List[DataFrame]:
        try:
            train_data_path = self.transform_artifact.transformed_train_file_path
            test_data_path = self.transform_artifact.transformed_test_file_path
            train_df  = spark_session.read.parquet(train_data_path)
            test_df = spark_session.read.parquet(test_data_path)
            print(f"Train row: {train_df.count()} Test row: {test_df.count()}")
            dataframes: List[DataFrame] = [train_df, test_df]
            return dataframes
        except Exception as exp:
            raise FinanceException(exp, sys)

    def get_model(self, label_indexer_model:StringIndexerModel)->Pipeline:
        try:
            stages = []
            logger.info("Creating Random Forest Classifier class.")
            rand_forest_clf  =RandomForestClassifier(labelCol=self.schema.target_indexed_label,
                                                featuresCol=self.schema.scaled_vector_input_features)

            logger.info("Creating Label generator")
            label_generator  = IndexToString(inputCol=self.schema.prediction_column_name, 
                                            outputCol=f"{self.schema.prediction_column_name}_{self.schema.target_column}",
                                            labels=label_indexer_model.labels)

            stages.append(rand_forest_clf)
            stages.append(label_generator)
            pipeline = Pipeline(stages=stages)
            return pipeline
        except Exception as exp:
            raise FinanceException(exp, sys)
    
    def get_scores(self, dataframe:DataFrame, metric_names:List[str])->List[tuple]:
        try:
            if metric_names is None:
                metric_names = self.trainer_config.metric_list
            scores:List[tuple]=[]
            for metric in metric_names:
                score = get_score(metric_name=metric,
                                dataframe=dataframe,
                                label_col = self.schema.target_indexed_label,
                                prediction_col=self.schema.prediction_column_name)

                scores.append((metric,score))

            return scores
        
        except Exception as exp:
            raise FinanceException(exp, sys)

    def export_trained_model(self, model:PipelineModel):
        try:
            transformed_pipeline_file_path = self.transform_artifact.exported_pipeline_file_path
            transformed_pipeline = PipelineModel.load(transformed_pipeline_file_path)

            # entire pipeline (DAta Transformation -> Model -> Label indexer)
            updated_stages = transformed_pipeline.stages + model.stages
            transformed_pipeline.stages = updated_stages
            trained_model_file_path = self.trainer_config.trained_model_file_path
            transformed_pipeline.save(trained_model_file_path)

            logger.info("Creating trained model directory")
            trained_model_file_path = self.trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)
            # transformed_pipeline.save(trained_model_file_path)
            ref_artifact = PartialModelTrainerRefArtifact(
                trained_model_file_path=trained_model_file_path,
                label_indexer_model_file_path=self.trainer_config.label_indexer_model_dir)

            logger.info(f"Model trainer reference artifact: {ref_artifact}")
            return ref_artifact

        except Exception as exp:
            raise FinanceException(exp, sys)
    def start_model_training(self)-> ModelTrainingArtifact:
        try:
            # 1 get Train test dataframe
            dataframes = self.get_train_test_dataframe()
            train_df, test_df = dataframes[0], dataframes[1]
            print(f"Train row: {train_df.count()} Test row: {test_df.count()}")
            label_indexer = StringIndexer(inputCol=self.schema.target_column, outputCol=self.schema.target_indexed_label)
            label_indexer_model = label_indexer.fit(train_df)
            # save label indexer
            os.makedirs(os.path.dirname(self.trainer_config.label_indexer_model_dir), exist_ok=True)
            label_indexer_model.save(self.trainer_config.label_indexer_model_dir)
            # apply label transformation
            train_df=label_indexer_model.transform(train_df)
            test_df=label_indexer_model.transform(test_df)

            # 2. Get Trainer Model
            model  =self.get_model(label_indexer_model)

            trained_model = model.fit(train_df)
            train_df_prediction = trained_model.transform(train_df)
            test_df_prediction  = trained_model.transform(test_df)
            print(f"number of row in training: {train_df.count()}")
            scores = self.get_scores(dataframe=train_df_prediction, metric_names=self.trainer_config.metric_list)
            print("scores", scores)
            

            # 3. trained Model
            train_metric_artifact = PartialModelTrainerMetricArtifact(f1_score=scores[0][1],
                                                                      precision_score=scores[1][1],
                                                                      recall_score=scores[2][1])


            logger.info(f"Model trainer train metric: {train_metric_artifact}")

            print(f"number of row in training: {train_df_prediction.count()}")

            scores = self.get_scores(dataframe=test_df_prediction,metric_names=self.trainer_config.metric_list)
            
            test_metric_artifact = PartialModelTrainerMetricArtifact(f1_score=scores[0][1],
                                                                     precision_score=scores[1][1],
                                                                     recall_score=scores[2][1])

            logger.info(f"Model trainer test metric: {test_metric_artifact}")
            ref_artifact = self.export_trained_model(model=trained_model)
            model_trainer_artifact = ModelTrainingArtifact(model_trainer_ref_artifact=ref_artifact,
                                                          model_trainer_train_metric_artifact=train_metric_artifact,
                                                          model_trainer_test_metric_artifact=test_metric_artifact)

            logger.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        except Exception as exp:
            raise FinanceException(exp,sys)