from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from pyspark.ml.feature import StandardScaler, VectorAssembler, OneHotEncoder, StringIndexer, Imputer
from src.entity.schema import FinanceDataSchema
from src.exception import FinanceException
from src.logger import logger
from src.config.spark_manager import spark_session
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, rand
from pyspark.ml.feature import IDF, Tokenizer, HashingTF
from src.ml.feature import DerivedFeatureGenerator, FrequencyImputer, FrequencyImputerModel
from pyspark.ml import Pipeline
import sys
import os

class DataTransformer():
    def __init__(self, data_transform_config:DataTransformationConfig,
                        data_validator:DataValidationArtifact,
                        schema=FinanceDataSchema()) -> None:
        try:
            super().__init__()
            self.transform_config=data_transform_config
            self.validator_artifact=data_validator
            self.schema=schema
        except Exception as exp:
            raise FinanceException(exp, sys)
        
        
    def read_data(self)->DataFrame:
        try:
            return spark_session.read.parquet(self.validator_artifact.accepted_file_path)
        except Exception as exp:
            raise FinanceException(exp, sys)

    def get_Transformation_Pipeline(self)-> Pipeline:
        try:
            # different pipeline stages
            # stage 1: Generate Derived numerical column
            # stage 2: Generate Imputer for numerical column
            # stage 3: Generate Frequency Imputer for Categorical column
            # stage 4:


            stages=[]
            
            # numerical column transformation
            # addaddditional derived columns
            derived_features  = DerivedFeatureGenerator(inputCols=self.schema.derived_input_features,outputCols=self.schema.derived_output_features)
            stages.append(derived_features)
            # Missing value imputation

            # creating imputer to fill null values
            imputer = Imputer(inputCols=self.schema.numerical_columns,
                              outputCols=self.schema.im_numerical_columns)
            stages.append(imputer)
             
            # frequency imputer for categorical values

            frequency_imputer = FrequencyImputer(inputCols=self.schema.one_hot_encoding_features, 
            outputCols=self.schema.im_one_hot_encoding_features)
            stages.append(frequency_imputer)

            # label imputed categorical features with numerical label using string indexer
            for im_one_hot_feature, string_indexer_col in zip(self.schema.im_one_hot_encoding_features,
                                            self.schema.string_indexer_one_hot_features):

                string_indexer = StringIndexer(inputCol=im_one_hot_feature,outputCol=string_indexer_col)
                stages.append(string_indexer)

            # convert string indexer labels to one hot encoding labels

            one_hot_encoder = OneHotEncoder(inputCols=self.schema.string_indexer_one_hot_features,
            outputCols=self.schema.tf_one_hot_encoding_features)
            stages.append(one_hot_encoder)

            # handling text features with TFIDF features
            tokenizer = Tokenizer(inputCol=self.schema.tfidf_features[0], outputCol="words")
            stages.append(tokenizer)
            # count vectorizer with hashing bucketing # same index could be used by multiple words/tokens
            hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=40)
            stages.append(hashing_tf)
            idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol=self.schema.tf_tfidf_features[0])
            stages.append(idf)

            # vectorized everything and combined all features in single feature vector
            vector_assembler = VectorAssembler(inputCols=self.schema.input_features,
                                               outputCol=self.schema.vector_assembler_output)

            stages.append(vector_assembler)

            # standaradized every vector
            standard_scaler = StandardScaler(inputCol=self.schema.vector_assembler_output,
                                             outputCol=self.schema.scaled_vector_input_features)
            stages.append(standard_scaler)
            pipeline = Pipeline(
                stages=stages
            )
            logger.info(f"Data transformation pipeline: [{pipeline}]")
            logger.info(f"Stages : {stages}")
            logger.info(f"Stages : {pipeline.stages}")
            return pipeline

        except Exception as exp:
            raise FinanceException(exp, sys)

    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logger.info("Data Transformation Started")
            # 1. read data frame
            dataframe:DataFrame= spark_session.read.parquet(self.validator_artifact.accepted_file_path)
            logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")
            # 2. train test split
            test_size=self.transform_config.test_size
            logger.info(f"Splitting dataset into train and test set using ration: {1 - test_size}:{test_size}")
            train_df, test_df = dataframe.randomSplit([1-test_size, test_size])
            logger.info(f"Train dataset has number of row: [{train_df.count()}] and"
                        f" column: [{len(train_df.columns)}]")

            logger.info(f"Train dataset has number of row: [{test_df.count()}] and"
                        f" column: [{len(test_df.columns)}]")
    
            # 3. Get Transformation Pipeline
            pipeline = self.get_Transformation_Pipeline()

            # 4. fit pipeline on training data
            transformed_pipeline = pipeline.fit(train_df)
            required_columns = [self.schema.scaled_vector_input_features, self.schema.target_column]
            # 5. apply transformation to train and test

            transformed_train_df = transformed_pipeline.transform(train_df)
            transformed_train_df = transformed_train_df.select(required_columns)

            transformed_test_df = transformed_pipeline.transform(test_df)
            transformed_test_df = transformed_test_df.select(required_columns)

            # 5. store train, test transformed file to file path and store pipeline object as well
            # creating required directories
            os.makedirs(self.transform_config.exported_pipeline_dir, exist_ok=True)
            os.makedirs(self.transform_config.transformed_train_dir,exist_ok=True)
            os.makedirs(self.transform_config.transformed_test_dir, exist_ok=True)

            transformed_train_data_file_path = os.path.join(self.transform_config.transformed_train_dir,
                                                            self.transform_config.file_name)

            transformed_test_data_file_path = os.path.join(self.transform_config.transformed_test_dir, 
                                                            self.transform_config.file_name)

            # 6. save transformation pipeline
            logger.info(f"Saving transformation pipeline at: [{self.transform_config.exported_pipeline_dir}]")
            transformed_pipeline.save(self.transform_config.exported_pipeline_dir)

            logger.info(f"Saving transformed train data at: [{transformed_train_data_file_path}]")
            print(transformed_train_df.count(), len(transformed_train_df.columns))
            transformed_train_df.write.parquet(transformed_train_data_file_path)

            logger.info(f"Saving transformed test data at: [{transformed_test_data_file_path}]")
            print(transformed_test_df.count(), len(transformed_test_df.columns))
            transformed_test_df.write.parquet(transformed_test_data_file_path)

            data_tf_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_data_file_path,
                transformed_test_file_path=transformed_test_data_file_path,
                exported_pipeline_file_path=self.transform_config.exported_pipeline_dir,

            )

            logger.info(f"Data transformation artifact: [{data_tf_artifact}]")
            return data_tf_artifact

        except Exception as exp:
            raise FinanceException(exp, sys)