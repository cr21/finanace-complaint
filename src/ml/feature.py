from src.config.spark_manager import spark_session
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Estimator, Transformer
from src.exception import FinanceException
from src.logger import logger
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import Param, Params, TypeConverters, HasOutputCols, \
    HasInputCols
# avai in pyspark>=2.3.0
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import desc
from pyspark.sql.functions import col, abs
from typing import List, Dict
from pyspark.sql.types import TimestampType, LongType


class FrequencyImputer(Transformer,               # Base class
                     HasInputCols,               # Sets up an inputCol parameter
                     HasOutputCols,              # Sets up an outputCol parameter
                     DefaultParamsReadable,     # Makes parameters readable from file
                     DefaultParamsWritable      # Makes parameters writable from file
                    ):
    
    @keyword_only
    def __init__(self) -> None:
        super(FrequencyImputer, self).__init__()
        self.topCategorys = Param(self, "topCategorys", "")
        self._setDefault(topCategorys="")
        kwargs = self._input_kwargs
        print(kwargs)
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols: List[str] = None, outputCols: List[str] = None, ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setTopCategorys(self, value: List[str]):
        return self._set(topCategorys=value)


    def getTopCategorys(self):
        return self.getOrDefault(self.topCategorys)

    def setInputCols(self, value: List[str]):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCols=value)

    def setOutputCols(self, value: List[str]):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCols=value)


    def _fit(self, dataset: DataFrame):
        inputCols = self.getInputCols()
        topCategorys = []
        for column in inputCols:
            categoryCountByDesc = dataset.groupBy(column).count().filter(f'{column} is not null').sort(
                desc('count'))
            topCat = categoryCountByDesc.take(1)[0][column]
            topCategorys.append(topCat)

        print(topCategorys)

        self.setTopCategorys(value=topCategorys)

        # transformer model
        estimator = FrequencyImputerModel(inputCols=self.getInputCols(),
                                          outputCols=self.getOutputCols())

        estimator.setTopCategorys(value=topCategorys)
        return estimator

    

class FrequencyImputerModel(FrequencyImputer, Transformer):

    def __init__(self, inputCols: List[str] = None, outputCols: List[str] = None, ):
        super(FrequencyImputerModel, self).__init__(inputCols=inputCols, outputCols=outputCols)

    def _transform(self, dataset: DataFrame):
        topCategorys = self.getTopCategorys()
        outputCols = self.getOutputCols()

        updateMissingValue = dict(zip(outputCols, topCategorys))

        inputCols = self.getInputCols()
        for outputColumn, inputColumn in zip(outputCols, inputCols):
            dataset = dataset.withColumn(outputColumn, col(inputColumn))
            # print(dataset.columns)
            # print(outputColumn, inputColumn)

        dataset = dataset.na.fill(updateMissingValue)

        return dataset



class DerivedFeatureGenerator(Transformer,               # Base class
                     HasInputCols,               # Sets up an inputCol parameter
                     HasOutputCols,              # Sets up an outputCol parameter
                     DefaultParamsReadable,     # Makes parameters readable from file
                     DefaultParamsWritable      # Makes parameters writable from file
                    ):

    topCategorys = Param(Params._dummy(), "getTopCategorys", "getTopCategorys",
                         typeConverter=TypeConverters.toListString)


    @keyword_only
    def __init__(self) -> None:
        super(self,DerivedFeatureGenerator).__init__()
        kwargs = self._input_kwargs
        self.second_within_day = 60 * 60 * 24
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols: List[str] = None, outputCols: List[str] = None, ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def setInputCols(self, value: List[str]):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCols(self, value: List[str]):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCols=value)

    def _fit(self, dataframe: DataFrame):
        return self

    def _transform(self, dataframe: DataFrame):
        inputCols = self.getInputCols()

        for column in inputCols:
            dataframe = dataframe.withColumn(column,
                                             col(column).cast(TimestampType()))
        # get diff in days
        dataframe = dataframe.withColumn(self.getOutputCols()[0], abs(
            col(inputCols[1]).cast(LongType()) - col(inputCols[0]).cast(LongType())) / (
                                             self.second_within_day))
        return dataframe
