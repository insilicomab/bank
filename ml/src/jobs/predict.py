import numpy as np
import pandas as pd

from src.dataset.schema import (
    PREDICTION_SCHEMA,
    PREPROCESSED_PREDICTION_SCHEMA,
    RAW_PREDICTION_SCHEMA,
    X_SCHEMA,
)
from src.middleware.logger import configure_logger
from src.models.base_model import AbstractBaseModel
from src.models.preprocess import DataPreprocessPipeline

logger = configure_logger(name=__name__)


class Predictor:
    def __init__(self):
        pass

    def postprocess(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        df = df[["id"]]
        df.loc[:, "prediction"] = predictions
        df = PREDICTION_SCHEMA.validate(df)
        logger.info(
            f"""
predicted df columns: {df.columns}
predicted df shape: {df.shape}
    """
        )
        return df

    def predict(
        self,
        model: AbstractBaseModel,
        data_preprocess_pipeline: DataPreprocessPipeline,
        data_to_be_predicted_df: pd.DataFrame,
    ) -> pd.DataFrame:
        data_to_be_predicted_df = RAW_PREDICTION_SCHEMA.validate(
            data_to_be_predicted_df
        )
        _data_to_be_predicted_df = data_preprocess_pipeline.preprocess(
            x=data_to_be_predicted_df
        )
        x = data_preprocess_pipeline.transform(_data_to_be_predicted_df)
        x = PREPROCESSED_PREDICTION_SCHEMA.validate(x)
        x_test = x.drop(["id"], axis=1)
        x_test = X_SCHEMA.validate(x_test)

        predictions = model.predict(x=x_test)
        predictions = self.postprocess(
            df=data_to_be_predicted_df, predictions=predictions
        )
        return predictions
