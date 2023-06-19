import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.dataset.schema import BASE_SCHEMA, PREPROCESSED_SCHEMA
from src.middleware.logger import configure_logger

logger = configure_logger(__name__)


# 特徴量（60歳以上かどうか）を生成する関数
def is_over_sixty(df: pd.DataFrame) -> pd.DataFrame:
    df["isOver60yr"] = df["age"].apply(lambda x: 1 if x >= 60 else 0)
    return df


# 特徴量（datetime）を生成する関数
def create_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # month を文字列から数値に変換
    month_dict = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df["month_int"] = df["month"].map(month_dict)

    # month と day を datetime に変換
    data_datetime = (
        df.assign(
            ymd_str=lambda x: "2014"
            + "-"
            + x["month_int"].astype(str)
            + "-"
            + x["day"].astype(str)
        )
        .assign(datetime=lambda x: pd.to_datetime(x["ymd_str"]))["datetime"]
        .values
    )

    # datetime を int に変換する
    index = pd.DatetimeIndex(data_datetime)
    df["datetime_int"] = np.log(index.astype(np.int64))

    # 不要な列を削除
    df = df.drop(["month", "day", "month_int"], axis=1)
    return df


# 1%、99%点を計算し、clipping
def clip_value(df: pd.DataFrame) -> pd.DataFrame:
    p01 = df["balance"].quantile(0.01)
    p99 = df["balance"].quantile(0.99)
    df["balance"] = df["balance"].clip(p01, p99)
    return df


class BasePreprocessPipeline(ABC, BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        x: pd.DataFrame,
        y=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def dump_pipeline(
        self,
        file_path: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def load_pipeline(
        self,
        file_path: str,
    ):
        raise NotImplementedError


class DataPreprocessPipeline(BasePreprocessPipeline):
    def __init__(self) -> None:
        self.categorical_features = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "poutcome",
        ]

        self.pipeline: Union[Pipeline, ColumnTransformer] = None
        self.define_pipeline()

    def define_pipeline(self):
        categorical_pipeline = Pipeline(
            [
                (
                    "simple_imputer",
                    SimpleImputer(
                        missing_values=np.nan, strategy="constant", fill_value=None
                    ),
                ),
                ("ordinal_encoder", OrdinalEncoder()),
            ]
        )
        self.pipeline = ColumnTransformer(
            [
                ("categorical", categorical_pipeline, self.categorical_features),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        logger.info(f"pipeline: {self.pipeline}")

    def preprocess(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        x = BASE_SCHEMA.validate(x)
        x = is_over_sixty(x)
        x = create_datetime(x)
        x = clip_value(x)
        return x

    def fit(
        self,
        x: pd.DataFrame,
        y=None,
    ):
        if self.pipeline is None:
            raise AttributeError
        x = PREPROCESSED_SCHEMA.validate(x)
        self.pipeline.fit(x)

        return self

    def transform(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.pipeline is None:
            raise AttributeError
        x = PREPROCESSED_SCHEMA.validate(x)
        pipe_df = self.pipeline.transform(x)
        df = self.postprocess(x, pipe_df)
        return df

    def fit_transform(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        if self.pipeline is None:
            raise AttributeError
        x = PREPROCESSED_SCHEMA.validate(x)
        pipe_df = self.pipeline.fit_transform(x)
        df = self.postprocess(x, pipe_df)
        return df

    def postprocess(self, df, pipe_df):
        for cat in self.categorical_features:
            df[f"{cat}"] = pipe_df[f"{cat}"]
        return df

    def dump_pipeline(
        self,
        file_path: str,
    ) -> str:
        file, ext = os.path.splitext(file_path)
        if ext != ".pkl":
            file_path = f"{file}.pkl"
        logger.info(f"save preprocess pipeline: {file_path}")
        dump(self.pipeline, file_path)
        return file_path

    def load_pipeline(
        self,
        file_path: str,
    ):
        logger.info(f"load preprocess pipeline: {file_path}")
        self.pipeline = load(file_path)
