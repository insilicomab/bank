from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

from src.middleware.logger import configure_logger
from src.models.base_model import AbstractBaseModel
from src.models.preprocess import DataPreprocessPipeline

logger = configure_logger(name=__name__)


@dataclass
class Evaluation:
    eval_df: pd.DataFrame
    area_under_the_curve: float


class Artifact(BaseModel):
    preprocess_file_path: Optional[str]
    model_file_path: Optional[str]


class Trainer(object):
    def __init__(self):
        pass

    def train(
        self,
        model: AbstractBaseModel,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def evaluate(
        self,
        model: AbstractBaseModel,
        x: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Evaluation:
        predictions = model.predict(x=x)

        auc = roc_auc_score(y_true=y, y_score=predictions)

        evaluation = Evaluation(
            eval_df=x,
            area_under_the_curve=auc,
        )
        logger.info(
            f"""
model: {model.name}
auc: {evaluation.area_under_the_curve}
            """
        )
        return evaluation

    def train_and_evaluate(
        self,
        model: AbstractBaseModel,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        data_preprocess_pipeline: Optional[DataPreprocessPipeline] = None,
        preprocess_pipeline_file_path: Optional[str] = None,
        save_file_path: Optional[str] = None,
    ) -> Tuple[Evaluation, Artifact]:
        logger.info("start training and evaluation")
        self.train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        evaluation = self.evaluate(
            model=model,
            x=x_test,
            y=y_test,
        )

        artifact = Artifact()
        if (
            data_preprocess_pipeline is not None
            and preprocess_pipeline_file_path is not None
        ):
            artifact.preprocess_file_path = data_preprocess_pipeline.dump_pipeline(
                file_path=preprocess_pipeline_file_path
            )

        if save_file_path is not None:
            artifact.model_file_path = model.save(file_path=save_file_path)

        logger.info("done training and evaluation")
        return evaluation, artifact
