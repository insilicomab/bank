import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.dataset.data_manager import load_df_from_csv
from src.dataset.schema import BASE_SCHEMA, X_SCHEMA, Y_SCHEMA
from src.middleware.logger import configure_logger
from src.models.preprocess import DataPreprocessPipeline

logger = configure_logger(__name__)


class DataRetriever:
    def __init__(self):
        pass

    def retrieve_dataset(self, file_path: str) -> pd.DataFrame:
        logger.info("start retrieve data")
        raw_df = load_df_from_csv(file_path)
        raw_df = BASE_SCHEMA.validate(raw_df)
        logger.info(
            f"""
Loaded dataset
raw_df columns: {raw_df.columns}
raw_df shape: {raw_df.shape}
    """
        )
        return raw_df

    def stratified_kfold_split(
        self, raw_df: pd.DataFrame, data_preprocess_pipeline: DataPreprocessPipeline
    ) -> list[list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        _train_df = data_preprocess_pipeline.preprocess(x=raw_df)
        train_df = data_preprocess_pipeline.fit_transform(x=_train_df)

        # print(
        #     data_preprocess_pipeline.pipeline.named_transformers_["categorical"]
        #     .named_steps["ordinal_encoder"]
        #     .categories_
        # )
        logger.info(
            f"""
preprocessed train df columns: {train_df.columns}
preprocessed train df shape: {train_df.shape}
    """
        )

        X_train = train_df.drop(["y", "id"], axis=1).reset_index(drop=True)
        Y_train = train_df["y"].reset_index(drop=True)

        X_train = X_SCHEMA.validate(X_train)
        Y_train = Y_SCHEMA.validate(Y_train)

        cv_dataset = [
            [
                X_train.iloc[train_index],
                X_train.iloc[val_index],
                Y_train.iloc[train_index],
                Y_train.iloc[val_index],
            ]
            for train_index, val_index in StratifiedKFold(
                n_splits=10, shuffle=True, random_state=1234
            ).split(X_train, Y_train)
        ]
        logger.info("done split data")
        return cv_dataset
