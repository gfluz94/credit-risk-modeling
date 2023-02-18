import os
from datetime import datetime
from enum import Enum, auto
from argparse import ArgumentParser
import yaml
import numpy as np
import pandas as pd
import logging
import dill

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from credit_risk_modeling.cleaning import DatetimeConverter, NumericExtractor
from credit_risk_modeling.feature_engineering import (
    TimeSinceCalculator,
    OHECategoriesCreator,
    ReferenceCategoriesDropper,
)
from credit_risk_modeling.evaluation import (
    get_regression_metrics,
    plot_regression_curves,
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger()

import warnings

warnings.filterwarnings("ignore")


class TransformerType(Enum):
    OHECategoriesCreator = auto()
    NumericCategoriesCreator = auto()
    NumericWinsorizer = auto()


if __name__ == "__main__":
    parser = ArgumentParser(description="Input parameters for training a PD model.")
    parser.add_argument(
        "--data-filepath",
        metavar="N",
        type=str,
        help="Local folder where data is located",
        default="../data/loan_data.csv",
    )
    parser.add_argument(
        "--preprocessing-config-file",
        metavar="N",
        type=str,
        help="YAML file containing configurations",
        default="preprocessing.yml",
    )
    parser.add_argument(
        "--target-variable",
        metavar="N",
        type=str,
        help="Name of target variable.",
        default="CCF",
    )
    parser.add_argument(
        "--loan-status-column",
        metavar="N",
        type=str,
        help="Column that contains categories related to default.",
        default="loan_status",
    )
    parser.add_argument(
        "--loss-categories",
        metavar="N",
        type=str,
        nargs="+",
        help="Categories to be considered as losses.",
        default=[
            "Charged Off",
            "Does not meet the credit policy. Status:Charged Off",
        ],
    )
    parser.add_argument(
        "--recovered-column",
        metavar="N",
        type=str,
        help="Column that contains recovered value.",
        default="total_rec_prncp",
    )
    parser.add_argument(
        "--total-loan-amount-column",
        metavar="N",
        type=str,
        help="Column that contains total funded amount for borrower.",
        default="funded_amnt",
    )
    parser.add_argument(
        "--datetime-cols",
        metavar="N",
        type=str,
        nargs="+",
        help="Datetime columns that need to be converted.",
        default=["earliest_cr_line", "issue_d"],
    )
    parser.add_argument(
        "--datetime-format",
        metavar="N",
        type=str,
        help="Origin format for datetime columns.",
        default="%b-%y",
    )
    parser.add_argument(
        "--time-unit",
        metavar="N",
        type=str,
        help="Time unit in which features are engineered on top of datetime columns.",
        default="month",
    )
    parser.add_argument(
        "--reference-date",
        metavar="N",
        type=str,
        help="Reference date for computing time difference for datetime columns.",
        default="Dec-17",
    )
    parser.add_argument(
        "--numeric-cols",
        metavar="N",
        type=str,
        nargs="+",
        help="Numeric columns that will be used for model development.",
        default=[
            "term",
            "emp_length",
            "months_since_issue_d",
            "int_rate",
            "funded_amnt",
            "months_since_earliest_cr_line",
            "annual_inc",
            "delinq_2yrs",
            "inq_last_6mths",
            "open_acc",
            "pub_rec",
            "total_acc",
            "dti",
            "total_rev_hi_lim",
        ],
    )
    parser.add_argument(
        "--regressor-estimators",
        metavar="N",
        type=int,
        help="XGBRegressor number of estimators.",
        default=200,
    )
    parser.add_argument(
        "--regressor-max-depth",
        metavar="N",
        type=int,
        help="XGBRegressor maximum depth.",
        default=4,
    )
    parser.add_argument(
        "--binary-artifacts-path",
        metavar="N",
        type=str,
        help="Path where binary model files should be dumped to.",
        default="../artifacts",
    )
    parser.add_argument(
        "--evaluation-artifacts-path",
        metavar="N",
        type=str,
        help="Path where evluation artifacts should be dumped to.",
        default="output",
    )
    parser.add_argument(
        "--test-size",
        metavar="N",
        type=float,
        help="Fraction of dataset that will be used for test purposes.",
        default=0.20,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether or not to display information on terminal.",
        default=False,
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        help="Random seed to ensure reproducibility.",
        default=99,
    )
    args = parser.parse_args()
    if args.verbose:
        logger.info("Parameters defined!")

    df = pd.read_csv(args.data_filepath)
    if args.verbose:
        logger.info("Dataframe loaded!")
        logger.info("Imputing potential missing information...")
    df["earliest_cr_line"] = df["earliest_cr_line"].fillna(df["issue_d"])

    if args.verbose:
        logger.info("Cleaning data and preprocessing dataframe...")
        logger.info("Transforming datetime into time delta...")
    datetime_converter = DatetimeConverter(
        field_names=args.datetime_cols, datetime_format=args.datetime_format
    )
    df = datetime_converter.transform(df)

    for datetime_col in args.datetime_cols:
        time_since_calculator = TimeSinceCalculator(
            field_name=datetime_col,
            reference_date=datetime.strptime(args.reference_date, args.datetime_format),
            time_unit=args.time_unit,
        )
        df = time_since_calculator.transform(df)

    if args.verbose:
        logger.info("Extracting numeric data from text...")
    emp_length_extractor = NumericExtractor(
        field_name="emp_length",
        regex_extraction=r"(.+)\syears?",
        post_mapping={r"10\+\s?": str(10), r"< 1\s?": str(0)},
    )
    df = emp_length_extractor.transform(df)
    emp_length_extractor = NumericExtractor(
        field_name="term",
        regex_extraction=r"(\d+)",
    )
    df = emp_length_extractor.transform(df)

    if args.verbose:
        logger.info("Creating target variable...")
    df = df[df[args.loan_status_column].isin(args.loss_categories)]
    df[args.target_variable] = np.clip(
        (df[args.total_loan_amount_column] - df[args.recovered_column])
        / df[args.total_loan_amount_column],
        0.0,
        1.0,
    )

    if args.verbose:
        logger.info("Reading from preprocessing configuration YAML file...")
    with open(args.preprocessing_config_file, "r") as stream:
        preprocessing_configuration = yaml.safe_load(stream)

    if args.verbose:
        logger.info("Creating preprocessing pipeline...")
    transformers = []
    base_fields = []
    reference_categories = []
    for field, field_metadata in preprocessing_configuration.items():
        transformers.append(
            OHECategoriesCreator(
                field_name=field,
                final_categories_dict=field_metadata[
                    TransformerType.OHECategoriesCreator.name
                ]["final_categories_dict"],
            )
        )
        reference_categories.append(field_metadata["ReferenceCategory"])
        base_fields.append(field)
    preprocessing_pipeline = Pipeline(
        steps=[
            *[
                (f"transformer_{i+1}", transformer)
                for i, transformer in enumerate(transformers)
            ],
            (
                "drop_reference_cats",
                ReferenceCategoriesDropper(reference_categories=reference_categories),
            ),
        ]
    )
    base_fields += args.numeric_cols
    df[args.numeric_cols] = df[args.numeric_cols].astype(float)

    if args.verbose:
        logger.info("Splitting data into train/test...")
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
    )
    X_train, y_train = df_train[base_fields], df_train[args.target_variable]
    X_test, y_test = df_test[base_fields], df_test[args.target_variable]

    if args.verbose:
        logger.info("Fitting model...")
    X_train = preprocessing_pipeline.fit_transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.regressor_estimators,
        max_depth=args.regressor_max_depth,
        seed=args.seed,
    )
    model.fit(X_train, y_train)

    if args.verbose:
        logger.info("Starting model evaluation...")
        logger.info("Computing regression metrics...")

    y_pred_train = np.clip(model.predict(X_train), 0.0, 1.0)
    y_pred_test = np.clip(model.predict(X_test), 0.0, 1.0)

    train_metrics = get_regression_metrics(y_pred=y_pred_train, y_true=y_train)
    test_metrics = get_regression_metrics(y_pred=y_pred_test, y_true=y_test)

    if args.verbose:
        logger.info("TRAIN METRICS")
        for metric, value in train_metrics.items():
            logger.info("%s = %s", metric, value)
        logger.info("TEST METRICS")
        for metric, value in test_metrics.items():
            logger.info("%s = %s", metric, value)

    metrics_df = pd.DataFrame([train_metrics], index=["TRAIN"])
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [test_metrics],
                index=["TEST"],
            ),
        ],
        axis=0,
    )
    metrics_df.to_csv(
        os.path.join(args.evaluation_artifacts_path, "regression_metrics.csv")
    )

    if args.verbose:
        logger.info("Plotting curves...")
    plot_regression_curves(
        y_pred=y_pred_train,
        y_true=y_train,
        name="TRAIN",
        save_eval_artifacts=True,
        eval_artifacts_path=args.evaluation_artifacts_path,
    )
    plot_regression_curves(
        y_pred=y_pred_test,
        y_true=y_test,
        name="TEST",
        save_eval_artifacts=True,
        eval_artifacts_path=args.evaluation_artifacts_path,
    )

    if args.verbose:
        logger.info("Exporting model artifacts...")

    with open(
        os.path.join(args.binary_artifacts_path, "ead_preprocessing.pkl"), "wb"
    ) as file:
        dill.dump(preprocessing_pipeline, file)
    with open(os.path.join(args.binary_artifacts_path, "ead_model.pkl"), "wb") as file:
        dill.dump(model, file)

    if args.verbose:
        logger.info("FINISHED!")
