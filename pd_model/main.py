import os
from datetime import datetime
from enum import Enum, auto
from argparse import ArgumentParser
import yaml
import pandas as pd
import logging
import dill

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from credit_risk_modeling.cleaning import (
    DatetimeConverter,
    NumericExtractor,
    NumericWinsorizer,
)
from credit_risk_modeling.feature_engineering import (
    TimeSinceCalculator,
    OHECategoriesCreator,
    NumericCategoriesCreator,
    ReferenceCategoriesDropper,
)
from credit_risk_modeling.eda import compute_woe, plot_woe_by_category, get_fine_classes
from credit_risk_modeling.evaluation import (
    get_metrics_across_thresholds,
    plot_distributions,
    plot_roc_pr_curves,
    plot_ks_curve,
)
from credit_risk_modeling.utils import convert_probabilities_to_scores

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
        default="default",
    )
    parser.add_argument(
        "--default-status-column",
        metavar="N",
        type=str,
        help="Column that contains categories related to default.",
        default="loan_status",
    )
    parser.add_argument(
        "--default-categories",
        metavar="N",
        type=str,
        nargs="+",
        help="Categories to be considered as default.",
        default=[
            "Charged Off",
            "Late (31-120 days)",
            "Default",
            "Does not meet the credit policy. Status:Charged Off",
        ],
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
        "--save-evaluation-artifacts",
        action="store_true",
        help="Whether or not to save evaluation output.",
        default=False,
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
    df[args.target_variable] = (
        df[args.default_status_column].isin(args.default_categories)
    ).astype(float)

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
        if TransformerType.NumericWinsorizer.name in field_metadata.keys():
            transformers.append(
                NumericWinsorizer(
                    field_name=field,
                    lower=field_metadata[TransformerType.NumericWinsorizer.name][
                        "lower"
                    ],
                    upper=field_metadata[TransformerType.NumericWinsorizer.name][
                        "upper"
                    ],
                )
            )
        if TransformerType.NumericCategoriesCreator.name in field_metadata.keys():
            transformers.append(
                NumericCategoriesCreator(
                    field_name=field,
                    boundaries=field_metadata[
                        TransformerType.NumericCategoriesCreator.name
                    ]["boundaries"],
                )
            )
        if TransformerType.OHECategoriesCreator.name in field_metadata.keys():
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

    if args.verbose:
        logger.info("Splitting data into train/test...")
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[args.target_variable],
    )
    X_train, y_train = df_train[base_fields], df_train[args.target_variable]
    X_test, y_test = df_test[base_fields], df_test[args.target_variable]

    if args.verbose:
        logger.info("Fitting model...")
    X_train = preprocessing_pipeline.fit_transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    model = LogisticRegression(class_weight="balanced", random_state=args.seed)
    model.fit(X_train, y_train)

    ### SAVE MODEL
