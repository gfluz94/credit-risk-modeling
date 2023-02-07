import pandas as pd
import numpy as np

from credit_risk_modeling.eda import get_fine_classes


def test_get_fine_classes():
    # OUTPUT
    toy_series = pd.Series(np.arange(30), name="test")
    output, (lower, upper) = get_fine_classes(s=toy_series, n_buckets=10)

    # EXPECTED
    classes = pd.Series(
        sorted(
            [
                "00. (-0.029, 2.9]",
                "01. (2.9, 5.8]",
                "02. (5.8, 8.7]",
                "03. (8.7, 11.6]",
                "04. (11.6, 14.5]",
                "05. (14.5, 17.4]",
                "06. (17.4, 20.3]",
                "07. (20.3, 23.2]",
                "08. (23.2, 26.1]",
                "09. (26.1, 29.0]",
            ]
            * 3
        ),
        name="test_categories",
        index=toy_series.index,
    )

    # ASSERT
    pd.testing.assert_series_equal(
        classes,
        output,
    )
    assert (lower, upper) == (0, 30)
