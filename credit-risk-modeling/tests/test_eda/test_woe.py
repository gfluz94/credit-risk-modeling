import pandas as pd
import numpy as np

from credit_risk_modeling.eda import compute_woe


def test_compute_woe():
    # OUTPUT
    toy_dastaset = pd.DataFrame(
        {
            "id": np.arange(20_000),
            "education": ["higher"] * 4_600 + ["lower"] * 15_400,
            "default": [0.0] * 4_000 + [1.0] * 4_000 + [0.0] * 12_000,
        }
    )
    output = compute_woe(
        df=toy_dastaset,
        id_column_name="id",
        feature_column_name="education",
        target_column_name="default",
    )

    # EXPECTED
    woe_values = pd.Series(
        [-0.5108256237659907, 0.125163142954006], index=["higher", "lower"], name="WoE"
    )
    woe_values.index.names = ["education"]
    information_value = 0.06359887667199968
    print(woe_values)
    print(output.WoE)

    # ASSERT
    pd.testing.assert_series_equal(
        woe_values,
        output.WoE,
    )
    assert information_value == output.IV.iloc[0]
