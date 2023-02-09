import numpy as np

from credit_risk_modeling.utils import convert_probabilities_to_scores


def test_convert_probabilities_to_scores(probabilities_of_default: np.array):
    # OUTPUT
    output = convert_probabilities_to_scores(y_proba=probabilities_of_default)

    # EXPECTED
    scores = np.array(
        [
            752.8742382611335,
            572.0814264733772,
            550.5213762333517,
            537.1728830150892,
            527.1228762045055,
            518.8221262189286,
            511.57072463123444,
            504.98457212617524,
            498.8221262189286,
            492.9130085484052,
            487.1228762045055,
            481.3327438606058,
            475.42362619008236,
            469.2611802828357,
            462.6750277777765,
            455.42362619008236,
            447.1228762045055,
            437.0728693939218,
            423.72437617565924,
            402.16432593563377,
            221.37151414787417,
        ]
    )

    # ASSERT
    np.testing.assert_array_equal(
        scores,
        output,
    )
