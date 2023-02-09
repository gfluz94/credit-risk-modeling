import pandas as pd

from credit_risk_modeling.evaluation import get_metrics_across_thresholds


def test_get_metrics_across_thresholds(scores_and_true_labels: pd.DataFrame):
    # OUTPUT
    output = get_metrics_across_thresholds(
        y_proba=scores_and_true_labels.pd.values,
        y_true=scores_and_true_labels.true_labels,
    )

    # EXPECTED
    metrics_df = pd.DataFrame(
        {
            0.05: {
                "# DEFAULT": 97.0,
                "RECALL": 0.9636363636363636,
                "PRECISION": 0.5463917525773195,
                "F1": 0.6973684210526315,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.1: {
                "# DEFAULT": 94.0,
                "RECALL": 0.9636363636363636,
                "PRECISION": 0.5638297872340425,
                "F1": 0.7114093959731544,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.15: {
                "# DEFAULT": 90.0,
                "RECALL": 0.8909090909090909,
                "PRECISION": 0.5444444444444444,
                "F1": 0.6758620689655171,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.2: {
                "# DEFAULT": 84.0,
                "RECALL": 0.8727272727272727,
                "PRECISION": 0.5714285714285714,
                "F1": 0.6906474820143884,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.25: {
                "# DEFAULT": 74.0,
                "RECALL": 0.7636363636363637,
                "PRECISION": 0.5675675675675675,
                "F1": 0.6511627906976745,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.3: {
                "# DEFAULT": 70.0,
                "RECALL": 0.7272727272727273,
                "PRECISION": 0.5714285714285714,
                "F1": 0.64,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.35: {
                "# DEFAULT": 67.0,
                "RECALL": 0.6909090909090909,
                "PRECISION": 0.5671641791044776,
                "F1": 0.6229508196721312,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.39999999999999997: {
                "# DEFAULT": 63.0,
                "RECALL": 0.6363636363636364,
                "PRECISION": 0.5555555555555556,
                "F1": 0.5932203389830508,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.44999999999999996: {
                "# DEFAULT": 59.0,
                "RECALL": 0.5818181818181818,
                "PRECISION": 0.5423728813559322,
                "F1": 0.5614035087719298,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.49999999999999994: {
                "# DEFAULT": 56.0,
                "RECALL": 0.5636363636363636,
                "PRECISION": 0.5535714285714286,
                "F1": 0.5585585585585585,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.5499999999999999: {
                "# DEFAULT": 49.0,
                "RECALL": 0.4909090909090909,
                "PRECISION": 0.5510204081632653,
                "F1": 0.5192307692307692,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.6: {
                "# DEFAULT": 48.0,
                "RECALL": 0.4909090909090909,
                "PRECISION": 0.5625,
                "F1": 0.5242718446601942,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.65: {
                "# DEFAULT": 39.0,
                "RECALL": 0.36363636363636365,
                "PRECISION": 0.5128205128205128,
                "F1": 0.425531914893617,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.7: {
                "# DEFAULT": 33.0,
                "RECALL": 0.3090909090909091,
                "PRECISION": 0.5151515151515151,
                "F1": 0.38636363636363635,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.75: {
                "# DEFAULT": 25.0,
                "RECALL": 0.2,
                "PRECISION": 0.44,
                "F1": 0.275,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.7999999999999999: {
                "# DEFAULT": 19.0,
                "RECALL": 0.14545454545454545,
                "PRECISION": 0.42105263157894735,
                "F1": 0.2162162162162162,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.85: {
                "# DEFAULT": 12.0,
                "RECALL": 0.09090909090909091,
                "PRECISION": 0.4166666666666667,
                "F1": 0.1492537313432836,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.9: {
                "# DEFAULT": 10.0,
                "RECALL": 0.07272727272727272,
                "PRECISION": 0.4,
                "F1": 0.12307692307692307,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
            0.95: {
                "# DEFAULT": 4.0,
                "RECALL": 0.01818181818181818,
                "PRECISION": 0.25,
                "F1": 0.03389830508474576,
                "ROC-AUC": 0.48444444444444446,
                "AVERAGE PRECISION": 0.5197081264053752,
            },
        }
    )

    # ASSERT
    pd.testing.assert_frame_equal(
        metrics_df.T,
        output,
    )
