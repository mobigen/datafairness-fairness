# Fairness Metrics

## openfx usage
```shell
curl --location --request POST 'http://fair5:31113/function/fairness' \
--header 'Content-Type: application/json' \
--data-raw '{
    "input": {
        "type": "iris",
        "target": "test_data_german"
    },
    "dataset": {
        "label": {
            "name": "credit",
            "favorable_classes": [
                1
            ]
        },
        "protected_attributes": [
            {
                "name": "sex",
                "privileged_classes": [
                    "male"
                ]
            },
            {
                "name": "age",
                "privileged_classes": "eval: x > 25"
            }
        ],
        "categorical_features": [
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "other_debtors",
            "property",
            "installment_plans",
            "housing",
            "skill_level",
            "telephone",
            "foreign_worker"
        ],
        "features_to_keep": null,
        "features_to_drop": [
            "personal_status"
        ],
        "custom_preprocessing": "def custom_preprocessing(df):   # f: pd.DataFrame -> pd.DataFrame. default is None\n    status_map = {'\''A91'\'': '\''male'\'', '\''A93'\'': '\''male'\'', '\''A94'\'': '\''male'\'',\n                  '\''A92'\'': '\''female'\'', '\''A95'\'': '\''female'\''}\n    df['\''sex'\''] = df['\''personal_status'\''].replace(status_map)\n\n    return df"
    },
    "metric": {
        "privileged_groups": [
            {
                "age": 1
            }
        ],
        "unprivileged_groups": [
            {
                "age": 0
            }
        ],
        "metrics": [
            "mean_difference",
            "statistical_parity_difference",
            "disparate_impact"
        ]
    },
    "mitigation": {
        "algorithm": "reweighing"
    }
}'
```

```json
{
   "result": "SUCCESS",
   "metrics": {
      "before": {
         "mean_difference": -0.14944769330734242,
         "statistical_parity_difference": -0.14944769330734242,
         "disparate_impact": 0.7948260481712757
      },
      "after": {
         "mean_difference": 2.220446049250313e-16,
         "statistical_parity_difference": 2.220446049250313e-16,
         "disparate_impact": 1.0000000000000002
      }
   }
}
```

## 분류 예측 모델 O
    TPR이란 positive인데 positive로 prediction된 비율로 recall로도 불림
    FPR이란 negative인데 positive로 prediction된 비율

* Equalized odds: 각 그룹의 TPR과 FPR을 비교함(두 그룹의 차이가 작을 수록 공정한 알고리즘임)

        AIF360: Average Odds Difference

* Equality of opportunity: 각 그룹의 TPR 비교함 (두 그룹의 차이가 작을수록 공정한 알고리즘임)
  
    (Equalized odds의 완화된 공식)
    
        AIF360: Equal Opportunity Difference
  
* Statistical Parity: 각 그룹의 Positive class로 예측되는 확률이 비슷해야 함

        AIF360: Statistical Parity Difference
  
* Predictive Parity: 각 그룹의 Positive predictive value가 유사해야 됨

  (positive class로 예측된 것 중 실제 positive 비율)

## 분류 예측 모델 X
    분류 예측 모델이 없이 SMOTE를 이용하여 group bias가 얼마나 감소하는지 확인이 가능함.

* Group Bias(예시에서 Gender Bias): 두 그룹의 샘플 개수를 비교함. (소수그룹의)/(총 샘플 수)가 0.5에 가까워야 함

  (minor group을 oversampling 또는 major group을 under sampling)

# Mitigation Algorithms
