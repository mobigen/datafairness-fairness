# SMOTE-ENN
over-sampling(SMOTE) + under-sampling(ENN)

## openfx usage
```shell
curl GET 'http://fair5:31113/function/enn' \
--header 'Content-Type: application/json' \
--data-raw '{
    "table": "test_data_diabetes",
    "label": "outcome"
}'
```

```json
{
   "result": "SUCCESS",
   "unbalanced_class": {
      "name": "outcome",
      "minority_class": 1,
      "majority_class": 0,
      "counts": {
         "before": {
            "count": {
               "0": 499,
               "1": 268
            },
            "rates": {
               "0": 0.65,
               "1": 0.35
            }
         },
         "after": {
            "count": {
               "0": 437,
               "1": 435
            },
            "rates": {
               "0": 0.5,
               "1": 0.5
            }
         }
      }
   }
}
```