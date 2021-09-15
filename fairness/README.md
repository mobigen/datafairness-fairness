# Fairness Metrics
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
