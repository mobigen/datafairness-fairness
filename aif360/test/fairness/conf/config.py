import os


# set input
data_dir = '/Users/psc/workspace/projects/2021/data-fairness/src/git/datafairness-fairness/aif360/test/data'
data_path = os.path.join(data_dir, 'german.csv')

# set df env
label_name = 'credit'       # model label values (column name)
favorable_classes = [1]     # mapped to 1(favorable), 0(unfavorable)
protected_attribute_names = ['sex', 'age']  # evaluate columns
privileged_classes = [['male'], lambda x: x > 25]   # each of privileged classes of protected attributes


## optional
categorical_features = [
    'status', 'credit_history', 'purpose',
    'savings', 'employment', 'other_debtors', 'property',
    'installment_plans', 'housing', 'skill_level', 'telephone',
    'foreign_worker'
]   # categorical feature names (column names) (able to expanded one-hot vector)
features_to_keep = []   # column names to keep.
                        # (all others are dropped except those present in
                        #    `protected_attribute_names`, `categorical_features`, `label_name` or `instance_weights_name`)
features_to_drop = ['personal_status']  # column names to drop.


def custom_preprocessing(df):   # f: DataFrame -> DataFrame. default is None
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['personal_status'].replace(status_map)

    return df



# SELECT BIAS METRICS

# # 세종대
# Average Odds Difference
# Equal Opportunity Difference
# Statistical Parity Difference
#
# # aif360 tutorial
# Statistical Parity Difference
# Equal Opportunity Difference
# Average Odds Difference
# Disparate Impact
# Theil Index




# SELECT MITIGATION ALGORITHMS

# # aif360 tutorial
# [pre-process] Reweighing
# [pre-process] Optimized Pre-Processing
# [in-process] Adversarial Debiasing
# [post-process] Reject Option Based Classification


