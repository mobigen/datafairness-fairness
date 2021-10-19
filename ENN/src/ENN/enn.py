import pandas as pd
import numpy as np
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from ENN import iris


# resample minority class.

recv_msg = {
    'table': 'test_data_diabetes',  # iris table name
    'label': 'outcome'              # label column name
}

df = iris.read_iris(recv_msg['table'])

label_counts = pd.DataFrame({'count': df[recv_msg['label']].value_counts()})

minority_class = label_counts['count'].argmin()
majority_class = label_counts['count'].argmax()


def as_base_type(val):
    if isinstance(val, np.int64):
        val = int(val)
    elif isinstance(val, np.float64):
        val = float(val)
    return val


minority_class = as_base_type(minority_class)
majority_class = as_base_type(majority_class)

label_counts['rates'] = round(label_counts / label_counts.sum(), 2)

results = {
    'name': recv_msg['label'],
    'minority_class': minority_class,
    'majority_class': majority_class,
    'counts': {
        'before': label_counts.to_dict()
    }
}

minority = df[df[recv_msg['label']] == minority_class].copy()
majority = df[df[recv_msg['label']] == majority_class].copy()

n = majority.shape[0] - minority.shape[0]

feature_columns = [c for c in df.columns if c != recv_msg['label']]

x = df.drop(recv_msg['label'], axis=1)
y = df[recv_msg['label']].values

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x, y)


count = 0
while count < n:
    # minority에서 random index/value 추출
    random_index = np.random.choice(minority.index, 1)[0]
    random_index_value = minority.loc[random_index, feature_columns]

    # random index의 값과 가장 가까운 index/value 추출
    # (해당 index가 minority에 속하지 않으면 re-sample)
    nn_index = knn.kneighbors([random_index_value])[1][0][1]
    if nn_index not in minority.index:
        continue
    nn_index_value = minority.loc[nn_index, feature_columns].values

    distance = nn_index_value - random_index_value

    # over-sampling
    new_index_value = random_index_value + np.random.rand() * distance
    new_index_value[recv_msg['label']] = minority_class
    df = df.append(new_index_value, ignore_index=True)

    count += 1

df[recv_msg['label']] = df[recv_msg['label']].astype(type(minority_class))

after_smote_x = df.drop(recv_msg['label'], axis=1)
after_smote_y = df[recv_msg['label']].values

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(after_smote_x, after_smote_y)

after_smote_y_pred = knn.predict(after_smote_x)
after_smote_y_answer = df.loc[after_smote_x.index][recv_msg['label']].values

issame = (after_smote_y_pred == after_smote_y_answer)
drop_index = after_smote_x.index[np.where(issame==False)]
df = df.drop(index=drop_index, axis=0)

label_counts = pd.DataFrame({'count': df[recv_msg['label']].value_counts()})
label_counts['rates'] = round(label_counts / label_counts.sum(), 2)
results['counts']['after'] = label_counts.to_dict()

# print(json.dumps(results, indent=2))
print(pd.DataFrame(results['counts']).T)



# todo: versus imblearn SMOTEENN
# from imblearn.combine import SMOTEENN
# from sklearn.ensemble import AdaBoostClassifier
# from imblearn.pipeline import Pipeline
#
# sme = SMOTEENN()
# x_res, y_res = sme.fit_resample(x_train, y_train)

