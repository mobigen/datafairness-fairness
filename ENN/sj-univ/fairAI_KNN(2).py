# %% md

### KNN에서 k값 설정

# %%

k = 3

# %%

import pandas as pd
import numpy as np

# %%

df = pd.read_csv('diabetes.csv')
df

# %%

(df['Outcome'] == 0).sum()

# %%

(df['Outcome'] == 1).sum()

# %% md

# 불균형데이터로 KNN

# %%

from sklearn import preprocessing

x_before = df.values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_after = min_max_scaler.fit_transform(x_before)
df = pd.DataFrame(x_after, columns=df.columns)
df['Outcome'] = df['Outcome'].astype(int)
df['Outcome'] = df['Outcome'].astype(int)
df

# %%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train['Outcome'] = df_train['Outcome'].astype(int)
df_test['Outcome'] = df_test['Outcome'].astype(int)

x_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome'].values
x_test = df_test.drop('Outcome', axis=1)
y_test = df_test['Outcome'].values

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred) * 100))
print('precision: %.2f%%' % (metrics.precision_score(y_test, y_pred) * 100))
print('recall: %.2f%%' % (metrics.recall_score(y_test, y_pred) * 100))

# %% md

# SMOTE-ENN방식

# %% md

## 1. (Start of SMOTE) Choose random data from the minority class.
## 2. Calculate the distance between the random data and its k nearest neighbors.
## 3. Multiply the difference with a random number between 0 and 1, then add the result to the minority class as a synthetic sample.
## 4. Repeat step number 2–3 until the desired proportion of minority class is met. (End of SMOTE)

# %%

minority = df_train[df_train['Outcome'] == 1].copy()
minority

# %%

n = (df_train['Outcome'] == 0).sum() - (df_train['Outcome'] == 1).sum()

minority_feature_columns = minority.columns[:-1]

count = 0

while True:
    random_index = np.random.randint(0, 767)
    # 랜덤으로 하나의 인덱스를 뽑는다.

    if random_index not in minority.index:
        continue
    # 만약 random_index가 minority라는 DataFrame에 없으면 다시 뽑는다.

    random_index_value = minority[minority_feature_columns].loc[random_index, :].values
    # 랜덤으로 뽑은 인덱스의 값을 받는다.

    nn_index = knn.kneighbors([minority[minority_feature_columns].loc[random_index, :].values])[1][0][1]
    # 랜덤으로 뽑은 인덱스의 값과 가장 가까운 인덱스를 받는다.

    if nn_index not in minority.index:
        continue
    # 만약 nn_index가 minority라는 DataFrame에 없으면 다시 뽑는다.

    nn_index_value = minority[minority_feature_columns].loc[nn_index, :].values
    ## 가장 가까운 인덱스의 값을 받는다.

    distance = nn_index_value - random_index_value
    ## 거리를 구한다.(diff라고 많이 하더라.)

    new_index_value = random_index_value + np.random.rand() * distance
    # oversampling 값을 뽑는다.

    new_index_value = np.append(new_index_value, 1)
    # oversampling 값에 마지막에 1(양성)을 추가시킨다.

    new_index = np.max(df_train.index) + 1  # 수정 완료
    # new_index는 매 step당 +2씩 증가 => step이 +2씩 증가하는 경우가 있었는데, 그렇게 되어지지 않도록 코드를 아예 수정했습니다.
    # new_index는 기존의 df_train의 index(data)를 덮어 씌워 raw data가 훼손됨 => df_train의 index의 max값 다음부터 추가하도록 수정하여 문제 해결했습니다.
    # oversapling 값에 index 번호를 매긴다.

    df_train.loc[new_index, :] = new_index_value
    # DataFrame에 oversampling된 값을 넣는다.

    count += 1

    if count == n:
        break
    # 양성데이터와 음성데이터가 1:1이 될 때까지 돌린다.

# %%

(df_train['Outcome'] == 0).sum()

# %%

(df_train['Outcome'] == 1).sum()

# %%

x_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome'].values

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred) * 100))
print('precision: %.2f%%' % (metrics.precision_score(y_test, y_pred) * 100))
print('recall: %.2f%%' % (metrics.recall_score(y_test, y_pred) * 100))

# %% md

## 5.(Start of ENN) Determine K, as the number of nearest neighbors. If not determined, then K=3.
## 6. Find the K-nearest neighbor of the observation among the other observations in the dataset, then return the majority class from the K-nearest neighbor.
## 7. If the class of the observation and the majority class from the observation’s K-nearest neighbor is different, then the observation and its K-nearest neighbor are deleted from the dataset.
## 8. Repeat step 2 and 3 until the desired proportion of each class is fulfilled. (End of ENN)

# %%

y_train_pred = knn.predict(x_train)
y_train_pred = y_train_pred.astype(int)

y_train_answer = df_train.loc[x_train.index]['Outcome'].values
y_train_answer = y_train_answer.astype(int)

issame = (y_train_pred == y_train_answer)

drop_index = x_train.index[np.where(issame == False)]

df_train = df_train.drop(index=drop_index, axis=0)
df_train

# %% md

## 결과물에서 unbalanced가 해결되지 않음 => 500:268로 unbalance하다고 의견주셨는데, 반복적으로 돌려가며 확인한 결과 아래와 같이 after enn도 양성과 음성의 개수가 balance하게 나타납니다.

# %%

(df_train['Outcome'] == 0).sum()

# %%

(df_train['Outcome'] == 1).sum()

# %%

x_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome'].values

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred) * 100))
print('precision: %.2f%%' % (metrics.precision_score(y_test, y_pred) * 100))
print('recall: %.2f%%' % (metrics.recall_score(y_test, y_pred) * 100))

# %% md

## 결과
### 불균형데이터 -> SMOTE -> SMOTE+ENN

# accuracy: 62.34 % -> 59.74 % -> 62.34 %
# precision: 46.67 % -> 44.74 % -> 47.50 %
# recall: 51.85 % -> 62.96 % -> 70.37 %
#
# 여러번 돌려본 결과
# accuracy: 60 초반 -> 50 후반 -> 60 초반
# precision: 40 중반 -> 40 중반 -> 40 후반
# recall: 50 초반 -> 60 초반 -> 70 초반
#
# 처음에 recall 값이 굉장히 낮았는데, 이는 소수 클래스 레이블을 올바르게 예측하기 위한 모델 성능이 충분하지 않음을 의미한다.
# 불균형데이터 -> SMOTE -> SMOTE + ENN으로 갈 수록 recall 값이 크게크게 늘어난다. accuracy와 precision은 거의 변동이 없다.

