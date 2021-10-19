# coding: utf-8

# ### KNN에서 k값 설정

# In[1]:


k = 3

# In[2]:


import pandas as pd
import numpy as np

# In[3]:


df_raw = pd.read_csv('diabetes.csv')
df_raw

# In[4]:


(df_raw['Outcome'] == 0).sum()

# In[5]:


(df_raw['Outcome'] == 1).sum()

# # 불균형데이터로 KNN

# In[6]:


from sklearn import preprocessing

x_before = df_raw.values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_after = min_max_scaler.fit_transform(x_before)  # scaling -> (0~1)
df = pd.DataFrame(x_after, columns=df_raw.columns)
df['Outcome'] = df['Outcome'].astype(int)
# df['Outcome'] = df['Outcome'].astype(int)
df


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
# df_train['Outcome'] = df_train['Outcome'].astype(int)
# df_test['Outcome'] = df_test['Outcome'].astype(int)

x_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome'].values
x_test = df_test.drop('Outcome', axis=1)
y_test = df_test['Outcome'].values


vc = pd.DataFrame(df_train['Outcome'].value_counts())
vc['rates'] = round(vc['Outcome']/vc['Outcome'].sum(), 2)


knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred) * 100))
print('precision: %.2f%%' % (metrics.precision_score(y_test, y_pred) * 100))
print('recall: %.2f%%' % (metrics.recall_score(y_test, y_pred) * 100))

from visualize import t_sne
import matplotlib.pyplot as plt

# # SMOTE-ENN방식: combining over and under sampling
#   - SMOTE: over sampling ~ 무작위로 선택된 소수클래스들에서 각 knn을 찾음 -> 이웃들 사이의 직선상에서 데이터 생성
#   - ENN: under sampling ~ 소수 클래스 주변의 다수 클래스 제거

# ## 1. (Start of SMOTE) Choose random data from the minority class.
# ## 2. Calculate the distance between the random data and its k nearest neighbors.
# ## 3. Multiply the difference with a random number between 0 and 1, then add the result to the minority class as a synthetic sample.
# ## 4. Repeat step number 2–3 until the desired proportion of minority class is met. (End of SMOTE)

# In[8]:


minority = df_train[df_train['Outcome'] == 1].copy()
minority


# In[10]:


n = (df_train['Outcome'] == 0).sum() - (df_train['Outcome'] == 1).sum()

minority_feature_columns = minority.columns[:-1]

count = 0

while True:
    # # random_index = np.random.randint(0,767)
    # # 랜덤으로 하나의 인덱스를 뽑는다.
    #
    # if random_index not in minority.index:
    #     continue

    # 만약 random_index가 minority라는 DataFrame에 없으면 다시 뽑는다.

    random_index = np.random.choice(minority.index, 1)[0]
    # random_index_value = minority[minority_feature_columns].loc[random_index,:].values
    random_index_value = minority.loc[random_index, minority_feature_columns].values
    # 랜덤으로 뽑은 인덱스의 값을 받는다.

    # nn_index = knn.kneighbors([minority[minority_feature_columns].loc[random_index,:].values])[1][0][1]
    nn_index = knn.kneighbors([random_index_value])[1][0][1]
    # 랜덤으로 뽑은 인덱스의 값과 가장 가까운 인덱스를 받는다.

    if nn_index not in minority.index:
        continue
    # 만약 nn_index가 minority라는 DataFrame에 없으면 다시 뽑는다.

    nn_index_value = minority.loc[nn_index, minority_feature_columns].values
    ## 가장 가까운 인덱스의 값을 받는다.

    distance = nn_index_value - random_index_value
    ## 거리를 구한다.(diff라고 많이 하더라.)

    new_index_value = random_index_value + np.random.rand() * distance
    # oversampling 값을 뽑는다.

    new_index_value = np.append(new_index_value, 1)
    # oversampling 값에 마지막에 1(양성)(Outcome)을 추가시킨다.

    # new_index = count + len(df)
    new_index = np.max(df_train.index) + 1
    # oversampling 값에 index 번호를 매긴다.
    print(count, len(df), new_index, new_index in df_train.index)

    df_train.loc[new_index, :] = new_index_value
    # DataFrame에 oversampling된 값을 넣는다.ㅇ

    count += 1

    if count == n:
        break
    # 양성데이터와 음성데이터가 1:1이 될 때까지 돌린다.

# In[11]:


x_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome'].values

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred) * 100))
print('precision: %.2f%%' % (metrics.precision_score(y_test, y_pred) * 100))
print('recall: %.2f%%' % (metrics.recall_score(y_test, y_pred) * 100))


# chk: raw scatter plot
reduction_data = t_sne(x_train.to_numpy())

reduction_df = pd.DataFrame(reduction_data[:-count], index=x_train[:-count].index)
reduction_df['Outcome'] = df_train['Outcome'][:-count].astype(int)
color = np.where(reduction_df['Outcome'] == 0, 'red', 'blue')
reduction_df.plot(kind='scatter',
                  x=0, y=1,
                  c=color)
plt.title('raw data (scale)')
plt.show()


# chk: smote scatter plot
reduction_df = pd.DataFrame(reduction_data, index=x_train.index)
reduction_df['Outcome'] = df_train['Outcome'].astype(int)
color = np.where(reduction_df['Outcome'] == 0, 'red', 'blue')
reduction_df.plot(kind='scatter',
                  x=0, y=1,
                  c=color)
plt.title('after smote')
plt.show()

vc = pd.DataFrame(df_train['Outcome'].value_counts())
vc['rates'] = vc['Outcome']/vc['Outcome'].sum()




# ## 5.(Start of ENN) Determine K, as the number of nearest neighbors. If not determined, then K=3.
# ## 6. Find the K-nearest neighbor of the observation among the other observations in the dataset, then return the majority class from the K-nearest neighbor.
# ## 7. If the class of the observation and the majority class from the observation’s K-nearest neighbor is different, then the observation and its K-nearest neighbor are deleted from the dataset.
# ## 8. Repeat step 2 and 3 until the desired proportion of each class is fulfilled. (End of ENN)

# In[12]:


y_train_pred = knn.predict(x_train)
y_train_pred = y_train_pred.astype(int)

y_train_answer = df_train.loc[x_train.index]['Outcome'].values
y_train_answer = y_train_answer.astype(int)

issame = (y_train_pred == y_train_answer)

drop_index = x_train.index[np.where(issame == False)]

df_train = df_train.drop(index=drop_index, axis=0)
df_train

# chk: after scatter plot
enn_reduction_df = reduction_df.drop(index=drop_index, axis=0)
enn_reduction_df['Outcome'] = df_train['Outcome'].astype(int)
color = np.where(enn_reduction_df['Outcome'] == 0, 'red', 'blue')
enn_reduction_df.plot(kind='scatter',
                      x=0, y=1,
                      c=color)
plt.title('after enn')
plt.show()

vc = pd.DataFrame(df_train['Outcome'].value_counts())
vc['rates'] = round(vc['Outcome']/vc['Outcome'].sum(), 2)



# In[13]:


x_train = df_train.drop('Outcome', axis=1)
y_train = df_train['Outcome'].values

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred) * 100))
print('precision: %.2f%%' % (metrics.precision_score(y_test, y_pred) * 100))
print('recall: %.2f%%' % (metrics.recall_score(y_test, y_pred) * 100))

# chk: limit
#   only binary label data
#   smote+enn 수행 후 major:minor != 1:1
#       (smote == 1:1   =>   enn != 1:1)

# ## 결과
# ### 불균형데이터 -> SMOTE -> SMOTE+ENN
# 
# accuracy : 62.34% -> 59.74% -> 62.34%  
# precision: 46.67% -> 44.74% -> 47.50%  
# recall: 51.85% -> 62.96% -> 70.37%  
# 
# 여러번 돌려본 결과  
# accuracy : 60초반 -> 50후반 -> 60초반  
# precision: 40중반 -> 40중반 -> 40후반  
# recall: 50초반 -> 60초반 -> 70초반
# 
# 처음에 recall 값이 굉장히 낮았는데, 이는 소수 클래스 레이블을 올바르게 예측하기 위한 모델 성능이 충분하지 않음을 의미한다.  
# 불균형데이터 -> SMOTE -> SMOTE+ENN으로 갈 수록 recall 값이 크게크게 늘어난다.  
# accuracy와 precision은 거의 변동이 없다.

# In[ ]:




