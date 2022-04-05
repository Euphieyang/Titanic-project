#!/usr/bin/env python
# coding: utf-8

# # Objects:

# 
# 1. Find an appropriate model for data prediction<br>
# 2. Find out the importance of features

# ## Find an appropriate model for data prediction

# ### 1. Import package

# In[30]:


#載入pandas
import pandas as pd

#載入邏輯斯回歸
from sklearn.linear_model import LogisticRegression

#載入決策數
from sklearn.tree import DecisionTreeClassifier

#載入隨機森林
from sklearn.ensemble import RandomForestClassifier

#切為訓練節測試集
from sklearn.model_selection import train_test_split

#ROC Curve
from sklearn.metrics import plot_roc_curve


# ### 2. 了解資料內容

# In[3]:


df=pd.read_csv('titanic.csv')
df.head()


# In[31]:


#了解資料
df.info()


# In[32]:


#性別做虛擬變數
df1=pd.get_dummies(df, columns=['Sex','Pclass'],drop_first=True)
df1.head()


# In[33]:


#確認資料處理
df1.info()


# #### 自變數X

# In[48]:


X=df1.iloc[:,[0,1,2,3,5,6,7]]


# #### 依變數Y

# In[49]:


#Y為survived: 0為沒有, 1為有
y=df['Survived']
print(y)


# # Machine Learning

# ## 訓練資料

# In[50]:


#以7:3的比例下去分
#設立每次所抓取的資料為固定同一群
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# ### 模型選擇

# #### 1. logisticregression

# In[53]:


#使用邏輯斯迴歸訓練模型
lr = LogisticRegression().fit(X_train, y_train)

#利用訓練好的模型去預測
y_lr_pred=lr.predict(X_test)


# #### 2. Decision Tree

# In[54]:


dt = DecisionTreeClassifier().fit(X_train, y_train)

y_dt_pred=dt.predict(X_test)


# #### 3. Random Forest

# In[55]:


rf=RandomForestClassifier().fit(X_train, y_train)

y_rf_pred=rf.predict(X_test)


# ## ROC Curve

# In[73]:


disp=plot_roc_curve(lr, X_test, y_test)
plot_roc_curve(dt, X_test, y_test, ax=disp.ax_);
plot_roc_curve(rf, X_test, y_test, ax=disp.ax_);

#從圖表可知隨機森林的表現最好，其次是邏輯斯回歸，最後才是決策樹
#可推測分類建議使用隨機森林模型進行預測會比較好


# ## Feature Importance

# ### 1. SelectKBest

# In[57]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[58]:


#score_func: 我們想要使用什麼模型來去選擇重要的特徵，算出特徵重要性
#k: 我們想要從data frame裡面選出幾個重要的特徵，這裡選7個
ordered_rank_features = SelectKBest(score_func = chi2, k=7)

#讓model fit X,y
ordered_feature=ordered_rank_features.fit(X,y)


# In[59]:


#製作新的Dataframe，新增一欄位
#算出feature score後，將ordered_feature.scores放入新增的欄位中
dfscores=pd.DataFrame(ordered_feature.scores_,columns=['Score'])

#將原先在X中的欄位放到新設定的dataframe中
dfcolumns=pd.DataFrame(X.columns)


#合併特徵跟分數
features_rank=pd.concat([dfcolumns,dfscores], axis=1)


# In[60]:


features_rank.columns=['Features','Score']
features_rank
#算出feature importance之後，將feature跟其分數印出來


# In[68]:


features_rank.nlargest(7,'Score')
#將算出來的feature importance根據分數大小進行排序
#可知Fare比其他特徵明顯重要，推測消費者較重視此項目


# ### 2. ExtraTreesClassifier

# In[69]:


from sklearn.ensemble import ExtraTreesClassifier
#將ExtraTreesClassifier引入程式
import matplotlib.pyplot as plt
#引入繪圖套件
model = ExtraTreesClassifier()

#使用ExtraTreesClassifier模型fit鐵達尼號乘客資料
model.fit(X,y)


# In[70]:


print(model.feature_importances_)
#使用ExtraTreesClassifier算出feature performance分數


# In[72]:


#index: feature name
#將每個feature name對應importance
ranked_feautures=pd.Series(model.feature_importances_, index=X.columns)
 
#畫出排名圖表
ranked_feautures.nlargest(7).plot(kind='barh', color='pink')
plt.show()


# ## Information Gain

# In[65]:


from sklearn.feature_selection import mutual_info_classif
#算出不純資訊(entropy)
#計算出資訊含量多不多


# In[66]:


mutual_info=mutual_info_classif(X,y)


# In[67]:


mutual_data=pd.Series(mutual_info, index=X.columns)
#index:X.columns即為x中的feature name
#將每個feature name對應的mutual_info算出來的資訊含量數值

mutual_data.sort_values(ascending=False)
#數值越高，資訊含量越高，由此表可知Sex_male最高
#可知Sex_male為 0.160252的資訊含量，以此類推


# In[ ]:





# In[ ]:





# In[ ]:




