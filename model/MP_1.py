#!/usr/bin/env python
# coding: utf-8

# # Data Loading

# In[9]:


#Data loading

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[10]:


df1 = pd.read_csv(r"C:\Users\Sneh\Documents\SEM6MATERIALS\Mini Project\Bengaluru_House_DataCSV.csv")
df1.head() 


# In[11]:


df1.shape


# In[12]:


df1.columns


# In[13]:


df1['area_type'].unique()


# In[14]:


df1['area_type'].value_counts()


# In[15]:


#dropping the column that we don't need for price prediction
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# In[16]:


df2.head()


# # Data Cleaning

# In[17]:


#checking for null values
df2.isnull().sum()


# In[18]:


df3 = df2.dropna()
df3.isnull().sum()


# In[19]:


df3.shape


# In[20]:


df3['size'].unique()


# In[21]:


#4 bedrooms and 4 bhk are indentical; so remove duplicate and make new feature(column)
# Feature Engineering
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[22]:


df3.head()


# In[23]:


df3[df3.bhk>20]


# In[24]:


df3.total_sqft.unique()


# In[25]:


#We have to do feature Engineering on total sqft column
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[26]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[27]:


#take avg of range
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[28]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(5)


# # Feature Engineering

# In[29]:


#Add new feature(Column) Price per sqft

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[30]:


#Explore Location column

len(df5.location.unique()) #high dimensity problem generated


# In[31]:


#clean space from location and count location using aggeregation

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[32]:


location_stats.values.sum()


# ### Dimensionality Reduction

# In[33]:


#put all the location with less then 10 data point into "Other" catagory
location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[34]:


len(df5.location.unique())


# In[35]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[36]:


df5.head(10)


# # Outlier Removal

# In[37]:


# let's assume minimum thresold per bhk to be 300 sqft

df5[df5.total_sqft/df5.bhk<300].head()


# In[38]:


df5.shape


# In[39]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# ### Outlier Removal Using Standard Deviation and Mean

# In[40]:


df6.price_per_sqft.describe()


# In[41]:


#min and ax value are showing wide variation; remove outliers per location using mean and one standard deviation

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.drop(['Country'], axis='columns', inplace=True)
df7.shape


# In[42]:


df7.head(5)


# In[43]:


#Visualisation for BHK property with price per sqft

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar") 


# In[44]:


plot_scatter_chart(df7,"Hebbal")


# In[45]:


# Outlier at 1700 sqft in chart of Rajaji nagar and 1300 in Hebbel; Price for 3 BHK is less then price for 2 BHK

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[46]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[47]:


plot_scatter_chart(df8,"Hebbal")


# ### Before and after outlier removel (Rajaji Nagar)
# 
# ![rajaji_nagar_outliers.png](attachment:rajaji_nagar_outliers.png)

# ### Before and after outlier removel (Hebbel)
# 
# ![hebbal_outliers.png](attachment:hebbal_outliers.png)

# In[48]:


#Histogram:-

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# ### Outlier Removal Using Bathrooms Feature

# In[49]:


df8.bath.unique()


# In[50]:


df8[df8.bath>10]


# In[51]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[52]:


# Let's assume that 2 more bathrooms than number of bedrooms in a home as a thresold value and remove oulier accordingly.

df8[df8.bath>df8.bhk+2]


# In[53]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[54]:


#drop column taht are not necessary
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(5)


# ## One Hot Encoding For Location (Dummies of Pandas)

# In[55]:


#Because ML Algo can't read location

dummies = pd.get_dummies(df10.location)
dummies.head(5)


# In[56]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns') #drop for prevent trape of dummy
df11.head()


# In[57]:


df12 = df11.drop('location',axis='columns')
df12.head(5)


# # Model Building

# In[58]:


df12.shape


# In[59]:


X = df12.drop(['price'],axis='columns') # x should only contain independent values
X.head(3)


# In[60]:


y = df12.price
y.head(3)


# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10) # 20% for testing amd 80% for model training


# In[62]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test) #Score of the ML Algo


# ### K Fold cross validation

# In[63]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# ### Finding best model using GridSearchCV

# In[64]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# #### From above table, It's certainly clear that linear regression is the best model to use for the prediction.

# ## Model Testing

# In[65]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[76]:


predict_price('Indira Nagar',1000, 2, 2)


# In[67]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[68]:


predict_price('Indira Nagar',1000, 2, 2)  # Posh Area


# In[69]:


predict_price('Indira Nagar',1000, 3, 3)


# # Export the model to a pickle file

# In[70]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# ### Export location and column information to a file as a JSON 

# In[71]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

