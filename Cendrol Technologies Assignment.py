#!/usr/bin/env python
# coding: utf-8

# # Cendrol Technologies Assignment - Q7
# 
# ### Problem Statement:
# The business problem tackled here is trying to improve customer service for `YourCabs.com`, a cab company in Bangalore.
# 
# The problem of interest is booking cancellations by the company due to unavailability of a car. The challenge is that cancellations can occur very close to the trip start time, thereby causing passengers inconvenience.
# 
# ### Goal:
# The goal of the competition is to create a predictive model for classifying new bookings as to whether they will eventually gets cancelled due to car unavailability.
# 
# ### Data Description
# - __id__ - booking ID
# - __user_id__ - the ID of the customer (based on mobile number)
# - __vehicle_model_id__ - vehicle model type.
# - __package_id__ - type of package (1=4hrs & 40kms, 2=8hrs & 80kms, 3=6hrs & 60kms, 4= 10hrs & 100kms, 5=5hrs & 50kms, 6=3hrs &
# 30kms, 7=12hrs & 120kms)
# - __travel_type_id__ - type of travel (1=long distance, 2= point to point, 3= hourly rental).
# - __from_area_id__ - unique identifier of area. Applicable only for point-to-point travel and packages
# - __to_area_id__ - unique identifier of area. Applicable only for point-to-point travel
# - __from_city_id__ - unique identifier of city
# - __to_city_id__ - unique identifier of city (only for intercity)
# - __from_date__ - time stamp of requested trip start
# - __to_date__ - time stamp of trip end
# - __online_booking__ - if booking was done on desktop website
# - __mobile_site_booking__ - if booking was done on mobile website
# - __booking_created__ - time stamp of booking
# - __from_lat__ - latitude of from area
# - __from_long__ - longitude of from area
# - __to_lat__ - latitude of to area
# - __to_long__ - longitude of to area
# - __Car_Cancellation (available only in training data)__ - whether the booking was cancelled (1) or not (0) due to unavailability of a car.
# - __Cost_of_error (available only in training data)__ - the cost incurred if the booking is misclassified. For an un-cancelled booking, the cost of misclassificaiton is 1. For a cancelled booking, the cost is a function of the cancellation time relative to the trip start time

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[73]:


df = pd.read_csv(r"C:\Users\Mohit\Downloads\YourCabs_training (1) (2) (1).csv")
df


# In[74]:


df.info()


# __As you can see there are various columns with missing values and some columns which need to be converted to date-time.__

# In[75]:


df.describe()


# In[76]:


df.nunique()


# In[77]:


df.shape


# In[78]:


df.columns


# ### Data Cleaning
# 
# - Change the format into the correct format
# - Drop the columns that are not contributing in the predictions. These are to be choosen on the basic observations and null value observations.

# In[79]:


# Converting "from_date" column from object to date-time

df['from_date'] = pd.to_datetime(df['from_date'])


# In[80]:


# Converting "to_date" column from float to date-time 
# Converting "booking_created" column from int to date-time

df['to_date'] = pd.to_datetime(df['to_date'])
df['booking_created'] = pd.to_datetime(df['booking_created'])

# Note: NaN will be converted to NaT( Not a timestamp )


# In[81]:


df.isnull().sum()


# In[82]:


# The "to_date" column has data in unknown units and has many null values too, thus we can remove it as it will not impact our goal.

df.drop(columns = "to_date", inplace = True)

# Other columns that can be removed in initial inspection - 
  # 'id', 'user_id', 'package_id', 'to_area_id', 'from_city_id', 'to_city_id'

df.drop(columns = ['id', 'user_id', 'package_id', 'to_area_id', 'from_city_id', 'to_city_id'], inplace= True)


# In[83]:


df.shape


# In[84]:


# Also, cost_of_error is not required in the prediction of Car_cancellation.
# Thus let's remove this column and make a new target dataframe for the output variable 

data = df.drop(['Car_Cancellation', 'Cost_of_error'], axis=1)
target = df[['Car_Cancellation']]


# In[85]:


data["vehicle_model_id"].value_counts(normalize=True)*100

# As, vehicle model id - 12 is covering for 73% of the entire column of vehicle_model_id


# In[86]:


# Dropping "vehicle_model_id" as single value has more percentage

data = data.drop(['vehicle_model_id'],axis=1) 


# In[87]:


data.shape


# In[88]:


data.columns


# In[89]:


data.head(5)


# ### Data Transformation

# #### from_lat and from_long columns

# In[91]:


# Filling null with median for Continuous data and mode for Discrete data

print(data['from_lat'].median())
print(data['from_long'].median())


# In[92]:


# let's examine what is the value of 'from_area_id' when the 'from_lat' value is its median.

data[data['from_lat'] == data['from_lat'].median()]

# Thus, from_area_id value is 1044.0


# In[93]:


# Filling Null values of 'from_lat' and 'from_long' with the median of 'from_lat' and 'from_long' respectively

data['from_lat'].fillna(data['from_lat'].median(), inplace=True)
data['from_long'].fillna(data['from_long'].median(), inplace=True)


# In[94]:


data.info()


# #### from_area_id column

# In[95]:


# As we have filled the null values of 'from_lat' and 'from_long' with their median,
# Now we will fill the null values of 'from_area_id' with the value which was 1044.0 

data['from_area_id'].fillna(data[data['from_lat'] == data['from_lat'].median()]['from_area_id'].max(), inplace=True)


# In[97]:


data.info()


# #### to_lat and to_long column

# In[105]:


# Filling null values of "to_lat" and "to_long" with median of each group formed based on from_area_id

data['to_lat'].fillna(data.groupby('from_area_id')['to_lat'].transform('median'), inplace=True)
data['to_long'].fillna(data.groupby('from_area_id')['to_long'].transform('median'), inplace=True)


# In[106]:


data.info()


# In[107]:


# Still some rows are left with null values in both the column

data[data['to_lat'].isnull()].head()


# In[110]:


data[data['to_lat'].isnull()]['from_area_id'].value_counts()


# In[118]:


data['to_lat'].fillna(data['to_lat'].median(), inplace=True)
data['to_long'].fillna(data['to_long'].median(), inplace=True)


# In[120]:


data.info()

# Data is cleaned with no null values


# # Data Modelling
# 
# ## Encoding 

# In[130]:


traveltype = pd.get_dummies(data['travel_type_id'],drop_first=True)
data = pd.concat([data,traveltype],axis=1)
data = data.drop(['travel_type_id'],axis=1)
data.rename(columns={2:'traveltype_pointtopoint',3:'traveltype_hourly'},inplace=True)


# In[131]:


data.head(2)


# In[127]:


pip install geopy


# In[132]:


from geopy import distance

def cal_distance(from_lat, from_long, to_lat, to_long):
    return distance.distance((from_lat, from_long), (to_lat, to_long)).km


# In[135]:


data['distance'] = data.apply(lambda row: cal_distance(row['from_lat'],row['from_long'],row['to_lat'],row['to_long']),axis=1)
data = data.drop(['from_lat','from_long','to_lat','to_long'],axis=1)


# In[136]:


data.head(4)


# ### Extracting date and time from timestamp

# In[137]:


data['from_date_dt'] = pd.to_datetime(data['from_date']).dt.strftime('%m/%d/%Y')
data['from_time_tm'] = pd.to_datetime(data['from_date']).dt.strftime('%H:%M')
data['booking_created_dt'] = pd.to_datetime(data['booking_created']).dt.strftime('%m/%d/%Y')
data['booking_created_tm'] = pd.to_datetime(data['booking_created']).dt.strftime('%H:%M')


# In[139]:


data['from_date_day'] = pd.to_datetime(data['from_date_dt']).dt.day_name()
data['booking_created_day'] = pd.to_datetime(data['booking_created_dt']).dt.day_name()
data['from_date_month'] = pd.to_datetime(data['from_date_dt']).dt.month_name()
data['booking_created_month'] = pd.to_datetime(data['booking_created_dt']).dt.month_name()
data['from_date_week'] = np.where((data['from_date_day']=='Saturday') | (data['from_date_day']=='Sunday'),'Weekend','Weekday')
data['booking_created_week'] = np.where((data['booking_created_day']=='Saturday') | (data['booking_created_day']=='Sunday'),'Weekend','Weekday')


# In[140]:


cond = [(pd.to_datetime(data['from_time_tm']).dt.hour.between(5, 8)),
        (pd.to_datetime(data['from_time_tm']).dt.hour.between(9, 12)),
        (pd.to_datetime(data['from_time_tm']).dt.hour.between(13, 16)),
        (pd.to_datetime(data['from_time_tm']).dt.hour.between(17, 20)),
        ((pd.to_datetime(data['from_time_tm']).dt.hour.between(21, 24)) | (pd.to_datetime(data['from_time_tm']).dt.hour==0)),
        (pd.to_datetime(data['from_time_tm']).dt.hour.between(1, 4))]
values = ['Early Morning','Morning','Afternoon','Evening','Night','Late Night']
data['from_date_session'] = np.select(cond,values)


# In[141]:


cond = [(pd.to_datetime(data['booking_created_tm']).dt.hour.between(5, 8)),
        (pd.to_datetime(data['booking_created_tm']).dt.hour.between(9, 12)),
        (pd.to_datetime(data['booking_created_tm']).dt.hour.between(13, 16)),
        (pd.to_datetime(data['booking_created_tm']).dt.hour.between(17, 20)),
        ((pd.to_datetime(data['booking_created_tm']).dt.hour.between(21, 24)) | (pd.to_datetime(data['booking_created_tm']).dt.hour==0)),
        (pd.to_datetime(data['booking_created_tm']).dt.hour.between(1, 4))]
values = ['Early Morning','Morning','Afternoon','Evening','Night','Late Night']
data['booking_created_session'] = np.select(cond,values)


# In[142]:


data['time_diff'] = (pd.to_datetime(data['from_date']) - pd.to_datetime(data['booking_created'])).astype('timedelta64[m]')


# In[143]:


data[data['time_diff'] < 0].head()


# In[144]:


data[data['time_diff'] < 0]['time_diff'].count()


# In[145]:


data = data.drop(['from_date','booking_created'],axis=1)


# In[146]:


data = data.drop(['from_date_dt','from_time_tm','booking_created_dt','booking_created_tm'],axis=1)


# In[147]:


data_merged = pd.concat([data,target],axis=1,join='inner')
data['from_area_id'] = round(data_merged.groupby('from_area_id')['Car_Cancellation'].sum()/data_merged.groupby('from_area_id')['Car_Cancellation'].count(),2)
data['from_area_id'].replace(np.nan,0,inplace=True)


# In[148]:


cond = [(data['from_area_id'].astype('float').between(0,0.33)),
        (data['from_area_id'].astype('float').between(0.34,0.66)),
        (data['from_area_id'].astype('float').between(0.67,1.0))]
values = ['Low Cancellation','Medium Cancellation','High Cancellation']
data['from_area_id'] = np.select(cond,values)


# In[149]:


data.head(4)


# ### Data Visualization

# In[150]:


data['from_area_id'].value_counts()


# In[153]:


plt.figure(figsize =(4,4))
for col in data.columns:
  if data[col].dtype == 'object':
    data[col].value_counts().plot.bar()
    plt.title(col)
    plt.show()


# ### Dividing data into Numerical and Categorical dataframes

# In[154]:


num = data.select_dtypes(include='number')
char = data.select_dtypes(include='object')


# In[155]:


num.head(2)


# In[156]:


char.head(2)


# ### Encoding Categorical variables

# In[157]:


X_char_dum = pd.get_dummies(char, drop_first = True)
X_char_dum.shape


# In[158]:


X_char_dum.head(3)


# In[159]:


data_all = pd.concat([num,X_char_dum],axis=1,join='inner')


# In[160]:


data_all.head(2)


# In[161]:


data_all.shape


# ### Train Test Split

# In[162]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_all,target,test_size=0.3,random_state=1)


# In[163]:


X_train.head()


# In[164]:


X_test.head()


# ### Classifiaction Model Building
# 

# In[167]:


import warnings
warnings.filterwarnings('ignore')


### 1. Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=4)
lr.fit(X_train,y_train)


# In[168]:


### 2. Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini',random_state=4)


# In[169]:


from sklearn.model_selection import GridSearchCV
param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250]}
gscv_dtc = GridSearchCV(dtc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_dtc.fit(X_train,y_train)


# In[170]:


gscv_dtc.best_params_


# In[171]:


dtc = DecisionTreeClassifier(criterion='gini',random_state=4,max_depth=7,min_samples_split=50)
dtc.fit(X_train,y_train)


# In[172]:


### 3. Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='gini',random_state=4)


# In[173]:


param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250]}
gscv_rfc = GridSearchCV(rfc, cv=10, param_grid=param_dist, n_jobs=-1)
gscv_rfc.fit(X_train,y_train)


# In[174]:


gscv_rfc.best_params_


# In[175]:


rfc=RandomForestClassifier(criterion='gini',random_state=4,max_depth=7,min_samples_split=50)
rfc.fit(X_train,y_train)


# ### Model Evaluation

# In[176]:


y_pred_lr=lr.predict(X_test)
y_pred_dtc=dtc.predict(X_test)
y_pred_rfc=rfc.predict(X_test)


# In[177]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[185]:


print('**Logistic Regression Metrics**')
print(' ')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_lr))
print('Precision:', metrics.precision_score(y_test, y_pred_lr))
print('Recall:', metrics.recall_score(y_test, y_pred_lr))
print('f1_score:', metrics.f1_score(y_test, y_pred_lr))


# In[186]:


print('**Decision Tree Metrics**')
print(' ')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_dtc))
print('Precision:', metrics.precision_score(y_test, y_pred_dtc))
print('Recall:', metrics.recall_score(y_test, y_pred_dtc))
print('f1_score:', metrics.f1_score(y_test, y_pred_dtc))


# In[187]:


print('**Random Forest Metrics**')
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rfc))
print('Precision:', metrics.precision_score(y_test, y_pred_rfc))
print('Recall:', metrics.recall_score(y_test, y_pred_rfc))
print('f1_score:', metrics.f1_score(y_test, y_pred_rfc))


# In[ ]:




