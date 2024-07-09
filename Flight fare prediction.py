#!/usr/bin/env python
# coding: utf-8

# # Project:- "PRCP-1025-FlightPricePrediction"

# ## Team ID : PTID-CDS-JUL-23-1658

# ## Task 1:-Prepare a complete data analysis report on the given data.

# ## Problem Statement

# * Flight ticket prices can be something hard to guess, today we might see a price, check out the price of the same flight tomorrow, it will be a different story. We might have often heard travelers saying that flight ticket prices are so unpredictable. That’s why we will try to use machine learning to solve this problem. This can help airlines by predicting what prices they can maintain.
# 
# ## Features wise information
# 
# 1.	Airline: This column will have all the types of airlines like Indigo, Jet Airways, Air India, and many more.
# 2.	Date_of_Journey: This column will let us know about the date on which the passenger’s journey will start.
# 3.	Source: This column holds the name of the place from where the passenger’s journey will start.
# 4.	Destination: This column holds the name of the place to where passengers wanted to travel.
# 5.	Route: Here we can know about what the route is through which passengers have opted to travel from his/her source to their destination.
# 6.	Arrival_Time: Arrival time is when the passenger will reach his/her destination.
# 7.	Duration: Duration is the whole period that a flight will take to complete its journey from source to destination.
# 8.	Total_Stops: This will let us know in how many places flights will stop there for the flight in the whole journey.
# 9.	Additional_Info: In this column, we will get information about food, kind of food, and other amenities.
# 10.	Price: Price of the flight for a complete journey including all the expenses before onboarding.
# 
# 

# # Step 1:-Import Required Libraries

# In[270]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
from scipy.stats import stats
import os
import sweetviz as sv
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# # Step 2:- Data collections

# In[2]:


data=pd.read_excel("Flight_Fare.xlsx")
data.head(50)


# In[3]:


data["Additional_Info"].unique()


# In[4]:


purple="\033[4;35m"
green="\033[4;32m"
red="\033[4;31m"
cyan="\033[4;36m"
bold="\033[1m"
reset="\033[0m"
print(bold+purple+"Top 10 dataset:\n\n"+reset,data.head(10))
print(bold+green+"Last 10 dataset:\n\n"+reset,data.tail(10))
print(bold+red+"\nAs per dataset we conclude that having missing value in Additional_Info features.")


# In[5]:


print(bold+cyan+"Shape of dataset:"+reset,data.shape)
print(bold+red+"\nIn our dataset total 11 features and 10683 records.")


# In[6]:


print(bold+green+"Size of dataset:"+reset,data.size)


# # Step 3 :- Identify dependent and independent variables.

# In[2]:


print(bold+cyan+"Dependent variable is:- price")
print(bold+green+"\nIndependent variable is :- Airline,Date_of_Journey,Source,Destination,Route,Dep_Time,Arrival_Time,Duration,Total_Stops,Additional_Info") 


# # Step 4:- Exploratory Data Analysis(EDA)

# ### Task 1:-Prepare a complete data analysis report on the given data.
# 
# 

# In[8]:


data.info()
print(bold+red+"\nAs per data find out Route & Total_stops features having missing value and Only price is integer feature rest of all features are object features.")


# In[9]:


print(bold+cyan+"Total features of datasets:-\n\n"+reset+green,data.columns)


# In[10]:


data.describe().T


# In[11]:


print(bold+red+"Average price of tickets:-9087.06")
print(bold+green+"Mimimum price of tickets:-1759.0")
print(bold+purple+"Maximum price of tickets:-79512.0")
print(bold+cyan+"Standard deviation form mean:-4611.36")


# In[12]:


data.describe(include=['O']).T


# In[13]:


report=sv.analyze(data)
report.show_html()


# In[14]:


print(bold+green+"As per sweetviz report collect data:\n")
print(bold+cyan+"Maximum Airline data is jet airways.")
print(bold+cyan+"Total five city data available.Top most source journey at delhi.")
print(bold+cyan+"Total five city data available.Top most destination journey at cochin")
print(bold+cyan+"22% pessanger travel from Delhi → Mumbai → Cochin")
print(bold+cyan+"53% fight traveled one stop during whole journey")


# In[15]:


plt.figure(figsize=(30,40))
plotnumber=1
for i in data:
    if plotnumber<12:
        plot=plt.subplot(6,2,plotnumber)
        sns.histplot(data,x=data[i],kde=True)
        plt.xlabel(i,fontsize=30)
        plt.ylabel("Price",fontsize=30)
    plotnumber+=1
plt.tight_layout()


# In[16]:


plt.figure(figsize=(15,10))
sns.scatterplot(data=data,x='Price',y='Airline',hue='Price',palette='seismic',marker='D',size='Price',sizes=(20,80))
plt.xlabel("Price",fontsize=20)
plt.ylabel("Airline",fontsize=20)
plt.show()


# In[17]:


print(bold+green+"As per graph find out jet Airways Business class have highest ticket price")


# In[18]:


plt.figure(figsize=(15,10))
sns.scatterplot(data=data,x='Price',y='Source',hue='Price',palette='seismic',marker='D',size='Price',sizes=(20,80))
plt.xlabel("Price",fontsize=20)
plt.ylabel("Source",fontsize=20)
plt.show()


# In[19]:


print(bold+green+"Find out from graph Highest bussiness class ticket price spend by passanger at Banglore city")


# In[20]:


plt.figure(figsize=(15,10))
sns.scatterplot(data=data,x='Price',y='Destination',hue='Price',palette='seismic',marker='D',size='Price',sizes=(20,80))
plt.xlabel("Price",fontsize=20)
plt.ylabel("Destination",fontsize=20)
plt.show()


# In[21]:


print(bold+green+"Find out that airlines passanger travel from banglore to delhi with highest bussiness class ticket charge.")


# In[22]:


plt.figure(figsize=(15,15))
ad=sns.barplot(data=data,x="Price",y="Date_of_Journey",palette="seismic",ci=0)
for p in ad.patches:
    y=p.get_y()+p.get_height()/2
    x=p.get_width()
    ad.annotate(f"{int(x)}",(x,y),ha="center",rotation=0,fontsize=20)
plt.xlabel("Price",fontsize=20,)
plt.ylabel("Date of Journey",fontsize=20)
plt.show()


# In[23]:


print(bold+green+"Ticket price is highest on 1 march 2019")


# # Step 5:- Data Preprocessing

# ### Fix the dublicate value in our datasets

# In[23]:


Dup_data=data.duplicated().sum()
print(bold+cyan+"Duplicate value of our datasets:-",Dup_data)


# In[24]:


data.duplicated().value_counts()


# In[25]:


print(bold+green+"Check location of duplicate value:-\n")
data.loc[data.duplicated()]


# In[26]:


print(bold+cyan+"Drop duplicated value in datase.")
data.drop_duplicates(inplace=True,keep="first")


# In[27]:


print(bold+red+"Reverify shape of dataset.")
data.shape


# ### Fix missing value in our datasets

# In[28]:


print(bold+green+"Total null value:-\n"+reset,data.isna().sum())
print(bold+purple+"Route and total stop have missing value")


# In[29]:


print(bold+purple+"Check missing location in Route features.")
data.loc[data["Route"].isna()]


# In[30]:


print(bold+purple+"Check missing location in Total_Stops features.")
data.loc[data["Total_Stops"].isna()]


# In[31]:


print(bold+purple+"Both value in same record missing so drop this records.")
data.dropna(inplace=True)


# In[32]:


print(bold+green+"Verified null value in datasets:-\n"+reset,data.isna().sum())


# ### Data Type Conversion

# In[33]:


data.info()


# In[34]:


print(bold+green+"Convert Date of journey into seperate column of Day, Month and Year.")
data["Day_Journey"]=pd.to_datetime(data["Date_of_Journey"]).dt.day
data["Month_Journey"]=pd.to_datetime(data["Date_of_Journey"]).dt.month
data["Year_Journey"]=pd.to_datetime(data["Date_of_Journey"]).dt.year


# In[35]:


print(bold+purple+"Verify Date of Journey features.")
data.head(1)


# In[36]:


print(bold+cyan+"Convert Duration of huor into minutes.")
data["Duration"]=data["Duration"].str.replace("h","*60").str.replace(" ","+").str.replace("m","*1").apply(eval)


# In[37]:


print(bold+purple+"Verify Duration features.")
data.head(1)


# In[38]:


print(bold+green+"Convert Dep_Time convert into seperate column as hour and minute.")
data["Dep_Hour"]=pd.to_datetime(data["Dep_Time"]).dt.hour
data["Dep_Minute"]=pd.to_datetime(data["Dep_Time"]).dt.minute


# In[39]:


print(bold+red+"Verify added features Dep_Hour and Dep_Minute")
data.head(1)


# In[40]:


print(bold+green+"Convert Dep_Time convert into seperate column as hour and minute.")
data["Arrival_Hour"]=pd.to_datetime(data["Arrival_Time"]).dt.hour
data["Arrival_Minute"]=pd.to_datetime(data["Arrival_Time"]).dt.minute


# In[41]:


print(bold+red+"Verify added features Arrival_Hour and Arrival_Minute")
data.head(1)


# In[42]:


print(bold+red+"Drop not required features.")
data.drop(["Date_of_Journey","Dep_Time","Arrival_Time"],axis=1,inplace=True)


# In[43]:


print(bold+green+"Verify drop all features.")
data.head(1)


# ### Encoding process

# In[44]:


print(bold+purple+"Categorical data convert into numerical features.")
le=LabelEncoder()
data["Airline"]=le.fit_transform(data["Airline"])
data["Source"]=le.fit_transform(data["Source"])
data["Destination"]=le.fit_transform(data["Destination"])
data["Route"]=le.fit_transform(data["Route"])
data["Total_Stops"]=le.fit_transform(data["Total_Stops"])
data["Additional_Info"]=le.fit_transform(data["Additional_Info"])


# In[45]:


print(bold+green+"Verify all features with encoding data")
data.head()


# In[46]:


print(bold+green+"1) Unique value of Airline is:-",data.Airline.unique())
print(bold+purple+"2) Unique value of Source is:-",data.Source.unique())
print(bold+red+"3) Unique value of Destination is:-",data.Destination.unique())
print(bold+cyan+"4) Unique value of Route is:-",data.Route.unique())
print(bold+green+"5) Unique value of Total_Stops is:-",data.Total_Stops.unique())
print(bold+purple+"6) Unique value of Additional_Info is:-",data.Additional_Info.unique())


# ### FIX OUTLIERS IN DATASETS

# In[48]:


plt.figure(figsize=(15,15))
plotnumber=1
for i in data:
    if plotnumber < 16:
        plot=plt.subplot(4,4,plotnumber)
        sns.boxplot(data=data,x=data[i])
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[49]:


print(bold+purple+"Boxplot graph we find out the features like Airline, Source, Duration, Additional info, price, Month_journey are presence outliers.")


# ### Airline

# In[48]:


data["Airline"].unique()


# In[49]:


print(bold+green+"Outliers are presence in Airline features")
plt.figure(figsize=(10,5))
sns.boxplot(data=data,x="Airline",color="r",showmeans=True)
plt.xlabel("Airline",fontsize=20)
plt.show()


# In[50]:


IQR=stats.iqr(data.Airline,interpolation="midpoint")
print("IQR:",IQR)

Q1=data.Airline.quantile(0.25)
print("Q1:",Q1)

Q3=data.Airline.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-3*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+3*IQR
print("Upper_limit:",Upper_limit)


# In[51]:


print(bold+purple+"As per graph data not normaly distributed.")
plt.figure(figsize=(15,5))
sns.histplot(data=data,x="Airline",kde=True)
plt.xlabel("Airline",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[52]:


print(bold+cyan+"Not have outliers below lower limit.")
a=data.loc[data["Airline"]<Lower_limit]
a


# In[53]:


print(bold+cyan+"Outliers present in above Upper limit.")
b=data.loc[data["Airline"]>Upper_limit]
b


# In[54]:


A_r=((len(a+b))/len(data))*100
print(bold+green+"Total outliers in Airlines features: {:.2f}%".format(A_r))


# In[55]:


data.loc[data["Airline"]>Upper_limit,"Airline"]=data["Airline"].median()


# In[56]:


print(bold+green+"Outliers does not available in features.")
data.loc[data["Airline"]>Upper_limit]


# ### Source

# In[57]:


data["Source"].unique()


# In[58]:


print(bold+green+"Outliers are presence in Source features")
plt.figure(figsize=(10,5))
sns.boxplot(data=data,x="Source",color="r",showmeans=True)
plt.xlabel("Source",fontsize=20)
plt.show()


# In[59]:


IQR=stats.iqr(data.Source,interpolation="midpoint")
print("IQR:",IQR)

Q1=data.Source.quantile(0.25)
print("Q1:",Q1)

Q3=data.Source.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)


# In[60]:


print(bold+purple+"As per graph data not normaly distributed.")
plt.figure(figsize=(15,5))
sns.histplot(data=data,x="Source",kde=True)
plt.xlabel("Source",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[61]:


print(bold+green+"Outliers are presence below lower limit.")
c=data.loc[data["Source"]<Lower_limit]
c


# In[62]:


print(bold+green+"Not have outliers are presence above upper limit.")
d=data.loc[data["Source"]>Upper_limit]
d


# In[63]:


Sour=(len(c+d)/len(data))*100
print(bold+purple+"Total percentage of outliers in source features: {:.2f}%".format(Sour))


# In[64]:


data.loc[data["Source"]<Lower_limit,"Source"]=data["Source"].median()


# In[65]:


print(bold+green+"Not outliers are presence below lower limit")
data.loc[data["Source"]<Lower_limit]


# ### Destination
data["Destination"].unique()print(bold+green+"Outliers does not presence in Destination features")
plt.figure(figsize=(10,5))
sns.boxplot(data=data,x="Destination",color="r",showmeans=True)
plt.xlabel("Destination",fontsize=20)
plt.show()IQR= stats.iqr(data["Destination"],interpolation="midpoint")
print("IQR:", IQR)

Q1= data["Destination"].quantile(0.25)
print("Q1:",Q1)

Q3= data["Destination"].quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)print(bold+cyan+"data does not normally distributed.")
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="Destination",kde=True)
plt.xlabel("Destination",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()print(bold+green+"Outliers are not presence below lower limit.")
e=data.loc[data["Destination"]<Lower_limit]
eprint(bold+green+"Outliers are not presence above upper limit.")
f=data.loc[data["Destination"]>Upper_limit]
f
# ### Route
data["Route"].unique()print(bold+green+"Outliers does not presence in route features")
plt.figure(figsize=(10,5))
sns.boxplot(data=data,x="Route",color="r",showmeans=True)
plt.xlabel("Route",fontsize=20)
plt.show()IQR= stats.iqr(data["Route"],interpolation="midpoint")
print("IQR:", IQR)

Q1= data["Route"].quantile(0.25)
print("Q1:",Q1)

Q3= data["Route"].quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)print(bold+cyan+"data does not normally distributed.")
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="Route",kde=True)
plt.xlabel("Route",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()print(bold+green+"Outliers are not presence below lower limit.")
g=data.loc[data["Route"]<Lower_limit]
gprint(bold+green+"Outliers are not presence above upper limit.")
h=data.loc[data["Route"]>Upper_limit]
h
# ### Duration

# In[66]:


data["Duration"].unique()


# In[67]:


print(bold+green+"Outliers are presence in Duration feature")
plt.figure(figsize=(10,5))
sns.boxplot(data=data,x="Duration",color="r",showmeans=True)
plt.xlabel("Duration",fontsize=20)
plt.show()


# In[68]:


IQR= stats.iqr(data["Duration"],interpolation="midpoint")
print("IQR:", IQR)

Q1= data["Duration"].quantile(0.25)
print("Q1:",Q1)

Q3= data["Duration"].quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)


# In[69]:


print(bold+cyan+"data does not normally distributed and right side skewed.")
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="Duration",kde=True)
plt.xlabel("Duration",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[70]:


print(bold+green+"Outliers are not presence below lower limit.")
i=data.loc[data["Duration"]<Lower_limit]
i


# In[71]:


print(bold+red+"Outliers are presence above upper limit.")
j=data.loc[data["Duration"]>Upper_limit]
j


# In[72]:


Dur=((len(i+j))/len(data))*100
print(bold+green+"Total outliers in Duration features: {:.2f}%".format(Dur))


# In[73]:


data.loc[data["Duration"]>Upper_limit,"Duration"]=data["Duration"].median()


# In[74]:


print(bold+green+"Outliers are not presence above upper limit.")
data.loc[data["Duration"]>Upper_limit]


# ### Total_Stops
data["Total_Stops"].unique()plt.figure(figsize=(15,10))
sns.boxplot(data=data,x="Total_Stops",showmeans=True,color="r")
plt.xlabel("Total_Stops",fontsize=20)
plt.ylabel("Total_Stops",fontsize=20)
plt.show()IQR=stats.iqr(data["Total_Stops"],interpolation="midpoint")
print("IQR:",IQR)
# ### Additional_Info

# In[75]:


data["Additional_Info"].unique()


# In[76]:


plt.figure(figsize=(15,10))
sns.boxplot(data=data,x="Additional_Info",showmeans=True,color="r")
plt.xlabel("Additional_Info",fontsize=20)
plt.ylabel("Additional_Info",fontsize=20)
plt.show()


# In[77]:


IQR= stats.iqr(data["Additional_Info"],interpolation="midpoint")
print("IQR:", IQR)

Q1= data["Additional_Info"].quantile(0.25)
print("Q1:",Q1)

Q3= data["Additional_Info"].quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)


# In[78]:


print(bold+cyan+"data does not normally distributed.")
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="Additional_Info",kde=True)
plt.xlabel("Additional_Info",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[79]:


print(bold+green+"Outliers are presence below lower limit.")
k=data.loc[data["Additional_Info"]<Lower_limit]
k


# In[80]:


print(bold+red+"Outliers are presence above upper limit.")
l=data.loc[data["Additional_Info"]>Upper_limit]
l


# In[81]:


Add_info=(len(k+l)/len(data))*100
print(bold+cyan+"Total outliers in Additional_info features: {:.2f}%".format(Add_info))


# In[82]:


data.loc[data["Additional_Info"]<Lower_limit,"Additional_Info"]=data["Additional_Info"].median()


# In[83]:


print(bold+green+"No outliers are presence in additional_info features.")
data.loc[data["Additional_Info"]<Lower_limit]


# ### Price

# In[84]:


data["Price"].unique()


# In[85]:


plt.figure(figsize=(15,10))
sns.boxplot(data=data,x="Price",showmeans=True,color="r")
plt.xlabel("Price",fontsize=20)
plt.ylabel("Price",fontsize=20)
plt.show()


# In[86]:


IQR= stats.iqr(data["Price"],interpolation="midpoint")
print("IQR:", IQR)

Q1= data["Price"].quantile(0.25)
print("Q1:",Q1)

Q3= data["Price"].quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)


# In[87]:


print(bold+cyan+"data does not normally distributed and graph right side skewed.")
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="Price",kde=True)
plt.xlabel("Price",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[88]:


print(bold+green+"Outliers are not presence below lower limit.")
m=data.loc[data["Price"]<Lower_limit]
m


# In[89]:


print(bold+red+"Outliers are presence above upper limit.")
n=data.loc[data["Price"]>Upper_limit]
n


# In[90]:


Price=((len(m+n))/len(data))*100
print(bold+green+"Total outliers in price features: {:.2f}%".format(Price))


# In[91]:


data.loc[data["Price"]>Upper_limit,"Price"]=data["Price"].median()


# In[92]:


print(bold+green+"No outliers are presence in price features.")
data.loc[data["Price"]>Upper_limit]


# ### Month_Journey

# In[93]:


data["Month_Journey"].unique()


# In[94]:


plt.figure(figsize=(15,10))
sns.boxplot(data=data,x="Month_Journey",showmeans=True,color="r")
plt.xlabel("Month_Journey",fontsize=20)
plt.ylabel("Month_Journey",fontsize=20)
plt.show()


# In[95]:


IQR= stats.iqr(data["Month_Journey"],interpolation="midpoint")
print("IQR:", IQR)

Q1= data["Month_Journey"].quantile(0.25)
print("Q1:",Q1)

Q3= data["Month_Journey"].quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1.5*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper_limit:",Upper_limit)


# In[96]:


print(bold+cyan+"data does not normally distributed.")
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="Month_Journey",kde=True)
plt.xlabel("Month_Journey",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[97]:


print(bold+green+"Outliers are not presence below lower limit.")
o=data.loc[data["Month_Journey"]<Lower_limit]
o


# In[98]:


print(bold+red+"Outliers are presence above upper limit.")
p=data.loc[data["Month_Journey"]>Upper_limit]
p


# In[99]:


month_journey=((len(o+p))/len(data))*100
print(bold+green+"Total outliers in price features: {:.2f}%".format(month_journey))


# In[100]:


data.loc[data["Month_Journey"]>Upper_limit,"Month_Journey"]=data["Month_Journey"].median()


# In[101]:


print(bold+green+"No outliers are presence in Month_Journey features.")
data.loc[data["Month_Journey"]>Upper_limit]


# # Step:- 6 Feature Selection

# In[102]:


data.describe().T


# In[103]:


print(bold+red+"Year_journey's std value is zero so drop this features.")
data.drop("Year_Journey",axis=1,inplace=True)


# In[104]:


data.corr()


# In[105]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True,color="r")


# In[106]:


print(bold+green+"No highly coreleated features between eachother.")


# In[107]:


data.drop("Additional_Info",axis=1,inplace=True)


# # Task 2:-Create a predictive model which will help the customers to predict future flight prices and plan their journey accordingly.

# # Step 7:- Model selection and Building

# In[108]:


print(bold+red+"Splitting data in X and y from")
X=data.drop("Price",axis=1)
y=data.Price


# In[109]:


print(bold+green+"After splitting of X data shape:",X.shape)
print(bold+green+"After splitting of y data shape:",y.shape)


# In[110]:


print(bold+purple+"Splitting data in train and test form.")
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[111]:


print(bold+green+"testing data of X :\n"+reset,X_test)
print(bold+green+"\nTesting data of y:\n"+reset,y_test)


# In[112]:


print(bold+green+"After splitting of X data shape:"+reset,X_train.shape)
print(bold+green+"After splitting of y data shape:"+reset,y_train.shape)
print(bold+green+"After splitting of X data shape:"+reset,X_test.shape)
print(bold+green+"After splitting of y data shape:"+reset,y_test.shape)


# ## Data Normalizing Techniques

# In[113]:


X_train_scale=X_train.copy()
X_test_scale=X_test.copy()
scale=StandardScaler()
# Apply MinMaxscaler
X_train_scale=scale.fit_transform(X_train)
X_test_scale=scale.transform(X_test)


# In[114]:


data.describe()


# ## Linear Regression

# In[172]:


lr=LinearRegression()
Linear_reg=lr.fit(X_train_scale,y_train)
Linear_reg


# In[173]:


y_train_pred=lr.predict(X_train_scale)
y_test_pred=lr.predict(X_test_scale)


# ## Evalution of Linear regression

# In[174]:


Training_r2=r2_score(y_train,y_train_pred)*100
print(bold+green+"R2 score in training data {:.2f}%".format(Training_r2))
Testing_r2=r2_score(y_test,y_test_pred)*100
print(bold+cyan+"R2 score in testing data {:.2f}%".format(Testing_r2))
l1=r2_score(y_test,y_test_pred)


# In[175]:


Test_MSE=mean_squared_error(y_test,y_test_pred)
print(bold+green+"Testing data MSE score: ",Test_MSE)
Test_MAE=mean_absolute_error(y_test,y_test_pred)
print(bold+red+"Testing data MAE score: ",Test_MAE)
Test_RMSE=np.sqrt(Test_MSE)
print(bold+purple+"Testing data RMSE score: ",Test_RMSE)
adjusted_R2_score=(1-(((1-l1)*(2093-1))/(2093-12-1)))*100
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score)


# # Lasso Regression

# In[186]:


lasso= Lasso(alpha=0.5,selection="random")
lasso_model=lasso.fit(X_train_scale,y_train)
lasso_model


# In[187]:


# Prediction of training data
y_train_lasso_predict=lasso.predict(X_train_scale)
# Prediction of testing data
y_test_lasso_predict=lasso.predict(X_test_scale)


# ## Evalution of Lasso Regression

# In[188]:


Training_lasso_r2=r2_score(y_train,y_train_lasso_predict)*100
print(bold+green+"R2 score for training data: {:.2f}%".format (Training_lasso_r2))
Testing_lasso_r2=r2_score(y_test,y_test_lasso_predict)*100
print(bold+red+"R2 score for testing data: {:.2f}%".format (Testing_lasso_r2))
l2=r2_score(y_test,y_test_lasso_predict)


# In[191]:


Test_MAE_lasso=mean_absolute_error(y_test,y_test_lasso_predict)
print(bold+green+"Testing data MAE score:",Test_MAE_lasso)
Test_MSE_lasso=mean_squared_error(y_test,y_test_lasso_predict)
print(bold+red+"Testing data MSE score:",Test_MSE_lasso)
Test_RMSE_lasso=np.sqrt(Test_MSE_lasso)
print(bold+cyan+"Testing data RMSE score:",Test_RMSE_lasso)
adjusted_R2_score_lasso=(1-(((1-l2)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_lasso)


# # Ridge regression 

# In[189]:


ridge=Ridge()
ridge.fit(X_train_scale,y_train)
# predict train data
y_train_pred=ridge.predict(X_train_scale)
# predict test data
y_test_pred=ridge.predict(X_test_scale)


# ## Evalution of Ridge Regression

# In[190]:


Training_ridge=r2_score(y_train,y_train_pred)*100
print(bold+red+"R2 score of training dataset ridge regression score: {:.2f}%".format(Training_ridge))
Test_ridge=r2_score(y_test,y_test_pred)*100
print(bold+green+"R2 score of testing dataset ridge regression score: {:.2f}%".format(Test_ridge))


# In[193]:


Test_MAE_ridge=mean_absolute_error(y_test,y_test_pred)
print(bold+green+"Testing data MAE score:",Test_MAE_ridge)
Test_MSE_ridge=mean_squared_error(y_test,y_test_pred)
print(bold+red+"Testing data MSE score:",Test_MSE_ridge)
Test_RMSE_ridge=np.sqrt(Test_MSE_ridge)
print(bold+cyan+"Testing data RMSE score:",Test_RMSE_ridge)
adjusted_R2_score_ridge=(1-(((1-l2)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_ridge)


# # Random forest

# In[159]:


# initialize model
RF=RandomForestRegressor()
#Train model
RF.fit(X_train_scale,y_train)
# Predict training data
y_train_predict=RF.predict(X_train_scale)
# Predict testing data
y_test_predict=RF.predict(X_test_scale)


# ## Evalution of Random forest regressor

# In[161]:


Training_RF=r2_score(y_train,y_train_predict)*100
print(bold+cyan+"R2 score for training data: {:.2f}%".format(Training_RF))
Testing_RF=r2_score(y_test,y_test_predict)*100
print(bold+green+"R2 score for testing data: {:.2f}%".format(Testing_RF))
RF1=r2_score(y_test,y_test_predict)


# In[162]:


RF_MAE=mean_absolute_error(y_test,y_test_predict)
print(bold+purple+"Testing data MAE score: ",RF_MAE)
RF_MSE=mean_absolute_error(y_test,y_test_predict)
print(bold+red+"Testing data MSE score:",RF_MSE)
RF_RMSE=np.sqrt(RF_MSE)
print(bold+green+"Testing data RMSE score:",RF_RMSE)
adjusted_R2_score_RF=(1-(((1-RF1)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_RF)


# ## Random Forest HyperParameter Tuning

# In[163]:


# Random search used setup a grid of hyperparameter value and selects random combination of train the model and score.
# Model 
rf_reg=RandomForestRegressor(random_state=40)

criterion=["mse","rmse","friedmen_mse"]
n_estimators= [int(x) for x in np.linspace(start=100,stop=1000,num=15)] # Number of trees used random forest.
max_features= ['auto','sqrt','log2'] # Maximum number of features allowed to try in individual tree
max_depth=[int(x) for x in np.linspace(start=2,stop=90,num=15)]  # Maximum number of depth of iteration
min_sample_split= [2,5,8,10,12]  # Minimum no of sample split 
min_sample_leaf= [1,2,3,4,5,6]  # Minimum number of sample of leaf split
bootstrap= [True,False] # Sampling 

#dictionary for hyperparameters
random_grid={'criterion':criterion,'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
             'min_samples_split':min_sample_split,'min_samples_leaf':min_sample_leaf,'bootstrap':bootstrap}

rf_reg_hyp= RandomizedSearchCV(estimator=rf_reg,scoring='r2',param_distributions=random_grid,n_iter=100,cv=2,
                              verbose=1,n_jobs=-1)
# build training model
rf_reg_hyp.fit(X_train_scale,y_train)
# Best parameter 
rf_best_param=rf_reg_hyp.best_params_

print(f'Best parameters: {rf_best_param}')


# In[164]:


print(bold+cyan+"Impute best parameter")
rf_reg_param=RandomForestRegressor(n_estimators= 228,min_samples_split= 10,min_samples_leaf= 1,max_features= 'sqrt',max_depth= 83,criterion='mse',bootstrap=False)


# In[165]:


# Model fit to data
rf_reg_param=rf_reg_param.fit(X_train_scale,y_train)
# predict data with training data
rf_train_pred_param=rf_reg_param.predict(X_train_scale)
# Predict data with testing data
rf_test_pred_param=rf_reg_param.predict(X_test_scale)
rf_test_pred_param


# ## Evalution of Random Forest Hyperparameter Tuning

# In[166]:


Train_rf_param=r2_score(y_train,rf_train_pred_param)*100
print(bold+cyan+"R2 score testing data: {:.2f}%".format(Train_rf_param))
Test_rf_param=r2_score(y_test,rf_test_pred_param)*100
print(bold+green+"R2 score testing data: {:.2f}%".format(Test_rf_param))
RF2=r2_score(y_test,rf_test_pred_param)


# In[167]:


RF_hyp_MAE=mean_absolute_error(y_test,rf_test_pred_param)
print(bold+purple+"Testing data MAE score:",RF_hyp_MAE)
RF_hyp_MSE=mean_absolute_error(y_test,rf_test_pred_param)
print(bold+red+"Testing data MSE score:",RF_hyp_MSE)
RF_hyp_RMSE=np.sqrt(RF_MSE)
print(bold+green+"Testing data RMSE score:",RF_hyp_RMSE)
adjusted_R2_score_RF_hyp=(1-(((1-RF2)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_RF_hyp)


# # Decision Tree

# In[194]:


# initialise model
DT=DecisionTreeRegressor()
# Model build
DT.fit(X_train_scale,y_train)
# Predict training data
DT_y_train_predict=DT.predict(X_train_scale)
# Predict testing data
DT_y_test_predict=DT.predict(X_test_scale)


# ## Evalution of Decision Tree Regressor

# In[195]:


Training_DT=r2_score(y_train,DT_y_train_predict)*100
print(bold+green+"Trainind data R2 score: {:.2f}%".format(Training_DT))
Testing_DT=r2_score(y_test,DT_y_test_predict)*100
print(bold+cyan+"Testing data R2 score: {:.2f}%".format(Testing_DT))
DT1=r2_score(y_test,DT_y_test_predict)


# In[196]:


DT_MAE=mean_absolute_error(y_test,y_test_predict)
print(bold+purple+"Testing data MAE score: ",DT_MAE)
DT_MSE=mean_absolute_error(y_test,y_test_predict)
print(bold+red+"Testing data MSE score: ",DT_MSE)
DT_RMSE=np.sqrt(DT_MSE)
print(bold+green+"Testing data RMSE score:",DT_RMSE)
adjusted_R2_score_DT=(1-(((1-DT1)*(2093-1))/(2093-12-1)))*100
print(bold+cyan+"Testing data adjusted R2 score:{:.2f}%".format(adjusted_R2_score_DT))


# ## Decision Tree Hyperparameter Tuning

# In[198]:


DT_reg= DecisionTreeRegressor(random_state=10,criterion='mse') # Object create decision tree with random state.
param={
       "splitter":("best","random"),     #  
       "max_depth":(list(range(1,30))),  # Max depth of tree
       "min_samples_split":[2,3,4],       # Minimum number of sample required to slit node
       "min_samples_leaf":(list(range(1,30)))  # Min number of sample required for leaf node.
       }
DT_cv=GridSearchCV(DT_reg,param,scoring="r2",verbose=1,cv=3)
# DT_reg= Model for training.
# param= hyperparameters (Dictonary created)
# scoring= performance matrix to performance checking.
# verbose= control the verbosity; the more message.
#>1=the computation time for each fold and parameter candidate is displayed;
#>2= the score is also displayed;
#>3= the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
#cv= number of flods

DT_cv.fit(X_train_scale,y_train)   # To training of gridsearch cv.
best_params=DT_cv.best_params_    # Give you best parameters.
print(f"Best parameters: {best_params}")


# In[199]:


print(bold+red+"Impute best parameter")
DT_reg=DecisionTreeRegressor(max_depth= 13, min_samples_leaf=10, min_samples_split= 2, splitter='best')


# In[200]:


# Model fit to training data
DT_reg.fit(X_train,y_train)
# Model predict with training data
DT_y_train_pred_hyp=DT_reg.predict(X_train)
# Model predict with testing data
DT_y_test_pred_hyp=DT_reg.predict(X_test)


# ## Evalution of Decision tree Regressor Hyperparameter tuning

# In[201]:


Training_DT_hyp=r2_score(y_train,DT_y_train_pred_hyp)*100
print(bold+green+"Trainind data R2 score: {:.2f}%".format(Training_DT))
Testing_DT_hyp=r2_score(y_test,DT_y_test_pred_hyp)*100
print(bold+green+"Testing data R2 score: {:.2f}%".format(Testing_DT))
DT2=r2_score(y_test,DT_y_test_pred_hyp)


# In[202]:


DT_MAE_hyp=mean_absolute_error(y_test,DT_y_test_pred_hyp)
print(bold+purple+"Testing data MAE score:",DT_MAE_hyp)
DT_MSE_hyp=mean_absolute_error(y_test,DT_y_test_pred_hyp)
print(bold+red+"Testing data MSE score:",DT_MSE_hyp)
DT_RMSE_hyp=np.sqrt(DT_MSE_hyp)
print(bold+green+"Testing data RMSE score:",DT_RMSE_hyp)
adjusted_R2_score_DT_hyp=(1-(((1-DT2)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_DT_hyp)


# # AdaBoost

# In[203]:


# Initialise model
ada_model= AdaBoostRegressor()
# fit the model with dataset
ada_model.fit(X_train,y_train)
# Predict model with training data
y_train_ada=ada_model.predict(X_train)
# Predict model with testing data
y_test_ada=ada_model.predict(X_test)


# # Evalution of AdaBoost 

# In[204]:


Training_ada=r2_score(y_train,y_train_ada)*100
print(bold+purple+"R2 score for training dataset: {:.2f}%".format(Training_ada))
Testing_ada=r2_score(y_test,y_test_ada)*100
print(bold+red+"R2 score for testing dataset: {:.2f}%".format(Testing_ada))
ADA1=r2_score(y_test,y_test_ada)


# In[205]:


MAE_ada=mean_absolute_error(y_test,y_test_ada)
print(bold+red+"Testing data MAE score:",MAE_ada)
MSE_ada=mean_squared_error(y_test,y_test_ada)
print(bold+purple+"Testing data MSE score:",MSE_ada)
RMSE_ada=np.sqrt(MSE_ada)
print(bold+green+"Testing data RMSE score:",RMSE_ada)
adjusted_R2_score_ada=1-(((1-ADA1)*(2093-1))/(2093-12-1))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_ada)


# ## Hyperparameter Tuning of ADABOOST

# In[222]:


Base_model=DecisionTreeRegressor(max_depth=15)
# Hyperparameter tuning of model
Ada_hyp=AdaBoostRegressor(base_estimator=Base_model,n_estimators=100,random_state=60)
# fit model with train dataset
Ada_hyp.fit(X_train_scale,y_train)


# In[223]:


# Predict model
y_train_ada_hyp=Ada_hyp.predict(X_train_scale)
y_test_ada_hyp=Ada_hyp.predict(X_test_scale)


# ## Evalution of ADABOOST

# In[224]:


ada_r2_train=r2_score(y_train,y_train_ada_hyp)*100
print(bold+purple+"Train data adaboost r2 score: {:.2f}%".format(ada_r2_train))
ada_r2_test=r2_score(y_test,y_test_ada_hyp)*100
print(bold+red+"Test data adaboost r2 score: {:.2f}%".format(ada_r2_test))


# In[225]:


MAE_ada_hyp=mean_absolute_error(y_test,y_test_ada_hyp)
print(bold+red+"Testing data MAE score:",MAE_ada_hyp)
MSE_ada_hyp=mean_squared_error(y_test,y_test_ada_hyp)
print(bold+purple+"Testing data MSE score:",MSE_ada_hyp)
RMSE_ada_hyp=np.sqrt(MSE_ada)
print(bold+green+"Testing data RMSE score:",RMSE_ada_hyp)
adjusted_R2_score_ada_hyp=1-(((1-ADA1)*(2093-1))/(2093-12-1))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_ada_hyp)


# # GradientBoosting

# In[135]:


# Intialise model
Grd_model=GradientBoostingRegressor()
# Fit data to model
Grd_model.fit(X_train_scale,y_train)
# predict training data
y_pred_grd=Grd_model.predict(X_train_scale)
y_test_grd=Grd_model.predict(X_test_scale)


# ## Evalution of GradientBoosting 

# In[136]:


Training_grd=r2_score(y_train,y_pred_grd)*100
print(bold+cyan+'R2 score of training data: {:.2f}%'.format(Training_grd))
Testing_grd=r2_score(y_test,y_test_grd)*100
print(bold+green+'R2 score of testing data: {:.2f}%'.format(Testing_grd))
GD1=r2_score(y_test,y_test_grd)


# In[137]:


MAE_grd=mean_absolute_error(y_test,y_test_grd)
print(bold+red+"Testing data MAE score:",MAE_grd)
MSE_grd=mean_squared_error(y_test,y_test_grd)
print(bold+purple+"Testing data MSE score:",MSE_grd)
RMSE_grd=np.sqrt(MSE_grd)
print(bold+green+"Testing data RMSE score:",RMSE_grd)
adjusted_R2_score_grd=1-(((1-GD1)*(2093-1))/(2093-12-1))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_grd)


# ## Hyperparameter Tuning of GradientBoosting

# In[140]:


grd_hyp=GradientBoostingRegressor(random_state=40)
# Hyperparameter for gridsearch cv
param={'learning_rate':[0.01,0.1,0.5,1],
       'n_estimators':[50,100,150],
       'min_samples_split':[2,3,4],
       'min_samples_leaf':[1,2,3,4],
       'max_depth':[3,4,5,6],
       
       }
grd_param_hyp=GridSearchCV(estimator=grd_hyp,param_grid=param,verbose=1,scoring='r2',cv=2)
# fit data to model
grd_param_hyp.fit(X_train_scale,y_train)

# Best parameter
best_param_hyp=grd_param_hyp.best_params_
print(f"Best parameters: {best_param_hyp}")
#print(f"Best score R2:" grd_param_hyp.best_score_ )


# In[226]:


print(bold+red+"Impute best parameter")
GB_reg_param=GradientBoostingRegressor(learning_rate=0.1,max_depth= 6, min_samples_leaf=1, min_samples_split= 4, n_estimators=150)


# In[227]:


GB_reg_param.fit(X_train_scale,y_train)
# Training data predict
y_train_grd_hyp=GB_reg_param.predict(X_train_scale)
# Testing data predict
y_test_grd_hyp=GB_reg_param.predict(X_test_scale)


# ## Evalution of GradientBoosting 

# In[228]:


Training_grd_hyp=r2_score(y_train,y_train_grd_hyp)*100
print(bold+cyan+'R2 score of training data: {:.2f}%'.format(Training_grd_hyp))
Testing_grd_hyp=r2_score(y_test,y_test_grd_hyp)*100
print(bold+green+'R2 score of testing data: {:.2f}%'.format(Testing_grd_hyp))
GD2=r2_score(y_test,y_test_grd_hyp)


# In[229]:


MAE_grd_hyp=mean_absolute_error(y_test,y_test_grd_hyp)
print(bold+red+"Testing data MAE score:",MAE_grd_hyp)
MSE_grd_hyp=mean_squared_error(y_test,y_test_grd_hyp)
print(bold+purple+"Testing data MSE score: ",MSE_grd_hyp)
RMSE_grd_hyp=np.sqrt(MSE_grd)
print(bold+green+"Testing data RMSE score:",RMSE_grd_hyp)
adjusted_R2_score_grd_hyp=1-(((1-GD2)*(2093-1))/(2093-12-1))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_grd_hyp)


# # XGBOOST

# In[230]:


# initialize of model
xgb=XGBRegressor()
# Fit dataset to model
xgb.fit(X_train_scale,y_train)
# predict training data
y_train_xgb=xgb.predict(X_train_scale)
y_test_xgb=xgb.predict(X_test_scale)


# ## Evalution of XGBoost 

# In[231]:


Training_xgb=r2_score(y_train,y_train_xgb)*100
print(bold+cyan+'R2 score of training data: {:.2f}%'.format(Training_xgb))
Testing_xgb=r2_score(y_test,y_test_xgb)*100
print(bold+green+'R2 score of testing data: {:.2f}%'.format(Testing_xgb))
XGB1=r2_score(y_test,y_test_xgb)


# In[232]:


MAE_xgb=mean_absolute_error(y_test,y_test_xgb)
print(bold+red+"Testing data MAE score:",MAE_xgb)
MSE_xgb=mean_squared_error(y_test,y_test_xgb)*100
print(bold+purple+"Testing data MSE score:",MSE_xgb)
RMSE_xgb=np.sqrt(MSE_xgb)
print(bold+green+"Testing data RMSE score:",RMSE_xgb)
adjusted_R2_score_xgb=1-(((1-XGB1)*(2093-1))/(2093-12-1))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_xgb)


# ## Hyperparameter tuning of XGBoost

# In[233]:


xgb_hyp=XGBRegressor(learning_rate=1,verbosity=2,max_depth=7)
best_param_xgb={#'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,0.9],
           #'max_depth': [2,3,4,5,7,9,10,12,14,15],
           'max_leaves':[2,3,4,5,6,7,8,9,10],
           #'verbosity':[0,1,2,3],
           'gamma':[0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150],
           'n_estimators':[10,20,30,50,65,80,100,115,130,150,200],
           'reg_alpha': [0,0.1,0.2,0.3,0.9,1.6,3.9],
           'reg_lambda': [0,0.1,0.2,0.3,0.9,1.6,3.9]
            }
XGB_rcv=RandomizedSearchCV(estimator=xgb_hyp,scoring='r2',param_distributions=best_param_xgb,
                   n_iter=100,n_jobs=-1,cv=3,random_state=50,verbose=3)
# Training data of randomizedsearchcv
XGB_rcv.fit(X_train_scale,y_train)

# Best parameters
XGB_best_param=XGB_rcv.best_params_
print(f"best parameters:{XGB_best_param}")


# In[237]:


print(bold+green+"Impute best parameter")
XGB_best_param=XGBRegressor(reg_lambda=3.9,reg_alpha=3.9,n_estimators=200,max_leaves=7,gamma=0.9,max_depth=5,verbosity=2)


# In[238]:


# model fit with training data
XGB_best_param.fit(X_train_scale,y_train)
# predict train data
y_train_XGB_pred=XGB_best_param.predict(X_train_scale)
# Predict test data
y_test_XGB_pred=XGB_best_param.predict(X_test_scale)


# ## Evalution hyperparameter XGBoost

# In[239]:


Training_xgb_hyp=r2_score(y_train,y_train_XGB_pred)*100
print(bold+cyan+'R2 score of training data: {:.2f}%'.format(Training_xgb_hyp))
Testing_xgb_hyp=r2_score(y_test,y_test_XGB_pred)*100
print(bold+green+'R2 score of testing data: {:.2f}%'.format(Testing_xgb_hyp))
XGB2=r2_score(y_test,y_test_XGB_pred)


# In[241]:


MAE_xgb_hyp=mean_absolute_error(y_test,y_test_XGB_pred)
print(bold+red+"Testing data MAE score:",MAE_xgb_hyp)
MSE_xgb_hyp=mean_squared_error(y_test,y_test_XGB_pred)
print(bold+purple+"Testing data MSE score:",MSE_xgb_hyp)
RMSE_xgb_hyp=np.sqrt(MSE_xgb)
print(bold+green+"Testing data RMSE score:",RMSE_xgb_hyp)
adjusted_R2_score_xgb_hyp=1-(((1-XGB2)*(2093-1))/(2093-12-1))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_xgb_hyp)


# # Support Vector Machine

# In[268]:


svmr=SVR(kernel='linear',gamma=0.1,C=70)
# Train the data
svmr.fit(X_train_scale,y_train)
# Predict train data
y_train_svmr=svmr.predict(X_train_scale)
# predict test data
y_test_svmr=svmr.predict(X_test_scale)


# ## Evalution of SVR model

# In[269]:


Training_svr=r2_score(y_train,y_train_svmr)*100
print(bold+red+"Training data R2 score: {:.2f}%".format(Training_svr))
Testing_svr=r2_score(y_test,y_test_svmr)*100
print(bold+red+"Training data R2 score: {:.2f}%".format(Testing_svr))
svr1=r2_score(y_test,y_test_svmr)


# In[267]:


SVR_MAE=mean_absolute_error(y_test,y_test_svmr)
print(bold+red+"Testing data MAE score:",SVR_MAE)
SVR_MSE=mean_squared_error(y_test,y_test_svmr)
print(bold+purple+"Testing data MSE score:",SVR_MSE)
SVR_RMSE=np.sqrt(SVR_MSE)
print(bold+green+"Testing data RMSE score:",SVR_RMSE)
adjusted_R2_score_SVR=(1-(((1-svr1)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_SVR)


# # KNN Regression

# In[278]:


Error_rate=[]
for i in range (1,12):
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_train_pred=knn.predict(X_train)
    y_test_pred=knn.predict(X_test)
    Error_rate.append(np.mean(y_test!=y_test_pred))
Error_rate


# In[281]:


plt.figure(figsize=(5,5))
plt.plot(range(1,12),Error_rate,color='green',marker='D',markersize=5)
plt.xlabel("K-value")
plt.ylabel("error_rate")
plt.show()


# In[303]:


KNN_model=KNeighborsRegressor(n_neighbors=4)
KNN_model.fit(X_train_scale,y_train)
y_train_pred=KNN_model.predict(X_train_scale)
y_test_pred=KNN_model.predict(X_test_scale)


# ## Evalution of KNN

# In[304]:


KNN_score_train=r2_score(y_train,y_train_pred)*100
print(bold+cyan+"Train data r2 score: {:.2f}%".format(KNN_score_train))
KNN_score_test=r2_score(y_test,y_test_pred)*100
print(bold+green+"Test data r2 score: {:.2f}%".format(KNN_score_test))
KNN_=r2_score(y_test,y_test_pred)


# In[305]:


KNN_MAE=mean_absolute_error(y_test,y_test_pred)
print(bold+red+"Testing data MAE score:",KNN_MAE)
KNN_MSE=mean_squared_error(y_test,y_test_pred)
print(bold+purple+"Testing data MSE score:",KNN_MSE)
KNN_RMSE=np.sqrt(SVR_MSE)
print(bold+green+"Testing data RMSE score:",KNN_RMSE)
adjusted_R2_score_KNN=(1-(((1-svr1)*(2093-1))/(2093-12-1)))
print(bold+cyan+"Testing data adjusted R2 score:",adjusted_R2_score_KNN)


# # Conclusion

# In[ ]:




