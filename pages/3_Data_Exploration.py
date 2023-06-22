# to handle datasets
import numpy as np
import pandas as pd
import datetime

# for plotting
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import streamlit as st
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

# for statistics
import scipy
import scipy.stats as stats

# to split data set 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

# feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# dimensionality reductiion
from sklearn.manifold import TSNE

# clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

# to build models
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# to evaluate models
from sklearn.metrics import mean_squared_error


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('AutoInsurance.csv')

if st.checkbox('Show original data'):
    st.write(df)


st.title("Data Exploration")


st.subheader("Summary Statistics")

# Change format Date
df_1 = df.copy()
df_1['Effective_Date'] = pd.to_datetime(df_1['Effective To Date'])

# End of Year (eoy) of the current Effective to Date
df_1['eoy_Effective_Date'] = pd.to_datetime('2011-12-31')

# Days missing to the end of the current year from effective to date
df_1['eoy_missing_days'] = (df_1['eoy_Effective_Date'] - df_1['Effective_Date']).dt.days

df_1 = df_1.drop(['Customer','Effective To Date','eoy_Effective_Date'], axis=1)

# Set the effective to date column as the index of the dataframe
df_1 = df_1.set_index("Effective_Date")
df_1 = df_1.sort_index(ascending=True)

# Rename some features for a practical use
df_new = df_1.copy()
df_new = df_new.rename(columns={"Customer Lifetime Value":"Customer_Lifetime_Value","Location Code":"Location_Code","Marital Status":"Marital_Status","Monthly Premium Auto":"Monthly_Premium_Auto","Months Since Last Claim":"Months_Since_Last_Claim",
                           "Months Since Policy Inception":"Months_Since_Policy_Inception","Number of Open Complaints":"Number_of_Open_Complaints","Number of Policies":"Number_of_Policies",
                           "Policy Type":"Policy_Type","Renew Offer Type":"Renew_Offer_Type","Sales Channel":"Sales_Channel","Total Claim Amount":"Total_Claim_Amount",
                           "Vehicle Class":"Vehicle_Class","Vehicle Size":"Vehicle_Size"})


# Summarize attribute distributions for data type of variables
obj_cols = [var for var in df_new.columns if df_new[var].dtype=='object']
df_new[obj_cols].describe().T

# Summarize attribute distributions for data type of variables
no_obj_cols = [var for var in df_new.columns if df_new[var].dtype!='object']
df_new[no_obj_cols].describe().T


# Split data set between target variable and features
X_full = df_new.copy()
y = X_full.Customer_Lifetime_Value
X_full.drop(['Customer_Lifetime_Value'], axis=1, inplace=True)

st.subheader("Target Variable Analysis")

# Vis Chart
def plot_target(data, var):
    plt.rcParams['figure.figsize']=(10,5)
    fig=plt.figure()
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('CLV histogram')
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('CLV boxplot')
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('CLV Q-Q plot')
    plt.yticks(rotation=45, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)

    
    st.pyplot(fig)

plot_target(df_new, var='Customer_Lifetime_Value')


plt.rcParams['figure.figsize']=(10,5)
fig=plt.figure()
plot_acf(df_new['Customer_Lifetime_Value'], lags=np.arange(len(df_new)))
plt.title('Autocorrelation Function Plot on CLV', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.xticks(rotation=45, fontsize=15)
st.pyplot(fig)


st.subheader("Numerical Features Analysis")

# Select numerical columns
numerical_cols = [cname for cname in X_full.columns if 
                X_full[cname].dtype in ['float64','int64']]

# Subset with numerical features
num = X_full[numerical_cols]

# list of numerical features
fnum = ['Income', 'Monthly_Premium_Auto', 'Months_Since_Last_Claim',
       'Months_Since_Policy_Inception', 'Number_of_Open_Complaints',
       'Number_of_Policies', 'Total_Claim_Amount', 'eoy_missing_days']


# Vis Chart
def plot_num(data, var):
    plt.rcParams['figure.figsize']=(15,5)
    fig = plt.figure()
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    
    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.yticks(rotation=0, fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    
    st.pyplot(fig)

plot_num(num, var='Income')
plot_num(num, var='Monthly_Premium_Auto')
plot_num(num, var='Months_Since_Last_Claim')
plot_num(num, var='Months_Since_Policy_Inception')
plot_num(num, var='Number_of_Open_Complaints')
plot_num(num, var='Number_of_Policies')
plot_num(num, var='Total_Claim_Amount')
plot_num(num, var='eoy_missing_days')



# new subset for a bivariate analysis
num2 = pd.concat([y,num], axis=1)

# Correlation heatmap
corr_matrix = num2.corr()

sns.set( rc = {'figure.figsize': (10, 8)})
fig=plt.figure()
sns.heatmap(corr_matrix, square = True, annot=True, fmt='.2f')
plt.suptitle('Correlation Heatmap between numerical features and outcome',size=15)
plt.yticks(fontsize="13")
plt.xticks(fontsize="13")
st.pyplot(fig)


