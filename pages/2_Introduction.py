import streamlit as st

st.title("Introduction")

#st.write("worldatlas")
#st.image("./images/earthquake-cause.png", width=500)


st.markdown("""

Measurement of customer lifetime value is the key economic variable conditioning the development and maintenance 
of long-term profitable relationships with customers. It also plays an important role in decisions concerning the acquisition of new customers and retention of current. 
Consequently, it affects the ability to continue the business of the company. Companies operating in the rapidly changing market are particularly susceptible 
to qualitatively accurate forecasts regarding the selection of the appropriate range of products, ways of implementation of purchasing processes, pricing policies, 
incentive schemes, etc. All these problems are directly or indirectly related eventually to customer because this is the customer that generates profits for the company 
necessary to run and continue the business and development. 
Customer lifetime value is a primary metric for understanding the customers. 
It is a prediction of the value about relationship with a customer can bring to the Company business.

""")

st.markdown("""
Data are coming from a Kaggle database.

[Source](https://www.kaggle.com/datasets/ranja7/vehicle-insurance-customer-data) 


### Data Description

#### Variables

**Customer**

The Customer ID 

**State**

The US Country of the policyholder

**Customer Lifetime Value**

The CLV amount. The reponse variable 

**Response**

Whether a customer responded to the marketing call or not (Yes for those who have responded and No for those who have not).

**Coverage**

Insurance coverages that the customers currently have

**Education**

Education level of policyholders

**Effective To Date**

**EmploymentStatus**

**Gender**

**Income**

**Location Code**

**Marital Status**

**Monthly Premium Auto**

**Months Since Last Claim**

**Months Since Policy Inception**

**Number of Open Complaints**

**Number of Policies**

**Policy Type**

**Policy**

**Renew Offer Type**

**Sales Channel**

**Total Claim Amount**

**Vehicle Class**

**Vehicle Size**

#### Evaluation Metric

The evaluation metric used for this data set is RMSE score

""")

