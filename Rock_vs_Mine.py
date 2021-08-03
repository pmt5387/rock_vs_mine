import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import warnings

warnings.filterwarnings("ignore")

st.markdown("# Rock vs Mine Prediction")
url='https://www.kongsberg.com/contentassets/54e13215b6b74f0dae41a3f3d75cecb0/naval-sonar-asw-and-mine-hunting-1020x514?quality=50'
st.image(url)
df=pd.read_csv('https://raw.githubusercontent.com/pmt5387/rock_vs_mine/main/Sonar_data.csv',header=None)
df.head()
#df.shape
#df.info()
st.write(df.describe())
st.text("Labels data whether it is Rock or Mine")
st.write(df[60].value_counts())
df.groupby(60).mean()
st.markdown("*__Separating features and labels__*")
x=df.drop(columns=60,axis=1)
y=df[60]

x

y


# # 4) Training and Test Data

# # *test size=10% i.e 0.1*
# # *stratify---> almost equally divided into based on y dataframe i.e rock and mine*
# # *random state---> data will be splitted into same way for multiple users*

# # ----> Splitting the data

# In[22]:


x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)


# In[23]:


print(x.shape,x_train.shape,x_test.shape)


# # Model Training-----> Logistic regression

# In[24]:


model=LogisticRegression()


# ## 1) Training the Logistic Regression model with training data

# In[25]:


st.markdown("*'x' training data*")
x_train


# In[26]:

st.markdown("*'y' training data*")
y_train


# In[27]:


model.fit(x_train,y_train)


# # Check Accuracy and Model Evaluation

# # Accuracy on Training data

# In[28]:


x_train_prediction=model.predict(x_train)
training_accuracy=accuracy_score(x_train_prediction,y_train)


# In[29]:


#print("Accuracy on training data:",training_accuracy)
st.write("Accuracy on training data:",training_accuracy)

# # Accuracy on Test data

# In[30]:


x_test_prediction=model.predict(x_test)
training_accuracy_test=accuracy_score(x_test_prediction,y_test)


# In[31]:


#print("Accuracy on test data:",training_accuracy_test)
st.write("Accuracy on test data:",training_accuracy_test)


# # Making a predictive System
st.markdown("## Making a predictive system")
# In[45]:

try:
    input_data=st.text_input("Enter the features data: ")
    time.sleep(9)
    values=input_data.split(',')
    valuess=map(float,values)
    x=list(valuess)
    y=tuple(x)
except Exception as E:
    pass
#input_data=tuple(input_data)


# In[46]:

#input_data=(0.0335,0.0134,0.0696,0.1180,0.0348,0.1180,0.1948,0.1607,0.3036,0.4372,0.5533,0.5771,0.7022,0.7067,0.7367,0.7391,0.8622,0.9458,0.8782,0.7913,0.5760,0.3061,0.0563,0.0239,0.2554,0.4862,0.5027,0.4402,0.2847,0.1797,0.3560,0.3522,0.3321,0.3112,0.3638,0.0754,0.1834,0.1820,0.1815,0.1593,0.0576,0.0954,0.1086,0.0812,0.0784,0.0487,0.0439,0.0586,0.0370,0.0185,0.0302,0.0244,0.0232,0.0093,0.0159,0.0193,0.0032,0.0377,0.0126,0.0156)
# Changing the input data to numpy array
input_data_as_np=np.asarray(y)


# In[48]:


#reshape the np array as we are predicting for one instance
input_data_reshaped=input_data_as_np.reshape(1,-1)


# In[53]:


prediction=model.predict(input_data_reshaped)


# In[51]:


def result():
    if prediction[0]=='R':
        #st.write("The object is Rock")
        return ("The object is Rock")
    else:
        #st.write("The object is Mine")
        return ("The object is Mine")


# In[52]:


st.write(result())
#print(result())


