#!/usr/bin/env python
# coding: utf-8

# # Keras Regression Code Along Project
# 
# Let's now apply our knowledge to a more realistic data set. Here we will also focus on feature engineering and cleaning our data!

# ## The Data
# 
# We will be using data from a Kaggle data set:
# 
# https://www.kaggle.com/harlfoxem/housesalesprediction
# 
# #### Feature Columns
#     
# * id - Unique ID for each home sold
# * date - Date of the home sale
# * price - Price of each home sold
# * bedrooms - Number of bedrooms
# * bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# * sqft_living - Square footage of the apartments interior living space
# * sqft_lot - Square footage of the land space
# * floors - Number of floors
# * waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# * view - An index from 0 to 4 of how good the view of the property was
# * condition - An index from 1 to 5 on the condition of the apartment,
# * grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# * sqft_above - The square footage of the interior housing space that is above ground level
# * sqft_basement - The square footage of the interior housing space that is below ground level
# * yr_built - The year the house was initially built
# * yr_renovated - The year of the house’s last renovation
# * zipcode - What zipcode area the house is in
# * lat - Lattitude
# * long - Longitude
# * sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# * sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors
# 
# #### The problem
# * Develop a neural network model to predict the house price based on the above features.

# In[3]:


#
# Your code to import libraries, numpy, pandas, matplotlib,  seaborn
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Your code to read the data from the file provided.
df = pd.read_csv("kc_house_data.csv")


# # Exploratory Data Analysis
# 
# Perform some data analysis using the libraries above.
# Visualise the features to understand the problem and use the appropriate features for the model.

# In[5]:


# Your code to check if any null value is present in the dataset. Hint - use isnull() in pandas
pd.isnull(df)


# In[6]:


# Your code to describe the dataset to get imortant properties of it.
df.describe()


# # Let's see how price columns look like i.e. how prices are distributed.

# In[7]:


plt.figure(figsize=(12,8))
sns.distplot(df['price'])


# #### Let's look at the columns - number of bedroom in more detail
# #### Write code to plot the numbers of bedrooms and the number of times they appear in the data.  
# ####The x-axis contains the number of bedrroms and the y axis will portray the number of times the particular bedroom appears in the column.
# 
# #### Hint - use sns.countplot.  

# In[8]:


# Your code to plot number of bedrooms and their counts
sns.countplot(x='bedrooms', data=df)
plt.title('Number of Bedrooms and Their Counts')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.show()


# # Plot a scatter plot between the price and sqft_living column.

# In[9]:


# You code to visualize an scatter plot.
plt.scatter(df['sqft_living'], df['price'])
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.title('Price and Sqft Living Scatter Plot')
plt.show()


# In[10]:


sns.boxplot(x='bedrooms',y='price',data=df)


# ### Geographical Properties
# 
# Visulaize an scatter plot between price and longitude (long) and latidtude (lat)  columns.
# 
# 

# In[11]:


# Your code to visualize an scatter plot between price and longitude (long) column.
plt.scatter(df['long'], df['price'])
plt.xlabel('long')
plt.ylabel('Price')
plt.title('Scatter plot between price and longitude ')
plt.show()


# In[12]:


# Your code to visualize an scatter plot between price and latitude(lat) column.
plt.scatter(df['price'], df['lat'])
plt.xlabel('lat')
plt.ylabel('Price')
plt.title('Scatter plot between price and latitude ')
plt.show()


# In[13]:


plt.figure(figsize=(12,8))
sns.scatterplot(x = 'long', y = 'lat', data = df, hue = 'price')


# # Sort the values in the dataframe according to price and print first few rows.

# In[14]:


# Your code to sort data frame according to the price (ascending order) and see first few rows. 
# Hint - Use df.sort_values and combine it with head()
sorted_df = df.sort_values(by='price', ascending=True).head()
print(sorted_df.head())


# #### The following code visualizes the price intensity with the latitude and longitude for 1% and 99% of the data separately.
# 
# #### You need to add comment on each line of the code.

# In[15]:


len(df)*(0.01)


# In[16]:


non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
non_top_1_perc


# In[17]:


plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat', 
                data=non_top_1_perc,hue='price', 
                palette='RdYlGn',edgecolor=None,alpha=0.2)


# In[ ]:





# ### Other Features
# # Let's have a box plot between waterfront and price.
# 
# # Explain what box plot is doing?

# In[18]:


sns.boxplot(x='waterfront',y='price',data=df)


# ## Working with Feature Data

# In[19]:


# Your code to print first few rows of the data.
df.head()


# In[20]:


# Your code to print info about the data
print(df.info())


# Following code is dropping the column ID.
# 
# Question - why are dropping this column?

# In[21]:


df.drop(columns=['zipcode'])


# In[22]:


df = df.drop('id',axis=1)


# In[23]:


df.head()


# ### Feature Engineering from Date
# 
# Transform the features into useful formats to apply appropriate Deep NN technique!

# In[24]:


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
#df['year'] = df['date'].apply(lambda date:date.year)

# Your code to check what above code is doing.


# In[25]:


sns.boxplot(x='year',y='price',data=df)


# In[26]:


# Your code to visualize boxplot between month and price
sns.boxplot(x='month',y='price',data=df)


# In[27]:


# we do not need the 'date' column anymore

df = df.drop('date',axis=1)


# In[28]:


df.columns


# In[29]:


df['zipcode'].value_counts()


# Should we remove zipcode?  If so, remove it.

# In[30]:


# Your code to remove zip code - Hint - Use df.drop
# Your code to remove 'date' column.
df = df.drop('zipcode', axis=1)


# In[31]:


df.head()


# In[32]:


# could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()


# In[33]:


df['sqft_basement'].value_counts()


# ## Scaling and Train Test Split
# 
# Scikit-Learn is used to split out the train-test library.

# First separate input and output. Input will be stored in the variable X and output in variable y.

# In[34]:


# Your code to store all columns except price column in variable X. Hint - Use pd.drop()
X = df.drop('price', axis=1)
# Your code to store output (price column) in variable y
y = df['price']


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[37]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Scaling
# 
# Features are scaled to be in a proper range to be useful for modeling.
# Scaling converts all values between 0-1.

# In[38]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data, such as the mean and variance of the features of the training set. These learned and fixed parameters are then used to scale our test data with the transform() function.

# In[39]:


# Your code to print shapes of X_train, X_test, y_train, y_test and see if shapes are okay.
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("y_train shape", y_train.shape)
print("y_test shape", X_test.shape)


# ## Creating a Model
# 
# Build a Deep NN model with appropriate layers using Keras.  

# In[40]:


#
# Import Libraries for Neural Network model development.
#

import tensorflow as tf
from tensorflow import keras



# In[41]:


input_shape=(X_train.shape[1],)
input_shape


# Develop your own Neural Network model with suitable number of input, output, and any number of hidden layers. Since we are predicting a value, the number of neurons in outpit layer should be one.

# In[44]:


#
# Your code to build MLP neural network model.
#
model = keras.Sequential([
    keras.layers.Dense(152, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(76, activation='relu'),
    keras.layers.Dense(38, activation='relu'),
    keras.layers.Dense(19, activation='relu'),
    keras.layers.Dense(1)
])


# In[45]:


print(model.summary())


# ## Training the Model
# 
# Write the code to train the neural network model. Use the following of your choice:
# 
# 
# 
# 1.   Optimization method
# 2.   Batch size
# 3.   Number of epochs.
# 
# Test for various optimizers and check which one performs better in terms of loss function = 'mse'.
# 
# Use following APIs
# * https://keras.io/api/optimizers/
# * https://keras.io/api/losses/regression_losses/
# * https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
# * https://keras.io/api/models/model_training_apis/
# 
# 

# In[47]:


# ! add your chosen optimiser !

model.compile(optimizer='adam', loss='mse')


# In[48]:


#
# Your code to train the model.
#
history = model.fit(X_train, y_train, batch_size=128, epochs=400,  
                    verbose=1)


# # Following code gets the history of losses at every epoch.

# In[49]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# In[50]:


history_dict = history.history
Loss = losses
plt.figure(num=1, figsize=(15,7))
plt.plot(Loss, 'bo', label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Evaluation on Test Data
# 
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
# 
# Scikit-Learn has metrics to evaluate the performance.

# In[51]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


# #### Next, we will test the performance of our model by predicting on test dataset X_test.

# In[52]:


# Your code to print X_test and see how test data looks like.
print(X_test)


# In[53]:


X_test.shape


# # Following you will predict the output based on the input data X_test.
# 
# Lonk to API - https://keras.io/api/models/model_training_apis/#predict-method

# In[54]:


predictions = model.predict(X_test)


# ## Following code will test the error in the predicted values. 
# ## Error is the difference between the predictions you made and real values (y_test)

# In[57]:


print(mean_absolute_error(y_test,predictions))


# In[58]:


print(np.sqrt(mean_squared_error(y_test,predictions)))


# The following code plots the predicted values in a scatter plot. We have also plotted the perfect predictions.

# In[59]:


explained_variance_score(y_test,predictions)


# In[60]:


df['price'].mean()


# In[61]:


df['price'].median()


# In[62]:


# Our predictions
plt.scatter(y_test, predictions)

# Perfect predictions
plt.plot(y_test, y_test, 'r')


# In the following code, we have plotted the error i.e. the difference between the actual and predicted values.

# In[66]:


errors = y_test.values.reshape(4323, 1) - predictions


# In[67]:


sns.distplot(errors)


# 
# ### Following code makes prediction on a brand new house. Comment each line of the code.
# 
# Try predicting price for a new home.

# In[68]:


single_house = df.drop('price',axis=1).iloc[0]


# In[69]:


single_house = scaler.transform(single_house.values.reshape(-1, 19))


# In[70]:


single_house


# In[71]:


model.predict(single_house)


# In[72]:


df.iloc[0]


# In[73]:


# The prediction result:

err = int(model.predict(single_house)) - df['price'].iloc[0]

print( 'absolute prediction error = ', err, ' $')
print( 'relative prediction error = ', err / df['price'].iloc[0] *100 , ' %')


# # Lab Logbook requirement:
# 
# # Please record the plot of the  model's  loss after every epoch and a summary in lab logbook. You can obtain the model summary using the model.summary() method. The API for obtaining the model summary is defined in the following link:
# 
# 
# # https://keras.io/api/models/model/#summary-method
# 
# 
# # Ensure that no code or other information is added to the logbook and that only required information is present.  Marks will not be awarded if anything else is found in the logbook or instructions are not clearly followed.
# 
# 
# 
# 
# 

# In[ ]:





# 
# ## Try different optimisations like changing model architecture, activation functions, training parameters.

# In[ ]:




