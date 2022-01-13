#!/usr/bin/env python
# coding: utf-8

# ### Generating 6 months stock price from NSE

# In[1]:


## pip install nsepy


# In[1]:


from nsepy import get_history
from datetime import date


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


DRL = get_history(symbol="DRREDDY", start=date(2021,7,12), end=date(2022,1,10))
DRL


# ### Merging Stock prices with Polarity

# In[4]:


DRL_revised = DRL.drop(["Symbol","Series","Prev Close","Last","Turnover","Trades","Deliverable Volume","%Deliverble"], axis = 1)


# In[5]:


DRL_revised


# In[6]:


Polarity = pd.read_excel("D:/ISB/Term 2/Foundational Project-1/Group Assignment/FinalOutput.xlsx")


# In[7]:


Polarity


# In[8]:


Polarity.set_index(["Date"],inplace=True)


# In[9]:


Polarity


# In[10]:


merged_data = pd.merge(DRL_revised,Polarity,left_index=True,right_index=True)


# In[11]:


merged_data


# ### Predicting trend with Random Forest Classifier Model

# In[12]:


# Dataframe with VWAP,  Volume, Polarity, Twitter_volume
DRL_df = merged_data[["VWAP", "Volume", "Polarity", "News_Tweet_volume"]]
DRL_df.head()


# In[13]:


# Sorting Polarity into Positive, Negative and Neutral sentiment

sentiment = [] 
for score in DRL_df["Polarity"]:
    if score >= 0.05 :
          sentiment.append("Positive") 
    elif score <= - 0.05 : 
          sentiment.append("Negative")        
    else : 
        sentiment.append("Neutral")   

DRL_df["Sentiment"] = sentiment
DRL_df.head()


# In[14]:


# Sentiment Count
DRL_df["Sentiment"].value_counts()


# In[15]:


#Stock Trend based on difference between current price to previous day price and coverting them to '0' as fall and '1' as rise in stock price
DRL_df["Price Diff"] = DRL_df["VWAP"].diff()
DRL_df.dropna(inplace = True)
DRL_df["Trend"] = np.where(DRL_df['Price Diff'] > 0 , 1, 0)

DRL_df.head()


# In[16]:


# Binary encoding Sentiment column
DRL_trend = DRL_df[["VWAP", "Volume", 'News_Tweet_volume', "Sentiment", "Trend"]]
DRL_trend = pd.get_dummies(DRL_trend, columns=["Sentiment"])
DRL_trend.head()


# In[17]:


# Defining features set
X = DRL_trend.copy()
X.drop("Trend", axis=1, inplace=True)
X.head()


# In[18]:


# Defining target vector
y = DRL_trend["Trend"].values.reshape(-1, 1)
y[:5]


# In[19]:


# Splitting into Train and Test sets
split = int(0.7 * len(X))

X_train = X[: split]
X_test = X[split:]

y_train = y[: split]
y_test = y[split:]


# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# In[21]:


# Using StandardScaler to scale features data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[22]:


# Create RFClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=46)

# Fit the model
rf_model = rf_model.fit(X_train_scaled, y_train.ravel())  


# In[25]:


# Make predictions
predictions = rf_model.predict(X_test_scaled)
pd.DataFrame({"Prediction": predictions, "Actual": y_test.ravel()}).head(20)

# Generate accuracy score for predictions using y_test
acc_score = accuracy_score(y_test, predictions)
print(f"Accuracy Score : {acc_score}")


# In[26]:


# Model Evaluation

# Generating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

# Displaying results
display(cm_df)


# In[27]:


# Generating classification report
print("Classification Report")
print(classification_report(y_test, predictions))


# ### Predicting trend using Naive Bayes Model

# In[28]:


from sklearn.naive_bayes import BernoulliNB


# In[29]:


classifier = BernoulliNB()
classifier.fit(X_train_scaled,y_train,)
y_pred = classifier.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))


# In[30]:


# Model Evaluation

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

# Displaying results
display(cm_df)


# In[31]:


# Generating classification report
print("Classification Report")
print(classification_report(y_test, y_pred))


# ### Random Forest Regressor Model - Stock Prediction

# In[32]:


from sklearn.ensemble import RandomForestRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


# In[33]:


# Dataframe with VWAP, Polarity, twitter_volume
DRL_df1 = merged_data[["VWAP", "Polarity", "News_Tweet_volume"]]
DRL_df1.head()


# In[34]:


# pct change based on Adj close value
DRL_df1["Pct_change"] = DRL_df1["VWAP"].pct_change()

# Drop null values
DRL_df1.dropna(inplace = True)
DRL_df1.head()


# In[35]:


# This function "window_data" accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(DRL_df1, window, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number):
    # Create empty lists "X_close", "X_polarity", "X_volume" and y
    X_close = []
    X_polarity = []
    X_volume = []
    y = []
    for i in range(len(DRL_df1) - window):
        
        # Get close, ts_polarity, tw_vol, and target in the loop
        close = DRL_df1.iloc[i:(i + window), feature_col_number1]
        ts_polarity = DRL_df1.iloc[i:(i + window), feature_col_number2]
        tw_vol = DRL_df1.iloc[i:(i + window), feature_col_number3]
        target = DRL_df1.iloc[(i + window), target_col_number]
        
        # Append values in the lists
        X_close.append(close)
        X_polarity.append(ts_polarity)
        X_volume.append(tw_vol)
        y.append(target)
        
    return np.hstack((X_close,X_polarity,X_volume)), np.array(y).reshape(-1, 1)


# In[36]:


# Predict Closing Prices using a 3 day window of previous closing prices
window_size = 3

# Column index 0 is the `VWAP` column
# Column index 1 is the `Polarity` column
# Column index 2 is the `News_Tweet_volume` column
feature_col_number1 = 0
feature_col_number2 = 1
feature_col_number3 = 2
target_col_number = 0
X, y = window_data(DRL_df1, window_size, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number)


# In[37]:


# Use 70% of the data for training and the remainder for testing
X_split = int(0.7 * len(X))
y_split = int(0.7 * len(y))

X_train = X[: X_split]
X_test = X[X_split:]
y_train = y[: y_split]
y_test = y[y_split:]


# In[38]:


# Use the MinMaxScaler to scale data between 0 and 1.
x_train_scaler = MinMaxScaler()
x_test_scaler = MinMaxScaler()
y_train_scaler = MinMaxScaler()
y_test_scaler = MinMaxScaler()

# Fit the scaler for the Training Data
x_train_scaler.fit(X_train)
y_train_scaler.fit(y_train)

# Scale the training data
X_train = x_train_scaler.transform(X_train)
y_train = y_train_scaler.transform(y_train)

# Fit the scaler for the Testing Data
x_test_scaler.fit(X_test)
y_test_scaler.fit(y_test)

# Scale the y_test data
X_test = x_test_scaler.transform(X_test)
y_test = y_test_scaler.transform(y_test)


# In[39]:


# Create the Random Forest regressor instance
model = RandomForestRegressor(n_estimators=1000, max_depth=2, bootstrap=False, min_samples_leaf=1)


# In[40]:


# Fit the model
model.fit(X_train, y_train.ravel())


# In[41]:


# Model Performance
# Make some predictions
predicted = model.predict(X_test)


# In[43]:


predicted


# In[44]:


# Evaluating the model
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R-squared :', metrics.r2_score(y_test, predicted))


# In[45]:


# Recover the original prices instead of the scaled version
predicted_prices = y_test_scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))


# In[46]:


# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({"Real": real_prices.ravel(),"Predicted": predicted_prices.ravel()}, 
                      index = DRL_df1.index[-len(real_prices): ]) 
stocks.head()


# In[47]:


stocks.plot(title = "Real vs Predicted values of DRL")


# ### XG Boost Regressor Model - Stock Prediction

# In[51]:


get_ipython().system('pip install xgboost')


# In[48]:


from xgboost import XGBRegressor


# In[49]:


# Create the XG Boost regressor instance
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)


# In[50]:


# Fit the model
model.fit(X_train, y_train.ravel())


# In[51]:


# Model Performance

# Make some predictions
predicted = model.predict(X_test)


# In[52]:


predicted


# In[53]:


# Evaluating the model
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R-squared :', metrics.r2_score(y_test, predicted))


# In[54]:


# Recover the original prices instead of the scaled version
predicted_prices = y_test_scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))


# In[55]:


# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({"Real": real_prices.ravel(),"Predicted": predicted_prices.ravel()}, 
                      index = DRL_df1.index[-len(real_prices): ]) 
stocks.head()


# In[56]:


# Plot the real vs predicted values as a line chart
stocks.plot(title = "Real vs Predicted values of DRL")

