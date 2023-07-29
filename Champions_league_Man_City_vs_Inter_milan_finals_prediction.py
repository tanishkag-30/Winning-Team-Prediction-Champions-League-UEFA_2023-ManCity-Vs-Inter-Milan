#!/usr/bin/env python
# coding: utf-8

# In[212]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[213]:


# load the csv data
data = pd.read_excel('Downloads/football.xlsx')
data.head()


# In[214]:


# Define the features to use in the model
features = ['goals_scored', 'assists', 'shots_on_target', 'pass_completion_rate', 'tackles_won', 'saves_made', 'dribbles_completed', 'interceptions_made', 'formation', 'playing_style', 'set_piece_routines', 'head_to_head_record', 'recent_form', 'previous_champions_league_performance','Outcome']


# In[215]:


# Convert playing style into numerical representation using one-hot encoding
data_encoded = pd.get_dummies(features, columns=['playing_style'])


# In[216]:


# Extract the features from the data
features = data_encoded.drop(columns=['Outcome'])
labels = data_encoded['Outcome']


# In[217]:


# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[218]:


# Train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[219]:


# Predict the outcome for the testing set
y_pred = model.predict(X_test)


# In[220]:


# Assuming Man City is label 0 and Inter Milan is label 1
team_predictions = model.predict([[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]])


# In[221]:


if team_predictions[0] == 0:
    best_team = "Manchester City"
    victory_quote = "The sky-blue brigade reigns supreme in the Champions League Final!"
else:
    best_team = "Inter Milan"
    victory_quote = "The Nerazzurri conquer the Champions League Final with grace and power!"


# In[222]:


print("\nAnd the winner of the Champions League Final is...")
print(best_team + "!")
print(victory_quote)


# In[ ]:




