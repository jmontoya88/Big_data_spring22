#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('US_Accidents_Dec21_updated.csv')

print(df.head(10))

# Display the variable list
print(df.columns.values)

# Display the number of rows and the number of columns in the data set to confirm the portrait shape
# The first element of the output is the number of rows and the second is the number of columns 
print(df.shape)


# In[6]:


# Show the number of missing values for each variable in the data frame
df.isnull().sum()

rvar_list =['Zipcode','City','Precipitation(in)','Wind_Chill(F)','Wind_Direction', 'Start_Lat','Number', 'Start_Lng', 'End_Lat','End_Lng','Country' ,'Airport_Code','Civil_Twilight','Timezone','Nautical_Twilight','Astronomical_Twilight']
df_sample1 = df.drop(columns=rvar_list)
#City and zipcode removed as we have all county and variables
# Separate all the variables into two lists for future column indexing
# One for numerical, the other for categorical 
cvar_list = [ 'Street','Side',
 'County', 'State','Weather_Timestamp', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset','Severity']
nvar_list = ['Distance(mi)','Temperature(F)', 'Humidity(%)',
 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']

# Check if there is any missing value left
df_sample1.isnull().sum()


# In[27]:


df_sample1[['Distance(mi)','Temperature(F)', 'Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']] = df_sample1[['Distance(mi)','Temperature(F)', 'Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']].fillna(df_sample1[['Distance(mi)','Temperature(F)', 'Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']].mean())
df_sample1[['Street','Side',
 'County', 'State','Weather_Timestamp', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset','Severity']].fillna(df_sample1[['Street','Side',
 'County', 'State','Weather_Timestamp', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset','Severity']].mode())


# In[31]:


import numpy as np
np.zeros((2845342, 36, 2845342), dtype='uint8')


# In[29]:


# Standardize the numerical variables 
df_sample2 = df_sample1.copy()
df_sample2[nvar_list] = (df_sample1[nvar_list] - df_sample1[nvar_list].mean())/df_sample1[nvar_list].std()

# Set the datatype for the variables in the cvar_list to be categorical in Python
# Set the datatype for the variables in the nvar_list to be numerical in Python 
df_sample3 = df_sample2.copy()
df_sample3[cvar_list] = df_sample2[cvar_list].astype('category')
df_sample3[nvar_list] = df_sample2[nvar_list].astype('float64')

import numpy as np
>>> a = np.zeros((156816, 36, 53806), dtype='uint8')
>>> a.nbytes

# Convert the categorical variables into dummies (Step 1 of dummy coding)
# prefix_sep is the sympol used to create the dummy variable names.

df_sample4 = df_sample3.copy()
df_sample4 = pd.get_dummies(df_sample3, prefix_sep='_')

# Remove the redundant dummies (Step 2 of dummy coding)
# Placeholder variable: rdummies
#rdummies = []
#df_sample5 = df_sample4.copy()
#df_sample5 = df_sample4.drop(columns=rdummies)

# Get the remaining variable list after the variable transformation
#print(df_sample5.columns.values)

# Display the milestone dataframe. Compare it with the original dataframe.
#print(df_sample5)
#print(df)


# In[ ]:




