
import datetime
import pandas as pd
import statsmodels.api as sm
import numpy as np
#-----------------------------------------------------------
timeStart = datetime.datetime.now()
print("timeStart:" + str(timeStart))
#-----------------------------------------------------------
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
print (df.head())
df.columns = ["admit", "gre", "gpa", "prestige"]
print (df.columns)
print (df.describe())
print (df.std())
data = df
data['intercept'] = 1.0
train_cols = data.columns[1:]
logit = sm.Logit(data['admit'], data[train_cols])
result = logit.fit()
print (result.summary())
print (result.conf_int())
print (np.exp(result.params))
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print (np.exp(conf))
#-----------------------------------------------------------
timeFinish = datetime.datetime.now()
print("timeFinish:" + str(timeFinish))
print("timeRunning:" + str((timeFinish - timeStart).microseconds) + "microseconds")
#-----------------------------------------------------------