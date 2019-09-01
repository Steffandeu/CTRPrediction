import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('aquastat_new.csv',index_col=False)
# print(data.shape)
print(data.info())
print("###########################################")
print(data.head())
print(data[["variable_id","variable_name"]].drop_duplicates())
print("###########################################")
# print(data.Area.nunique())
time_period = data.Year.nunique()
print(time_period)
print(data["Year"])
# data[data["Var"] = "total_"]
print("*********************************")
print(data[data.variable_name=="Gross Domestic Product (GDP)"].Value.isnull().sum())


def country_slice(df,country):
    df = df[df.Area==country]
    #Pivot table
    df = df.pivot(index='variable_name',columns="Year",values ="Value")
    df.index.name = country
    return df

def time_slice(df,time_period):
    df = df[df.Year==time_period]
    df = df.pivot(index='variable_name', columns="Year", values="Value")
    df.columns.name = time_period
    return df

#国家和变量的二维时间序列变化
def time_series(df,country,variable):
    series = df[(df.Area==country) & (df.variable_name==variable)]
    #drop years with no data
    series = series.dropna()[["Year","Value"]]

    #change year to int and set as index
    series.Year = series.Year.astype(int)
    series.set_index("Year",inplace=True)
    series.columns = [variable]
    return series



# print(country_slice(data,"Armenia"))
print("**********************************")
print(time_series(data,"Armenia","Gross Domestic Product (GDP)"))

import pivottablejs
import missingno as msno
import pandas_profiling


recent = time_slice(data,"2016")
# msno.matrix(recent,labels=True)
msno.bar(data)

# collisions = missingno_data.nyc_collision_factors()
collisions = data.nyc_collision_factors()
import folium

pandas_profiling.ProfileReport(data)
