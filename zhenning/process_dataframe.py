import pandas as pd
import numpy as np

def get_code1(df,label, inplace=True):
    df[label].dropna(inplace=True)
    df[label] = df[label].astype('int32')
    df.replace({label: {5:6, 16:11, 14:15, 17:18}}, inplace=inplace)
    df = df[df[label] <= 19]
    return df
    
def get_code2(df,label, inplace=True):
    df[label].dropna(inplace=True)
    df[label] = df[label].astype('int32')
    df.replace({label: {27:26, 28:26}}, inplace=inplace)
    df = df[df[label] > 19]
    return df

# for each cat in labelp, change all of them to it corresponds labels 
def flip_cat(df, labelp, labels, cat=18):
    temp = df[(df[labelp]==cat) & (df[labels]!='-')]
    df = df[((df[labelp]==cat)&(df[labels]=='-')) | (df[labelp]!=cat)]
    l = temp[labels]
    temp[labelp] = temp[labels]
    temp[labels] = l
    df = df.append(temp)
    return df
    
def remove_from_df(df, label, l):
    for val in l:
        df = df[df[label]!=val]
    return df
        
def lower_case(df, label):
    df[label] = df[label].str.lower()
    return df

def upper_case(df, label):
    df[label] = df[label].str.upper()
    return df

def remove_dup(df, label=None):
    if label == None:
        df["is_duplicate"]= df.duplicated()
        df = df[df["is_duplicate"]==False]
    else:
        df["is_duplicate"]= df[label].duplicated()
        df = df[df["is_duplicate"]==False]
    return df