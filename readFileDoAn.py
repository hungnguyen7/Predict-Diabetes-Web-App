import pandas as  pd
import numpy as np

df= pd.read_csv("./KKDL/diabetes_data_upload.csv")
# df.head()
print(df.dtypes)
# 520 dong và 17 cột
print(df.shape)
#kiemtra null
print(df.isnull().sum())
#--Data không null!!
