import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("name_of_dataset.csv") 

#to implement on smaller dataset (subsets of the main dataset), the following lines of codes can be used
n = 200

X_init = np.array(df.iloc[:, 0:-1]) #taking all the columns except for the final column
y_init = np.array(df.iloc[:, -1]).astype(int)
X_main_tr, X_main_ts, y_main_tr, y_main_ts = train_test_split(X_init, y_init, test_size=0.2, random_state=42)
X = np.array(X_main_tr[:n]) # training dataset
y = np.array(y_main_tr[:n])
X_ts = np.array(X_main_ts[n:n+50]) #testing dataset
y_ts = np.array(y_main_ts[n:n+50])
