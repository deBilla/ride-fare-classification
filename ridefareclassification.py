import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('train.csv')
print (df.shape)

df = df.fillna(0) 


X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
y = [1 if each == 'correct' else 0 for each in y]
X = X.drop(['pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon'],
                            axis=1)
X = X.replace(to_replace=np.nan, value=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier()


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

new_df = pd.read_csv('test.csv')

new_X = new_df.drop(['tripid', 'pickup_time', 'drop_time', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon'],
                            axis=1)

y_pred = clf.predict(new_X)
print (y_pred)

Y_results = pd.DataFrame(new_df['tripid'])
Y_results.insert(1, column='prediction', value=y_pred)
Y_results.to_csv('pred.csv', index=False)
