import pandas as pd

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv'
df = pd.read_csv(url)
df.dropna() 

#manages wrong date format (month as str), splits it in year, month & day columns
df['Month'] = pd.to_datetime(df['Month'], format="%m-%d-%y")
df['Year'] = df['Month'].dt.year
df['Month_Num'] = df['Month'].dt.month
df['Day'] = df['Month'].dt.day

total_rows = len(df)
test_rows = 7 #last 7 days for testing
test_index = total_rows - test_rows #split index

df_train = df.iloc[:test_index] 
df_test = df.iloc[test_index:]

#splits dataframe in both data for training & testing usage
X_train = df_train[['Year','Month_Num','Day']]
y_train = df_train[['Sales']]
X_test = df_test[['Year','Month_Num','Day']]
y_test = df_test[['Sales']]

#model training & inference
from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

pred_df = df_test
pred_df['Predicted Sales'] = y_pred


#metrics & visual representation
"""import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from io import StringIO

metrics_names = ['Precision', 'Recall', 'F1-Score']
metrics_names1 = ['Precision 1', 'Recall 1', 'F1-Score 1']
metrics_names2 = ['Precision 2', 'Recall 2', 'F1-Score 2']

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
report = classification_report(y_test, y_pred)
df_report = pd.read_fwf(StringIO(report), index_col=0)
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
metrics_values = [np.mean(precision), np.mean(recall), np.mean(f1_score)] #calculates metrics tuples means
metrics_values1 = [precision[0], recall[0], f1_score[0]]
metrics_values2 = [precision[1], recall[1], f1_score[1]]
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: " + acc) #gets results accuracy

#tuples (class 1 & 2) splitting
df_metrics1 = pd.DataFrame({'Metrics': metrics_names1, 'Values': metrics_values1})
df_metrics2 = pd.DataFrame({'Metrics': metrics_names2, 'Values': metrics_values2})
fig = plt.figure(figsize =(10, 7))
plt.bar(metrics_names1, metrics_values1)
plt.bar(metrics_names2, metrics_values2)
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.ylim(0, 1.0)
sns.set(style="whitegrid")
plt.legend(['Class 1', 'Class 2'],loc='upper left')
plt.title('Classification Report from Shampoo Sales')
plt.show()"""