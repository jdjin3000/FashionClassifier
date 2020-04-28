import pandas as pd
from operator import itemgetter

df = pd.read_csv('pred.csv')
preprocessedList = []
'''
for index, row in df.iterrows():
	if index == 0:
		continue
	lists = list(row)[1:]
	for i in range(4):
		del lists[lists.index(max(lists))]
	criteria = max(lists)
	temp = [i if criteria <= i else 0 for i in list(row)[1:]]
	preprocessedList.append([row[0]] + temp)
'''
for index, row in df.iterrows():
	if index == 0:
		continue
	lists = list(row)[1:]
	for i in range(4):
		del lists[lists.index(max(lists))]
	criteria = max(lists)

	lists = list(row)[1:]
	lists = [round(i,3) if i >= criteria else 0 for i in lists]
	preprocessedList.append([row[0]] + lists)


my_df = pd.DataFrame(data=preprocessedList, columns=list(df.columns.values))
my_df = my_df.sort_values(by=list(my_df.columns.values)[1:], ascending=False)

my_df.to_csv('test2.csv', index=False)
print("Sorting Completed Successfully!")