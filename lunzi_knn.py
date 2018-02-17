import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from math import sqrt

'''
class aa:
	def __init__(self, a=1, b=2):
		self.a = a
		self.b = b
	def bb(self):
		self.dd = 111
		print('dd')
	def cc(self):
		print(self.dd)
a = aa()
a.bb()
a.cc()

#############

		index_x_train = clf.index_x_train
		columns_x_train = clf.columns_x_train
		
		y_test = pd.Series([])
		x_test = x_vali
		
		for i in x_test.index:
			x_train['distance'] = ((x_train.loc[index_x_train, columns_x_train] - x_test.loc[i, :])**2).sum(axis=1).apply(sqrt)			
			x_train.sort_values('distance', ascending=True, inplace=True)
			#temp = pd.Series(Counter(x_train.iloc[:k,:]['distance']))
			temp = x_train.iloc[:k,:]['distance']	# 取距离最小的k个，temp是个series
			temp = y_train[temp.index]		# 取对应的label，temp是个series
			temp = pd.Series(Counter(temp))		# k个label里，每个label对应的个数
			label_pred = temp.idxmax()
			y_test[i] = label_pred
		return y_test
				
'''

class knn:
	def __init__(self, k=3):
		self.k = k
	def fit(self, x, y):
		self.x_train = pd.DataFrame(x)
		self.y_train = pd.Series(y)
		
		self.index_x_train = self.x_train.index
		self.columns_x_train = self.x_train.columns
		
		self.x_train['label'] = self.y_train
		
	def predict(self, x):
		self.x_test = pd.DataFrame(x)
		self.y_test = pd.Series([])

		for i in self.x_test.index:
			self.x_train['distance'] = ((self.x_train.iloc[self.index_x_train, self.columns_x_train] - self.x_test.loc[i, :])**2).sum(axis=1).apply(sqrt)
			self.x_train.sort_values('distance', ascending=True, inplace=True)
			self.temp = self.x_train.iloc[:self.k,:]['distance']	# 取距离最小的k个，temp是个series
			self.temp = self.y_train[self.temp.index]		# 取对应的label，temp是个series
			self.temp = pd.Series(Counter(self.temp))		# k个label里，每个label对应的个数
			self.label_pred = self.temp.idxmax()
			self.y_test[i] = self.label_pred
		return self.y_test
		
def data_set():
	x_train = pd.DataFrame([[1, 0], [1, 1], [3, 0], [4, 1]])
	x_vali = pd.DataFrame([[0, 0], [3, 1]])
	y_train = pd.Series(list('aabb'))
	y_vali = pd.Series(list('ab'))
	return x_train, y_train, x_vali, y_vali
	
x_train, y_train, x_vali, y_vali = data_set()

clf = knn()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_vali)










	
		
		
		
		
		
		
		
		
		
		
		
		
		
