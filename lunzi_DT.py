#ID3决策树只能接收“定性值”作为特征

import pandas as pd
import numpy as np
from math import log

'''
def load():
	a = pd.DataFrame([	[ 0.5, 0.5 ],
						[ 0.7, 0.3 ],
						[ 2.3, 1.5 ],
						[ 2.8, 1.0 ] ])
	b = pd.Series([ 2, 2, 3, 3 ])
	
	c = pd.DataFrame([	[ 1.0, 0.1 ],
						[ 2.0, 2.2 ],
						[ 0.3, 0.5 ],
						[ 3.5, 2.0 ] ])
	d = pd.Series([ 2, 3, 2, 3 ])
	
	return a,b,c,d
'''
def load():
	a = pd.DataFrame([	[ 1, 1 ],
						[ 1, 3 ],
						[ 2, 1 ],
						[ 2, 2 ],
						[ 2, 4 ],
						[ 5, 3 ],
						
						[ 3, 2 ],
						[ 3, 3 ],
						[ 4, 1 ],
						[ 4, 3 ],
						[ 1, 4 ],
						[ 4, 4 ]	], 
						index=list('qwertyuiop[]'), columns=['m','n'])
	
	b = pd.Series([	'a', 'a', 'a', 'a', 'a', 'a',
					'b', 'b', 'b', 'b', 'b', 'b' ], 
					index=list('qwertyuiop[]'))
	
	c = pd.DataFrame([	[ 1, 2 ],
						[ 4, 2 ]  ], columns=['m','n'])
	
	d = pd.Series([ 'a', 'b' ])
	
	return a,b,c,d
	
x_train, y_train, x_vali, y_vali = load()

x,y=x_train,y_train

def entropy(s):
	s_times = s.value_counts()
	p = s_times.apply(float) / s_times.sum()
	log_p = p.apply(log, args=(2,))
	H = (-1) * sum( p * log_p )
	return H		

def create_tree(x, y):
	if x.shape[1] == 1:
		return y.value_counts().idxmax()
	if len(y.unique()) == 1:
		return y.unique()[0]

	Hs = x.apply(entropy, axis=0)
	best_feature = Hs.idxmax()
	tree = { best_feature: {} }
	
	for i in x[ best_feature ].unique():
		x_new = x[ x[ best_feature ] == i ]
		y_new = y[ x[ best_feature ] == i ]
		x_new = x_new.drop( best_feature, axis=1 )
		
		tree [ best_feature ] [ i ] = create_tree(x_new, y_new)
	return tree

def predict_series(s, tree):
	tree_new = tree.copy()
	while type(tree_new) == dict :
		column = tree_new.keys()[0]
		tree_new = tree_new [column] [ s[column] ]
	return tree_new

def predict_fun(x, tree):
	#y = x.T.apply( predict_series, args=(tree,) )
	y = []
	for i in x.index:
		s = x.loc[i,:]
		y += predict_series(s, tree)	
	return pd.Series(y)

class DT_classifier:
	def fit(self, x, y):
		self.tree = create_tree(x, y)	
	def predict(self, x):
		return predict_fun(x, self.tree)	
		
		
clf = DT_classifier()
clf.fit(x_train, y_train)		
y_pred = clf.predict(x_vali)		
		
		
		
		
	
	
	
	
	
	#def predict(self, x):
