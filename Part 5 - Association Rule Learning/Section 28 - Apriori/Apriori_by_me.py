import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Market_Basket_Optimisation.csv', header=None)
#As theres no title in the column, so we need to specify header=none
#creating a list of transactions
transaction=[]
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])

#training apriori on dataset

#directly importing apyori.py function
    
from apyori import apriori
rules=apriori(transaction, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

#visualising the results
result=list(rules)
