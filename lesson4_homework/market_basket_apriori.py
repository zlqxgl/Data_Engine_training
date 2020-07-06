import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#读取文件
dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None) 
# 查看维度，shape为(7501,20)
print(dataset)

#将数据存放到transactions中
transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0, 20):
        if str(dataset.values[i, j]) != 'nan':
           temp.append(str(dataset.values[i, j]))
    transactions.append(temp)

# 对数据进行独热编码
temp = TransactionEncoder()
temp_hot_encoded = temp.fit_transform(transactions)
df = pd.DataFrame(temp_hot_encoded, columns=temp.columns_)
print(df.head())
df.to_csv('df.csv')

# 挖掘频繁项集
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print('频繁项集：',frequent_itemsets)
# 计算关联规则
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules = rules.sort_values(by='lift', ascending=False)
print('关联规则:',rules)
rules.to_csv('rules.csv')


