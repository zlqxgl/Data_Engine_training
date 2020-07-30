import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#读取数据
with open ('订单表.csv') as f:
    order_list = pd.read_csv(f)

# 提取每一天每一位顾客购买的产品列表
transaction_list = []
order_group_by_date = order_list.groupby("订单日期")
for date, daily_order in order_group_by_date:
    daily_order_group_by_customer = daily_order.groupby("客户ID")
    for customer_id, customer_daily_order in daily_order_group_by_customer:
        transaction_list.append([str(el) for el in list(customer_daily_order["产品ID"])])
transaction_list = pd.DataFrame(transaction_list)
#print(transaction_list)

# 使用空字节填充NaN
dataset = transaction_list.fillna('')
#print(dataset)

# 将每一行数据组合成一列，用逗号分隔
dataset_combined = dataset[0]
for i in range(1, len(dataset.columns)):
    dataset_combined = dataset_combined + ',' + dataset[i]
dataset_combined = pd.DataFrame(dataset_combined)
dataset_combined.columns = ['transactions']
#print(dataset_combined)

# 将物品进行one hot编码
dataset = dataset_combined['transactions'].str.get_dummies(',')
#print(dataset)

# 使用mlxtend查找数据中的关联规则
frequent_itemsets = apriori(dataset, min_support=0.015, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support" , ascending=False).reset_index(drop=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
rules = rules.sort_values(by="lift" , ascending=False).reset_index(drop=True)
print("频繁项集：", frequent_itemsets)
print("关联规则：", rules)

