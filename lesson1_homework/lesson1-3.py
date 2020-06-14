'''Action3: 对汽车质量数据进行统计
数据集：car_complain.csv
600条汽车质量投诉
Step1，数据加载
Step2，数据预处理
拆分problem类型 => 多个字段
Step3，数据统计
对数据进行探索：品牌投诉总数，车型投诉总数
哪个品牌的平均车型投诉最多
'''

# 对汽车投诉信息进行分析
import pandas as pd

#数据加载，读取文件
result = pd.read_csv('car_complain.csv')
#数据预处理，拆分problem类型，形成多个字段
result = result.drop('problem', 1).join(result.problem.str.get_dummies(','))

#数据统计
df = result.groupby(['brand'])['id'].agg(['count']).sort_values('count',ascending=False)
print("品牌投诉总数：")
print(df)
df2 = result.groupby(['car_model'])['id'].agg(['count']).sort_values('count',ascending=False)
print("车型投诉总数：")
print(df2)
df3 = result.groupby(['brand', 'car_model'])['id'].agg(['count']).groupby(['brand']).mean().sort_values('count', ascending=False)
print("平均车型投诉最多的品牌：")
print(df3)

