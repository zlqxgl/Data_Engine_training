'''Action2: 统计全班的成绩
班里有5名同学，现在需要你用Python来统计下这些人在语文、英语、数学中的平均成绩、
最小成绩、最大成绩、方差、标准差。然后把这些人的总成绩排序，得出名次进行成绩输出'''
import numpy as np
import pandas as pd

df = pd.DataFrame(np.array([[68,65, 30], [95,76,98], [98,86,88],[90,88,77],[80,90,90]]), \
                  columns=['语文', '数学', '英语'], index=['张飞', '关羽', '刘备','典韦','许褚'])
print(df.mean())
print(df.max())
print(df.min())
print(df.var())
print(df.std())
df['sum']=np.sum(df,1)
print(df.sort_values(by='sum',ascending=False))

