from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import matplotlib.pyplot as plt

# 数据加载
data = pd.read_csv('CarPrice_Assignment.csv',encoding="gbk")
#print(data.head())

#将aspiration和drivewheel两列具体类型，根据经验权重，转换为数字
data['aspiration']=data['aspiration'].replace(to_replace='std',value=2)
data['aspiration']=data['aspiration'].replace(to_replace='turbo',value=1)
data['drivewheel']=data['drivewheel'].replace(to_replace='rwd',value=0)
data['drivewheel']=data['drivewheel'].replace(to_replace='fwd',value=1)
data['drivewheel']=data['drivewheel'].replace(to_replace='4wd',value=2)
#抽取重要的特征构成新的数组
new_colums=['aspiration','drivewheel','wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio',\
            'stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']
train_x = data[new_colums]
#print(train_x.head(10))

# # 规范化到 [0,1] 空间
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
train_x=pd.DataFrame(train_x)
print(train_x.head(10))

#PCA降维到2维空间，观察是否聚集
pca=sklearnPCA(n_components=2)
transformed=pd.DataFrame(pca.fit_transform(train_x))
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(transformed[0],transformed[1])
plt.show()

#手肘法查找K的值
sse=[]
for k in range(1,11):
    kmeans= KMeans(n_clusters=k)
    kmeans.fit(train_x)
    sse.append(kmeans.inertia_)
x=range(1,11)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x,sse,'o-')
plt.show()

#K-means聚类并绘图
kmeans=KMeans(n_clusters=3)
kmeans.fit(train_x)
predict_y=kmeans.predict(train_x)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(transformed[predict_y==0][0],transformed[predict_y==0][1],label='Class0',c='red')
ax.scatter(transformed[predict_y==1][0],transformed[predict_y==1][1],label='Class1',c='blue')
ax.scatter(transformed[predict_y==2][0],transformed[predict_y==2][1],label='Class2',c='green')

plt.legend()
plt.show()

#合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'cluster id'},axis=1,inplace=True)
print(result.head())
# 将结果导出到CSV文件中
result.to_csv('CarPrice_clustered.csv',index=False)


