from pyspark import SparkConf, SparkContext
import random
import collections
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)




# 計算距離
def calculate_distance(point1,point2):
    return np.linalg.norm(np.array(point1)-np.array(point2))

# 重新聚類
def re_cluster(x,data_points,center,cluster_dict):
    # 找尋目標資料點的最近中心點
    goal_center = min([(calculate_distance(data_points[x],data_points[center_point]),center_point) for center_point in center])[1]
    
    # 儲存至Broadcast Variable
    cluster_dict.value[goal_center].append(x)
    
    return cluster_dict.value




def KMeans_cluster(data_points,k,iterations):
    
    # 隨機生成中心點
    center = random.sample(range(len(data_points)),k)
    
    
    
    # 訓練迭代數
    for iteration in range(iterations):
        
        # 建立對應點
        rdd = sc.parallelize(list(range(len(data_points))))
        
        # 去除中心點
        rdd = sc.parallelize(rdd.filter(lambda x:x not in center).collect())
        
        
        # 建立Broadcast Variable
        cluster_dict = {i:[] for i in center}
        cluster_dict = sc.broadcast(cluster_dict)
        
        # 依中心點開始聚類
        new_cluster = rdd.map(lambda j:re_cluster(j,data_points,center,cluster_dict)).collect()[-1]

        
        
        center = []
        
        for key in new_cluster:
            rdd = sc.parallelize(new_cluster[key]+[key])
            
            # 找尋平均值
            sum_dis = rdd.map(lambda x:data_points[x]).sum()
            average_dis = sum_dis/(len(new_cluster[key])+1)
            
            rdd = sc.parallelize(new_cluster[key]+[key])
            
            # 找尋新中心點
            min_center = min(rdd.map(lambda x:(calculate_distance(average_dis,data_points[x]),x)).collect())[-1]
            center.append(min_center)
        
        
    return new_cluster

if __name__ == '__main__':
	from sklearn import cluster, datasets

	# 讀入鳶尾花資料
	iris = datasets.load_iris()
	iris_X = iris.data

	# 進行聚類
	new_cluster = KMeans_cluster(iris_X,100,100)       
	
	# 以下為製圖
	fig = plt.figure(1, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	X_reduced = PCA(n_components=3).fit_transform(iris.data)

	from itertools import cycle
	cycol = cycle('bgrcmk')
	    
	for i in new_cluster:
	    
	    color = next(cycol)
	    for q in new_cluster[i]:
	        ax.scatter(X_reduced[q, 0], X_reduced[q, 1], X_reduced[q, 2],c=color,
	                   cmap=plt.cm.Set1, edgecolor='k', s=40)

	    ax.scatter(X_reduced[i,0],X_reduced[i,1],X_reduced[i,2],c=color,cmap = plt.cm.Set1,edgecolor='k',s=200)
	ax.set_title("First three PCA directions")
	ax.set_xlabel("1st eigenvector")
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("2nd eigenvector")
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel("3rd eigenvector")
	ax.w_zaxis.set_ticklabels([])

	plt.show()
	