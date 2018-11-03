import random
import collections	
import numpy as np

map_list = []

# 計算距離
def calculate_distance(point1,point2):

	return np.linalg.norm(np.array(point1)-np.array(point2))

def KMeans_cluster(data_points,k,iterations):
	global map_list

	# 建立儲存哈希表
	map_list = [[None for i in data_points] for i in data_points]
	
	# 隨機生成中心點
	centers = random.sample(range(len(data_points)),k)

	# 建立對應點
	data_points_index = list(range(len(data_points)))

	# 訓練蝶代數
	for iteration in range(iterations):
		new_cluster = {center:[] for center in centers}

		for data_point_index in data_points_index:
			if data_point_index not in centers:

				min_value = float('inf')
				min_goal = None
				min_data_point = None

				for center in centers:
					if map_list[center][data_point_index]==None:
						map_list[center][data_point_index] = calculate_distance(data_points[data_point_index],data_points[center])
					if min_value>map_list[center][data_point_index]:
						min_value = map_list[center][data_point_index]
						goal_center = center
				new_cluster[goal_center].append(data_point_index)

		centers = []
		for key in new_cluster:
			avg_dis = np.mean([data_points[x] for x in ([key]+new_cluster[key])],0)

			min_center = min([(calculate_distance(data_points[x],avg_dis),x) for x in ([key]+new_cluster[key])])[1]
			centers.append(min_center)
		
	return new_cluster

if __name__ == '__main__':
	
	from sklearn import cluster, datasets

	# 讀入鳶尾花資料
	iris = datasets.load_iris()
	iris_X = iris.data

	new_cluster = KMeans_cluster(iris_X,6,10)




